#pragma once
// Experimental weight-only compression primitives for C++17.
// This header is intentionally separate from gpt.cpp: it does not change
// training, checkpoint serialization, or the existing FP32 inference path.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace weight_loss {

struct Int4Matrix {
    int rows = 0;
    int cols = 0;
    // Two signed 4-bit values per byte, low nibble first; two's-complement.
    std::vector<std::uint8_t> packed;
    // One symmetric dequantization scale per output row.
    std::vector<float> row_scales;
};

inline int unpack_int4(const Int4Matrix& q, int index) {
    const std::uint8_t byte = q.packed.at(static_cast<std::size_t>(index / 2));
    const int nibble = (index % 2 == 0) ? (byte & 0x0f) : (byte >> 4);
    return nibble >= 8 ? nibble - 16 : nibble;
}

inline Int4Matrix quantize_int4(const float* weights, int rows, int cols) {
    if (!weights || rows <= 0 || cols <= 0)
        throw std::invalid_argument("INT4: weights and positive dimensions required");

    const std::size_t count = static_cast<std::size_t>(rows) * cols;
    Int4Matrix q;
    q.rows = rows;
    q.cols = cols;
    q.packed.assign((count + 1) / 2, 0);
    q.row_scales.assign(static_cast<std::size_t>(rows), 0.0f);

    for (int r = 0; r < rows; ++r) {
        float max_abs = 0.0f;
        for (int c = 0; c < cols; ++c) {
            const float v = weights[static_cast<std::size_t>(r) * cols + c];
            if (!std::isfinite(v))
                throw std::invalid_argument("INT4: weights must be finite");
            max_abs = std::max(max_abs, std::fabs(v));
        }
        const float scale = max_abs == 0.0f ? 0.0f : max_abs / 7.0f;
        q.row_scales[static_cast<std::size_t>(r)] = scale;
        for (int c = 0; c < cols; ++c) {
            const std::size_t i = static_cast<std::size_t>(r) * cols + c;
            int value = 0;
            if (scale != 0.0f) {
                value = static_cast<int>(std::round(weights[i] / scale));
                value = std::max(-7, std::min(7, value));
            }
            const std::uint8_t nibble = static_cast<std::uint8_t>(value) & 0x0f;
            if ((i & 1u) == 0) q.packed[i / 2] |= nibble;
            else q.packed[i / 2] |= static_cast<std::uint8_t>(nibble << 4);
        }
    }
    return q;
}

inline void matvec_int4(const Int4Matrix& weights, const float* x, float* y) {
    if (!x || !y || weights.rows <= 0 || weights.cols <= 0 ||
        weights.row_scales.size() != static_cast<std::size_t>(weights.rows) ||
        weights.packed.size() !=
            (static_cast<std::size_t>(weights.rows) * weights.cols + 1) / 2)
        throw std::invalid_argument("INT4: invalid matrix or vector");

    float max_abs = 0.0f;
    for (int c = 0; c < weights.cols; ++c) {
        if (!std::isfinite(x[c]))
            throw std::invalid_argument("INT4: activations must be finite");
        max_abs = std::max(max_abs, std::fabs(x[c]));
    }
    const float x_scale = max_abs == 0.0f ? 0.0f : max_abs / 127.0f;
    std::vector<std::int8_t> xq(static_cast<std::size_t>(weights.cols), 0);
    if (x_scale != 0.0f) {
        for (int c = 0; c < weights.cols; ++c) {
            const int v = static_cast<int>(std::round(x[c] / x_scale));
            xq[static_cast<std::size_t>(c)] =
                static_cast<std::int8_t>(std::max(-127, std::min(127, v)));
        }
    }

    for (int r = 0; r < weights.rows; ++r) {
        std::int32_t acc = 0;
        for (int c = 0; c < weights.cols; ++c) {
            const int w = unpack_int4(weights, r * weights.cols + c);
            acc += static_cast<std::int32_t>(w) * xq[static_cast<std::size_t>(c)];
        }
        y[r] = weights.row_scales[static_cast<std::size_t>(r)] * x_scale *
               static_cast<float>(acc);
    }
}

struct BinaryMatrix {
    int rows = 0;
    int cols = 0;
    // Bit 1 means non-negative; bit 0 means negative. Least-significant bit first.
    std::vector<std::uint32_t> packed;
    // Per-row mean absolute weight; keeps magnitude information.
    std::vector<float> row_scales;
};

inline int popcount32(std::uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
#else
    int count = 0;
    while (x) { x &= (x - 1u); ++count; }
    return count;
#endif
}

inline BinaryMatrix quantize_binary(const float* weights, int rows, int cols) {
    if (!weights || rows <= 0 || cols <= 0)
        throw std::invalid_argument("BIN1: weights and positive dimensions required");
    BinaryMatrix q;
    q.rows = rows;
    q.cols = cols;
    const std::size_t words_per_row = (static_cast<std::size_t>(cols) + 31) / 32;
    q.packed.assign(static_cast<std::size_t>(rows) * words_per_row, 0u);
    q.row_scales.assign(static_cast<std::size_t>(rows), 0.0f);

    for (int r = 0; r < rows; ++r) {
        double abs_sum = 0.0;
        for (int c = 0; c < cols; ++c) {
            const float v = weights[static_cast<std::size_t>(r) * cols + c];
            if (!std::isfinite(v))
                throw std::invalid_argument("BIN1: weights must be finite");
            abs_sum += std::fabs(v);
            if (v >= 0.0f) {
                const std::size_t word = static_cast<std::size_t>(r) * words_per_row +
                                         static_cast<std::size_t>(c) / 32;
                q.packed[word] |= (std::uint32_t{1} << (c % 32));
            }
        }
        q.row_scales[static_cast<std::size_t>(r)] =
            static_cast<float>(abs_sum / cols);
    }
    return q;
}

inline void matvec_binary(const BinaryMatrix& weights, const float* x, float* y) {
    if (!x || !y || weights.rows <= 0 || weights.cols <= 0 ||
        weights.row_scales.size() != static_cast<std::size_t>(weights.rows))
        throw std::invalid_argument("BIN1: invalid matrix or vector");
    const std::size_t words_per_row = (static_cast<std::size_t>(weights.cols) + 31) / 32;
    if (weights.packed.size() != static_cast<std::size_t>(weights.rows) * words_per_row)
        throw std::invalid_argument("BIN1: invalid packed data");

    double abs_sum = 0.0;
    for (int c = 0; c < weights.cols; ++c) {
        if (!std::isfinite(x[c]))
            throw std::invalid_argument("BIN1: activations must be finite");
        abs_sum += std::fabs(x[c]);
    }
    const float x_scale = static_cast<float>(abs_sum / weights.cols);
    for (int r = 0; r < weights.rows; ++r) {
        int signed_matches = 0;
        for (std::size_t w = 0; w < words_per_row; ++w) {
            const int remaining = weights.cols - static_cast<int>(w * 32);
            const int bits = std::min(32, remaining);
            const std::uint32_t mask = bits == 32 ? 0xffffffffu :
                                       ((std::uint32_t{1} << bits) - 1u);
            std::uint32_t x_signs = 0;
            for (int b = 0; b < bits; ++b) {
                if (x[static_cast<std::size_t>(w) * 32 + b] >= 0.0f)
                    x_signs |= (std::uint32_t{1} << b);
            }
            const std::uint32_t w_signs =
                weights.packed[static_cast<std::size_t>(r) * words_per_row + w];
            signed_matches += 2 * popcount32((~(w_signs ^ x_signs)) & mask) - bits;
        }
        y[r] = weights.row_scales[static_cast<std::size_t>(r)] * x_scale *
               static_cast<float>(signed_matches);
    }
}

} // namespace weight_loss
