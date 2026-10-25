#include "weight_loss.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static bool near(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

int main() {
    // INT4: odd element count exercises the unused high nibble.
    const float w[] = {1.0f, -2.0f, 0.0f, 0.5f, 3.0f, -1.0f};
    const auto q4 = weight_loss::quantize_int4(w, 2, 3);
    assert(q4.packed.size() == 3);
    assert(q4.row_scales.size() == 2);
    assert(weight_loss::unpack_int4(q4, 0) == 3);
    assert(weight_loss::unpack_int4(q4, 1) == -7);
    float x[] = {1.0f, -1.0f, 0.25f};
    float y4[2] = {};
    weight_loss::matvec_int4(q4, x, y4);
    assert(std::isfinite(y4[0]) && std::isfinite(y4[1]));

    const float zeros[] = {0.0f, 0.0f, 0.0f};
    const auto qzero = weight_loss::quantize_int4(zeros, 1, 3);
    float yz = 1.0f;
    weight_loss::matvec_int4(qzero, x, &yz);
    assert(yz == 0.0f);

    // BIN1: 35 columns exercises a partial final word and its mask.
    std::vector<float> wb(70);
    std::vector<float> xb(35);
    for (int i = 0; i < 70; ++i) wb[i] = (i % 3 == 0) ? -0.5f : 1.25f;
    for (int i = 0; i < 35; ++i) xb[i] = (i % 2 == 0) ? 0.75f : -2.0f;
    const auto qb = weight_loss::quantize_binary(wb.data(), 2, 35);
    float yb[2] = {};
    weight_loss::matvec_binary(qb, xb.data(), yb);
    assert(std::isfinite(yb[0]) && std::isfinite(yb[1]));

    const float bad[] = {1.0f, std::numeric_limits<float>::infinity()};
    bool rejected = false;
    try { (void)weight_loss::quantize_int4(bad, 1, 2); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);

    std::cout << "weight_loss tests passed\n";
    return 0;
}
