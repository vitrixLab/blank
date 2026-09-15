// =============================================================================
// gpt.cpp — Tiny character-level GPT in pure C++
// Architecture from AI.png:
//   n_embd     = 16
//   n_head     = 4    (d_k = 4 per head)
//   n_layer    = 4    (Transformer blocks)
//   vocab_size = 65   (character vocabulary)
//   block_size = 55   (context window)
//   Total params: 14,208 | Trainable matrices: 27
//
// Forward pass (section 1 of AI.png):
//   Input token (id)
//     → Token Embedding wte (65×16)      + wpe (55×16)
//     → RMSNorm
//     → [Transformer Block × 4]
//         ├─ 1. Attention sub-layer
//         │    RMSNorm → Q/K/V (16×16 each) → MHSA (n_head=4, d_k=4)
//         │    → wo (16×16) → residual add
//         └─ 2. MLP sub-layer
//              RMSNorm → fc1 (16×64) → ReLU → fc2 (64×16) → residual add
//     → Linear lm_head (16×65, tied to wte^T)
//     → Softmax → Cross-Entropy Loss
//
// Backward pass (section 2 of AI.png):
//   Loss ← Softmax ← lm_head ← Residual Add ←
//     MLP (fc2→ReLU→fc1→RMSNorm) ← Residual Add ←
//       Attention (wo→Softmax→Q/K/V→RMSNorm) ← … ← RMSNorm (input)
//     ← Token Embedding + Position Embedding
//
// Build:  g++ -O2 -o gpt gpt.cpp
// Train:  ./gpt train data.txt
// Gen:    ./gpt gen   model.bin "seed text"
// =============================================================================

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Hyper-parameters (match AI.png exactly)
// ─────────────────────────────────────────────────────────────────────────────
struct Config {
    int vocab_size = 65;   // character vocabulary
    int block_size = 55;   // context window (token sequence length)
    int n_embd     = 16;   // embedding dimension
    int n_head     = 4;    // attention heads
    int n_layer    = 4;    // transformer blocks
    // derived
    int d_k() const { return n_embd / n_head; } // 4 per head

    // Total params = 14208
    // = 4 * (4*16*16 + 16*64 + 64*16)  [blocks]
    // +  65*16   [wte]
    // +  55*16   [wpe]
    // lm_head is weight-tied to wte (same storage, counted once)
    // ─ Trainable matrices = 27:
    //   24 (4 blocks × 6 projections) + wte + wpe + lm_head = 27
};

// ─────────────────────────────────────────────────────────────────────────────
// 2.  Flat tensor helpers
// ─────────────────────────────────────────────────────────────────────────────
// We represent every tensor as a contiguous float vector.
// Conventions: matrix M[rows][cols] → M[r*cols + c]

static inline float& at(std::vector<float>& v, int r, int c, int cols)
{ return v[r * cols + c]; }
static inline float  at(const std::vector<float>& v, int r, int c, int cols)
{ return v[r * cols + c]; }

// ─────────────────────────────────────────────────────────────────────────────
// 3.  Random initialisation (Xavier-like for matrices, zero for others)
// ─────────────────────────────────────────────────────────────────────────────
static std::mt19937 rng(42);

static void xavier_init(std::vector<float>& w, int fan_in, int fan_out) {
    float scale = std::sqrt(2.0f / (fan_in + fan_out));
    std::normal_distribution<float> dist(0.0f, scale);
    for (auto& x : w) x = dist(rng);
}

static void zero(std::vector<float>& v) { std::fill(v.begin(), v.end(), 0.0f); }

// ─────────────────────────────────────────────────────────────────────────────
// 4.  Model parameters (weights) and their gradients
// ─────────────────────────────────────────────────────────────────────────────
// Named exactly as in the diagram (AI.png).
// Matrices: stored [out_features × in_features] (row = output neuron).

struct AttentionWeights {
    // Q, K, V projections: (n_embd × n_embd) = (16×16)
    std::vector<float> wq, wk, wv;   // [n_embd × n_embd]
    // Output projection wo: (n_embd × n_embd) = (16×16)
    std::vector<float> wo;            // [n_embd × n_embd]

    // Gradients (same shape)
    std::vector<float> dwq, dwk, dwv, dwo;

    // Adam moment estimates (1st and 2nd)
    std::vector<float> mwq, vwq, mwk, vwk, mwv, vwv, mwo, vwo;

    void init(int n_embd) {
        int sz = n_embd * n_embd;
        wq.resize(sz); wk.resize(sz); wv.resize(sz); wo.resize(sz);
        xavier_init(wq, n_embd, n_embd);
        xavier_init(wk, n_embd, n_embd);
        xavier_init(wv, n_embd, n_embd);
        xavier_init(wo, n_embd, n_embd);

        dwq.resize(sz,0); dwk.resize(sz,0); dwv.resize(sz,0); dwo.resize(sz,0);
        mwq.resize(sz,0); vwq.resize(sz,0);
        mwk.resize(sz,0); vwk.resize(sz,0);
        mwv.resize(sz,0); vwv.resize(sz,0);
        mwo.resize(sz,0); vwo.resize(sz,0);
    }
};

struct MLPWeights {
    // fc1: (4*n_embd × n_embd) = (64×16)
    // fc2: (n_embd × 4*n_embd) = (16×64)
    std::vector<float> fc1, fc2;      // [64×16], [16×64]
    std::vector<float> dfc1, dfc2;
    std::vector<float> mfc1, vfc1, mfc2, vfc2;

    void init(int n_embd) {
        int hidden = 4 * n_embd; // 64
        fc1.resize(hidden * n_embd); xavier_init(fc1, n_embd,  hidden);
        fc2.resize(n_embd * hidden); xavier_init(fc2, hidden, n_embd);
        dfc1.resize(hidden*n_embd,0); dfc2.resize(n_embd*hidden,0);
        mfc1.resize(hidden*n_embd,0); vfc1.resize(hidden*n_embd,0);
        mfc2.resize(n_embd*hidden,0); vfc2.resize(n_embd*hidden,0);
    }
};

struct BlockWeights {
    AttentionWeights attn;
    MLPWeights       mlp;
    void init(int n_embd) { attn.init(n_embd); mlp.init(n_embd); }
};

struct ModelWeights {
    // Token embedding wte   [vocab_size × n_embd] = [65 × 16]
    std::vector<float> wte;
    std::vector<float> dwte;
    std::vector<float> mwte, vwte;

    // Position embedding wpe [block_size × n_embd] = [55 × 16]
    std::vector<float> wpe;
    std::vector<float> dwpe;
    std::vector<float> mwpe, vwpe;

    // lm_head: weight-tied to wte (same pointer semantics; we share wte for the
    // output projection and just add gradients back to dwte).

    // Transformer blocks × 4
    std::vector<BlockWeights> blocks;

    Config cfg;

    void init(const Config& c) {
        cfg = c;
        int E = c.n_embd, V = c.vocab_size, T = c.block_size;

        wte.resize(V*E); xavier_init(wte, E, V);
        dwte.resize(V*E,0); mwte.resize(V*E,0); vwte.resize(V*E,0);

        wpe.resize(T*E); xavier_init(wpe, E, T);
        dwpe.resize(T*E,0); mwpe.resize(T*E,0); vwpe.resize(T*E,0);

        blocks.resize(c.n_layer);
        for (auto& b : blocks) b.init(E);
    }

    // Count parameters (matches "Total Parameters: 14,208" in AI.png)
    int count_params() const {
        int E = cfg.n_embd, V = cfg.vocab_size, T = cfg.block_size;
        int layer_params = 4 * (4*E*E + 2*(4*E)*E); // 4 * (Q+K+V+wo) * E*E + fc1+fc2
        // = 4 * (4*16*16 + 16*64 + 64*16) = 4 * 3072 = 12288
        return layer_params + V*E + T*E; // + wte + wpe (lm_head is tied)
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 5.  RMSNorm  (no learnable scale — consistent with 27 matrices in AI.png)
// ─────────────────────────────────────────────────────────────────────────────
// Forward:  y[i] = x[i] / rms(x),  rms = sqrt(mean(x^2) + eps)
static void rmsnorm_fwd(const float* x, float* y, int n, float eps = 1e-5f) {
    float ms = 0.0f;
    for (int i = 0; i < n; i++) ms += x[i]*x[i];
    ms = ms / n + eps;
    float rms_inv = 1.0f / std::sqrt(ms);
    for (int i = 0; i < n; i++) y[i] = x[i] * rms_inv;
}

// Backward:  dx += dy/rms * (1 - y*y/n)   (accumulated into dx)
static void rmsnorm_bwd(const float* x, const float* dy, float* dx, int n, float eps = 1e-5f) {
    float ms = 0.0f;
    for (int i = 0; i < n; i++) ms += x[i]*x[i];
    ms = ms / n + eps;
    float rms = std::sqrt(ms);
    float rms_inv = 1.0f / rms;

    // d(rms_inv)/dms  = -1/(2 * ms^1.5)
    // dy_dot_y = sum(dy * y)   where y = x * rms_inv
    float dy_dot_y = 0.0f;
    for (int i = 0; i < n; i++) dy_dot_y += dy[i] * (x[i] * rms_inv);

    for (int i = 0; i < n; i++) {
        // dx[i] += rms_inv * dy[i]  −  x[i] * dy_dot_y / (n * ms)
        dx[i] += rms_inv * dy[i] - x[i] * dy_dot_y / (float(n) * ms);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6.  Softmax (in-place, over a length-n array)
// ─────────────────────────────────────────────────────────────────────────────
static void softmax(float* x, int n) {
    float m = *std::max_element(x, x+n);
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

// ─────────────────────────────────────────────────────────────────────────────
// 7.  Activation caches (store intermediates needed for backward pass)
// ─────────────────────────────────────────────────────────────────────────────
// Each token position t ∈ [0, T) passes through the network independently
// (no batch dimension — we process one sequence at a time for clarity).

struct AttentionCache {
    // Inputs / intermediates for one block's attention sub-layer
    std::vector<float> x_in;       // [T×E]  input  (before RMSNorm)
    std::vector<float> xn;         // [T×E]  after  RMSNorm
    std::vector<float> Q, K, V;    // [T×E]  projected Q/K/V
    std::vector<float> att;        // [T×T]  attention weights (post-softmax)
    std::vector<float> att_pre;    // [T×T]  pre-softmax scores (for bwd)
    std::vector<float> sv;         // [T×E]  sum_v (weighted V)
    std::vector<float> y;          // [T×E]  output of wo
};

struct MLPCache {
    std::vector<float> x_in;       // [T×E]  input  (before RMSNorm)
    std::vector<float> xn;         // [T×E]  after  RMSNorm
    std::vector<float> h;          // [T×H]  after fc1  (H = 4*E = 64)
    std::vector<float> h_relu;     // [T×H]  after ReLU
    std::vector<float> y;          // [T×E]  after fc2
};

struct BlockCache {
    AttentionCache attn;
    MLPCache       mlp;
};

struct ForwardCache {
    std::vector<float>      embed;   // [T×E]  wte+wpe
    std::vector<float>      embed_n; // [T×E]  after first RMSNorm
    std::vector<BlockCache> blocks;  // one per transformer block
    std::vector<float>      logits;  // [T×V]  lm_head output (pre-softmax)
    std::vector<float>      probs;   // [T×V]  after softmax
};

// ─────────────────────────────────────────────────────────────────────────────
// 8.  Matrix × vector  utilities
// ─────────────────────────────────────────────────────────────────────────────
// y = W * x     (W: [out × in], x: [in], y: [out])
static void matvec(const float* W, const float* x, float* y, int out, int in) {
    for (int o = 0; o < out; o++) {
        float acc = 0.0f;
        for (int i = 0; i < in; i++) acc += W[o*in + i] * x[i];
        y[o] = acc;
    }
}

// Accumulate  dW += dy * x^T   and   dx += W^T * dy
static void matvec_bwd(
    const float* W, const float* x,
    const float* dy, float* dW, float* dx,
    int out, int in)
{
    for (int o = 0; o < out; o++) {
        for (int i = 0; i < in; i++) {
            dW[o*in + i] += dy[o] * x[i];
        }
    }
    if (dx) {
        for (int i = 0; i < in; i++) {
            float acc = 0.0f;
            for (int o = 0; o < out; o++) acc += W[o*in + i] * dy[o];
            dx[i] += acc;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9.  Multi-Head Self-Attention  (forward)
// ─────────────────────────────────────────────────────────────────────────────
// Implements the diagram's ATTENTION SUB-LAYER:
//   xn → Q=xn·Wq, K=xn·Wk, V=xn·Wv
//   split into n_head heads (d_k = 4)
//   scores = Q·K^T / sqrt(d_k), causal mask, softmax
//   weighted sum of V → concat heads → wo → output
static void attention_fwd(
    const Config& cfg,
    const AttentionWeights& W,
    const std::vector<float>& x_in, // [T×E]
    AttentionCache& cache)
{
    int T = cfg.block_size, E = cfg.n_embd;
    int H = cfg.n_head, dk = cfg.d_k();

    cache.x_in = x_in;
    cache.xn.resize(T*E); cache.Q.resize(T*E); cache.K.resize(T*E);
    cache.V.resize(T*E);  cache.att.resize(T*T,0); cache.att_pre.resize(T*T,0);
    cache.sv.resize(T*E,0); cache.y.resize(T*E,0);

    // RMSNorm per token
    for (int t = 0; t < T; t++)
        rmsnorm_fwd(&x_in[t*E], &cache.xn[t*E], E);

    // Project Q, K, V for all tokens
    for (int t = 0; t < T; t++) {
        matvec(W.wq.data(), &cache.xn[t*E], &cache.Q[t*E], E, E);
        matvec(W.wk.data(), &cache.xn[t*E], &cache.K[t*E], E, E);
        matvec(W.wv.data(), &cache.xn[t*E], &cache.V[t*E], E, E);
    }

    float scale = 1.0f / std::sqrt((float)dk);

    // Per-head scaled dot-product attention
    zero(cache.sv);
    for (int h = 0; h < H; h++) {
        int hoff = h * dk;

        // Compute scores[t1][t2] = Q[t1,h] · K[t2,h] * scale  (causal mask)
        for (int t1 = 0; t1 < T; t1++) {
            for (int t2 = 0; t2 < T; t2++) {
                float score = 0.0f;
                for (int d = 0; d < dk; d++)
                    score += cache.Q[t1*E + hoff+d] * cache.K[t2*E + hoff+d];
                score *= scale;
                // Causal masking: future positions → -inf
                cache.att_pre[t1*T + t2] = (t2 <= t1) ? score : -1e9f;
            }
            // Softmax over t2 for this t1
            float* row = &cache.att_pre[t1*T];
            // compute softmax into cache.att[t1*T + ...]
            float* arow = &cache.att[t1*T];
            float m = row[0];
            for (int t2 = 0; t2 < T; t2++) m = std::max(m, row[t2]);
            float s = 0.0f;
            for (int t2 = 0; t2 <= t1; t2++) { arow[t2] = std::exp(row[t2]-m); s += arow[t2]; }
            for (int t2 = t1+1; t2 < T; t2++) arow[t2] = 0.0f;
            for (int t2 = 0; t2 <= t1; t2++) arow[t2] /= s;
        }

        // Weighted sum of V: sv[t1, h*dk .. (h+1)*dk] = sum_t2 att[t1,t2]*V[t2, h*dk..]
        for (int t1 = 0; t1 < T; t1++) {
            for (int d = 0; d < dk; d++) {
                float acc = 0.0f;
                for (int t2 = 0; t2 <= t1; t2++)
                    acc += cache.att[t1*T+t2] * cache.V[t2*E + hoff+d];
                cache.sv[t1*E + hoff+d] = acc;
            }
        }
    }

    // Output projection: y = sv · wo
    for (int t = 0; t < T; t++)
        matvec(W.wo.data(), &cache.sv[t*E], &cache.y[t*E], E, E);
}

// ─────────────────────────────────────────────────────────────────────────────
// 10.  Multi-Head Self-Attention  (backward)
// ─────────────────────────────────────────────────────────────────────────────
// Returns dx_in gradient (accumulated).
static void attention_bwd(
    const Config& cfg,
    const AttentionWeights& W,
    AttentionWeights& dW,
    const AttentionCache& cache,
    const std::vector<float>& dy,   // [T×E]  gradient of output y
    std::vector<float>& dx_in)      // [T×E]  gradient accumulator for x_in
{
    int T = cfg.block_size, E = cfg.n_embd;
    int H = cfg.n_head, dk = cfg.d_k();

    // 1. grad through output projection wo
    std::vector<float> dsv(T*E, 0.0f);
    std::vector<float> dxn(T*E, 0.0f);

    for (int t = 0; t < T; t++) {
        // dsv[t] += wo^T * dy[t],  dwo += dy[t] * sv[t]^T
        matvec_bwd(W.wo.data(), &cache.sv[t*E],
                   &dy[t*E], dW.dwo.data(), &dsv[t*E], E, E);
    }

    float scale = 1.0f / std::sqrt((float)dk);

    // 2. grad through attention + weighted sum
    std::vector<float> dQ(T*E, 0.0f), dK(T*E, 0.0f), dV(T*E, 0.0f);
    std::vector<float> datt(T*T, 0.0f);

    for (int h = 0; h < H; h++) {
        int hoff = h * dk;

        // dV: dV[t2, h] += sum_t1 att[t1,t2] * dsv[t1, h]
        for (int t2 = 0; t2 < T; t2++) {
            for (int d = 0; d < dk; d++) {
                float acc = 0.0f;
                for (int t1 = t2; t1 < T; t1++)  // t2 <= t1 (causal)
                    acc += cache.att[t1*T + t2] * dsv[t1*E + hoff+d];
                dV[t2*E + hoff+d] += acc;
            }
        }

        // datt[t1,t2] = sum_d dsv[t1,h,d] * V[t2,h,d]
        for (int t1 = 0; t1 < T; t1++) {
            for (int t2 = 0; t2 <= t1; t2++) {
                float acc = 0.0f;
                for (int d = 0; d < dk; d++)
                    acc += dsv[t1*E + hoff+d] * cache.V[t2*E + hoff+d];
                datt[t1*T + t2] = acc;
            }
        }

        // Backward through softmax → pre-softmax scores
        for (int t1 = 0; t1 < T; t1++) {
            const float* a  = &cache.att[t1*T];
            float*       da = &datt[t1*T];
            // Jacobian of softmax: d(loss)/d(score_j) = a_j*(da_j - sum_k a_k*da_k)
            float dot = 0.0f;
            for (int t2 = 0; t2 <= t1; t2++) dot += a[t2] * da[t2];
            for (int t2 = 0; t2 <= t1; t2++) da[t2] = a[t2] * (da[t2] - dot);
            // future tokens have zero gradient (masked)
            for (int t2 = t1+1; t2 < T; t2++) da[t2] = 0.0f;
        }

        // grad through Q·K^T * scale
        // dQ[t1,h] += scale * sum_t2 da[t1,t2] * K[t2,h]
        // dK[t2,h] += scale * sum_t1 da[t1,t2] * Q[t1,h]
        for (int t1 = 0; t1 < T; t1++) {
            for (int d = 0; d < dk; d++) {
                float acc = 0.0f;
                for (int t2 = 0; t2 <= t1; t2++)
                    acc += datt[t1*T + t2] * cache.K[t2*E + hoff+d];
                dQ[t1*E + hoff+d] += scale * acc;
            }
        }
        for (int t2 = 0; t2 < T; t2++) {
            for (int d = 0; d < dk; d++) {
                float acc = 0.0f;
                for (int t1 = t2; t1 < T; t1++)  // causal: t1 >= t2
                    acc += datt[t1*T + t2] * cache.Q[t1*E + hoff+d];
                dK[t2*E + hoff+d] += scale * acc;
            }
        }
    }

    // 3. grad through Q/K/V linear projections → back to xn
    for (int t = 0; t < T; t++) {
        matvec_bwd(W.wq.data(), &cache.xn[t*E], &dQ[t*E], dW.dwq.data(), &dxn[t*E], E, E);
        matvec_bwd(W.wk.data(), &cache.xn[t*E], &dK[t*E], dW.dwk.data(), &dxn[t*E], E, E);
        matvec_bwd(W.wv.data(), &cache.xn[t*E], &dV[t*E], dW.dwv.data(), &dxn[t*E], E, E);
    }

    // 4. grad through RMSNorm → accumulate into dx_in
    for (int t = 0; t < T; t++)
        rmsnorm_bwd(&cache.x_in[t*E], &dxn[t*E], &dx_in[t*E], E);
}

// ─────────────────────────────────────────────────────────────────────────────
// 11.  MLP  (forward)
// ─────────────────────────────────────────────────────────────────────────────
// MLP sub-layer: RMSNorm → fc1 (16→64) → ReLU → fc2 (64→16) → output
static void mlp_fwd(
    const Config& cfg,
    const MLPWeights& W,
    const std::vector<float>& x_in, // [T×E]
    MLPCache& cache)
{
    int T = cfg.block_size, E = cfg.n_embd, Hid = 4*E;

    cache.x_in = x_in;
    cache.xn.resize(T*E);
    cache.h.resize(T*Hid);
    cache.h_relu.resize(T*Hid);
    cache.y.resize(T*E);

    for (int t = 0; t < T; t++) {
        // RMSNorm
        rmsnorm_fwd(&x_in[t*E], &cache.xn[t*E], E);
        // fc1
        matvec(W.fc1.data(), &cache.xn[t*E], &cache.h[t*Hid], Hid, E);
        // ReLU
        for (int i = 0; i < Hid; i++)
            cache.h_relu[t*Hid+i] = std::max(0.0f, cache.h[t*Hid+i]);
        // fc2
        matvec(W.fc2.data(), &cache.h_relu[t*Hid], &cache.y[t*E], E, Hid);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 12.  MLP  (backward)
// ─────────────────────────────────────────────────────────────────────────────
static void mlp_bwd(
    const Config& cfg,
    const MLPWeights& W,
    MLPWeights& dW,
    const MLPCache& cache,
    const std::vector<float>& dy,   // [T×E]
    std::vector<float>& dx_in)      // [T×E] grad accumulator
{
    int T = cfg.block_size, E = cfg.n_embd, Hid = 4*E;

    std::vector<float> dh_relu(T*Hid, 0.0f);
    std::vector<float> dh(T*Hid, 0.0f);
    std::vector<float> dxn(T*E, 0.0f);

    for (int t = 0; t < T; t++) {
        // grad through fc2
        matvec_bwd(W.fc2.data(), &cache.h_relu[t*Hid],
                   &dy[t*E], dW.dfc2.data(), &dh_relu[t*Hid], E, Hid);
        // grad through ReLU
        for (int i = 0; i < Hid; i++)
            dh[t*Hid+i] = (cache.h[t*Hid+i] > 0.0f) ? dh_relu[t*Hid+i] : 0.0f;
        // grad through fc1
        matvec_bwd(W.fc1.data(), &cache.xn[t*E],
                   &dh[t*Hid], dW.dfc1.data(), &dxn[t*E], Hid, E);
        // grad through RMSNorm
        rmsnorm_bwd(&cache.x_in[t*E], &dxn[t*E], &dx_in[t*E], E);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 13.  Full forward pass
// ─────────────────────────────────────────────────────────────────────────────
static float forward(
    const Config& cfg,
    const ModelWeights& W,
    const std::vector<int>& tokens, // [T] integer token ids
    ForwardCache& cache,
    bool compute_loss = true)
{
    int T = cfg.block_size, E = cfg.n_embd, V = cfg.vocab_size;

    // ── Embed: wte + wpe ──────────────────────────────────────────────────────
    cache.embed.resize(T*E);
    for (int t = 0; t < T; t++) {
        int id = tokens[t];
        for (int e = 0; e < E; e++)
            cache.embed[t*E+e] = W.wte[id*E+e] + W.wpe[t*E+e];
    }

    // ── Input RMSNorm ─────────────────────────────────────────────────────────
    cache.embed_n.resize(T*E);
    for (int t = 0; t < T; t++)
        rmsnorm_fwd(&cache.embed[t*E], &cache.embed_n[t*E], E);

    // ── Transformer Blocks × 4 ───────────────────────────────────────────────
    cache.blocks.resize(cfg.n_layer);
    std::vector<float> x = cache.embed_n;  // current activations [T×E]

    for (int l = 0; l < cfg.n_layer; l++) {
        auto& bc = cache.blocks[l];

        // Attention sub-layer + residual
        attention_fwd(cfg, W.blocks[l].attn, x, bc.attn);
        std::vector<float> x_after_attn(T*E);
        for (int i = 0; i < T*E; i++)
            x_after_attn[i] = x[i] + bc.attn.y[i];  // residual add

        // MLP sub-layer + residual
        mlp_fwd(cfg, W.blocks[l].mlp, x_after_attn, bc.mlp);
        std::vector<float> x_after_mlp(T*E);
        for (int i = 0; i < T*E; i++)
            x_after_mlp[i] = x_after_attn[i] + bc.mlp.y[i];  // residual add

        x = x_after_mlp;
    }

    // ── lm_head (weight-tied to wte): logits = x · wte^T ────────────────────
    // lm_head shape: (V × E) — we reuse W.wte as the weight matrix
    cache.logits.resize(T*V);
    for (int t = 0; t < T; t++) {
        for (int v = 0; v < V; v++) {
            float acc = 0.0f;
            for (int e = 0; e < E; e++)
                acc += x[t*E+e] * W.wte[v*E+e];  // wte[v] is the v-th row
            cache.logits[t*V+v] = acc;
        }
    }

    // ── Softmax over vocabulary ───────────────────────────────────────────────
    cache.probs.resize(T*V);
    for (int t = 0; t < T; t++) {
        std::copy(&cache.logits[t*V], &cache.logits[t*V+V], &cache.probs[t*V]);
        softmax(&cache.probs[t*V], V);
    }

    // ── Cross-Entropy Loss (next-token prediction) ────────────────────────────
    if (!compute_loss) return 0.0f;

    float loss = 0.0f;
    for (int t = 0; t < T-1; t++) {
        int target = tokens[t+1];
        loss -= std::log(cache.probs[t*V + target] + 1e-9f);
    }
    return loss / float(T-1);
}

// ─────────────────────────────────────────────────────────────────────────────
// 14.  Full backward pass
// ─────────────────────────────────────────────────────────────────────────────
// gradient of cross-entropy + softmax combined:
//   d(loss)/d(logits[t,v]) = probs[t,v] - 1{v == target[t]}
static void backward(
    const Config& cfg,
    ModelWeights& W,
    const std::vector<int>& tokens,
    const ForwardCache& cache)
{
    int T = cfg.block_size, E = cfg.n_embd, V = cfg.vocab_size;

    // ── Grad of loss w.r.t. logits (softmax + cross-entropy combined) ─────────
    // Backward flows from LOSS → Softmax → Logits (lm_head)
    std::vector<float> dlogits(T*V, 0.0f);
    float scale_loss = 1.0f / float(T-1);
    for (int t = 0; t < T-1; t++) {
        int target = tokens[t+1];
        for (int v = 0; v < V; v++)
            dlogits[t*V+v] = scale_loss * cache.probs[t*V+v];
        dlogits[t*V + target] -= scale_loss;
    }

    // Reconstruct final x (output of last block) for lm_head backward
    // We need the activations that were input to lm_head.
    // These are the same as what went into the last block's mlp residual add.
    // We re-derive from cache by forward-reconstructing:
    // (In a real implementation store this; here we re-compute for clarity.)
    std::vector<float> x_last(T*E);
    {
        std::vector<float> x = cache.embed_n;
        for (int l = 0; l < cfg.n_layer; l++) {
            auto& bc = cache.blocks[l];
            std::vector<float> xa(T*E);
            for (int i = 0; i < T*E; i++) xa[i] = x[i] + bc.attn.y[i];
            std::vector<float> xm(T*E);
            for (int i = 0; i < T*E; i++) xm[i] = xa[i] + bc.mlp.y[i];
            x = xm;
        }
        x_last = x;
    }

    // ── Grad through lm_head (tied to wte) ────────────────────────────────────
    // lm_head: logits[t,v] = dot(x[t], wte[v])
    // d(loss)/d(wte[v,e]) += sum_t dlogits[t,v] * x_last[t,e]   (lm_head part)
    // d(loss)/d(x_last[t,e]) = sum_v dlogits[t,v] * wte[v,e]    (back to x)
    std::vector<float> dx_last(T*E, 0.0f);

    for (int t = 0; t < T; t++) {
        for (int v = 0; v < V; v++) {
            float dl = dlogits[t*V+v];
            for (int e = 0; e < E; e++) {
                W.dwte[v*E+e] += dl * x_last[t*E+e];   // lm_head weight grad
                dx_last[t*E+e] += dl * W.wte[v*E+e];   // back-prop to x
            }
        }
    }

    // ── Backward through transformer blocks (reverse order) ───────────────────
    // As shown in AI.png section 2: "reverse order of forward operations"
    std::vector<float> dx = dx_last;

    for (int l = cfg.n_layer - 1; l >= 0; l--) {
        auto& bc = cache.blocks[l];

        // Grad through MLP residual add: dx splits to MLP path and residual path
        std::vector<float> dx_mlp_in = dx;         // goes into MLP backward
        std::vector<float> dx_residual_mlp = dx;   // identity residual

        std::vector<float> dx_after_attn(T*E, 0.0f);
        // residual pass-through
        for (int i = 0; i < T*E; i++) dx_after_attn[i] = dx_residual_mlp[i];
        // MLP backward: adds gradient into dx_after_attn
        mlp_bwd(cfg, W.blocks[l].mlp, W.blocks[l].mlp /* grad stored in weights */,
                bc.mlp, dx_mlp_in, dx_after_attn);

        // Grad through attention residual add
        std::vector<float> dx_attn_in = dx_after_attn;    // goes into attn bwd
        std::vector<float> dx_residual_attn = dx_after_attn; // identity

        std::vector<float> dx_block_in(T*E, 0.0f);
        for (int i = 0; i < T*E; i++) dx_block_in[i] = dx_residual_attn[i];
        // Attention backward: adds gradient into dx_block_in
        attention_bwd(cfg, W.blocks[l].attn, W.blocks[l].attn /* grad in weights */,
                      bc.attn, dx_attn_in, dx_block_in);

        dx = dx_block_in;
    }

    // ── Grad through input RMSNorm → embed ────────────────────────────────────
    std::vector<float> d_embed(T*E, 0.0f);
    for (int t = 0; t < T; t++)
        rmsnorm_bwd(&cache.embed[t*E], &dx[t*E], &d_embed[t*E], E);

    // ── Grad through wte (embedding lookup) and wpe ───────────────────────────
    // d(wte[id,e]) += d_embed[t,e] for each (t, id=tokens[t])
    for (int t = 0; t < T; t++) {
        int id = tokens[t];
        for (int e = 0; e < E; e++) {
            W.dwte[id*E+e] += d_embed[t*E+e];
            W.dwpe[t*E+e]  += d_embed[t*E+e];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 15.  Adam optimiser  (β₁=0.9, β₂=0.999, ε=1e-8)
// ─────────────────────────────────────────────────────────────────────────────
struct Adam {
    float lr = 3e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    int   step = 0;

    void update(float* w, float* g, float* m, float* v, int n) {
        float b1t = std::pow(b1, step);
        float b2t = std::pow(b2, step);
        float alpha = lr * std::sqrt(1.0f - b2t) / (1.0f - b1t);
        for (int i = 0; i < n; i++) {
            m[i] = b1 * m[i] + (1.0f - b1) * g[i];
            v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
            w[i] -= alpha * m[i] / (std::sqrt(v[i]) + eps);
            g[i]  = 0.0f; // zero grad after update
        }
    }

    void update_all(ModelWeights& W) {
        ++step;
        int E = W.cfg.n_embd, V = W.cfg.vocab_size, T = W.cfg.block_size;
        update(W.wte.data(), W.dwte.data(), W.mwte.data(), W.vwte.data(), V*E);
        update(W.wpe.data(), W.dwpe.data(), W.mwpe.data(), W.vwpe.data(), T*E);
        for (auto& b : W.blocks) {
            auto& a  = b.attn;
            auto& m  = b.mlp;
            int  sz  = E*E;
            int  szf = 4*E*E;
            update(a.wq.data(), a.dwq.data(), a.mwq.data(), a.vwq.data(), sz);
            update(a.wk.data(), a.dwk.data(), a.mwk.data(), a.vwk.data(), sz);
            update(a.wv.data(), a.dwv.data(), a.mwv.data(), a.vwv.data(), sz);
            update(a.wo.data(), a.dwo.data(), a.mwo.data(), a.vwo.data(), sz);
            update(m.fc1.data(), m.dfc1.data(), m.mfc1.data(), m.vfc1.data(), szf);
            update(m.fc2.data(), m.dfc2.data(), m.mfc2.data(), m.vfc2.data(), szf);
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 16.  Character tokeniser
// ─────────────────────────────────────────────────────────────────────────────
struct Tokenizer {
    std::unordered_map<char,int> c2i;
    std::vector<char>            i2c;

    void build(const std::string& text) {
        std::vector<bool> seen(256, false);
        for (char c : text) seen[(unsigned char)c] = true;
        for (int i = 0; i < 256; i++) {
            if (seen[i]) { c2i[(char)i] = (int)i2c.size(); i2c.push_back((char)i); }
        }
    }

    int  vocab_size() const { return (int)i2c.size(); }
    int  encode(char c) const { auto it = c2i.find(c); return it==c2i.end()?0:it->second; }
    char decode(int i) const  { return i >= 0 && i < (int)i2c.size() ? i2c[i] : '?'; }

    std::vector<int> encode(const std::string& s) const {
        std::vector<int> ids(s.size());
        for (size_t i = 0; i < s.size(); i++) ids[i] = encode(s[i]);
        return ids;
    }
    std::string decode(const std::vector<int>& ids) const {
        std::string s(ids.size(), '?');
        for (size_t i = 0; i < ids.size(); i++) s[i] = decode(ids[i]);
        return s;
    }

    void save(const std::string& path) const {
        std::ofstream f(path); f << std::string(i2c.begin(), i2c.end());
    }
    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::string s; std::getline(f, s, '\0');
        for (int i = 0; i < (int)s.size(); i++) { c2i[s[i]] = i; i2c.push_back(s[i]); }
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// 17.  Model save / load  (simple binary format)
// ─────────────────────────────────────────────────────────────────────────────
static void save_weights(const ModelWeights& W, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    // Write config
    f.write((char*)&W.cfg, sizeof(Config));
    // Write wte, wpe
    f.write((char*)W.wte.data(), W.wte.size() * sizeof(float));
    f.write((char*)W.wpe.data(), W.wpe.size() * sizeof(float));
    // Write blocks
    for (const auto& b : W.blocks) {
        auto wr = [&](const std::vector<float>& v){ f.write((char*)v.data(), v.size()*sizeof(float)); };
        wr(b.attn.wq); wr(b.attn.wk); wr(b.attn.wv); wr(b.attn.wo);
        wr(b.mlp.fc1); wr(b.mlp.fc2);
    }
    std::printf("[save] wrote %s\n", path.c_str());
}

static bool load_weights(ModelWeights& W, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    Config cfg; f.read((char*)&cfg, sizeof(Config));
    W.init(cfg);
    auto rd = [&](std::vector<float>& v){ f.read((char*)v.data(), v.size()*sizeof(float)); };
    rd(W.wte); rd(W.wpe);
    for (auto& b : W.blocks) {
        rd(b.attn.wq); rd(b.attn.wk); rd(b.attn.wv); rd(b.attn.wo);
        rd(b.mlp.fc1); rd(b.mlp.fc2);
    }
    std::printf("[load] loaded %s\n", path.c_str());
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// 18.  Text generation  (greedy or temperature-based sampling)
// ─────────────────────────────────────────────────────────────────────────────
static std::string generate(
    const Config& cfg,
    const ModelWeights& W,
    const Tokenizer& tok,
    const std::string& prompt,
    int max_new_tokens = 200,
    float temperature  = 1.0f)
{
    int T = cfg.block_size, V = cfg.vocab_size;

    // Seed context with prompt (right-pad or truncate to T)
    std::vector<int> ctx(T, 0);
    std::vector<int> enc = tok.encode(prompt);
    int start = std::max(0, (int)enc.size() - T);
    for (int i = start; i < (int)enc.size() && i - start < T; i++)
        ctx[i - start] = enc[i];

    std::string out = prompt;
    ForwardCache cache;

    for (int step = 0; step < max_new_tokens; step++) {
        forward(cfg, W, ctx, cache, false);

        // Last token's logits → sample
        int last = std::min((int)enc.size() + step, T-1);
        if ((int)enc.size() + step >= T) last = T-1;

        std::vector<float> logits(V);
        std::copy(&cache.logits[last*V], &cache.logits[last*V+V], logits.begin());

        int next_id;
        if (temperature <= 0.0f) {
            // Greedy
            next_id = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        } else {
            // Temperature-scaled softmax sampling
            for (auto& l : logits) l /= temperature;
            softmax(logits.data(), V);
            std::discrete_distribution<int> dist(logits.begin(), logits.end());
            next_id = dist(rng);
        }

        out += tok.decode(next_id);

        // Shift context window left and append new token
        std::rotate(ctx.begin(), ctx.begin()+1, ctx.end());
        ctx[T-1] = next_id;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// 19.  Training loop
// ─────────────────────────────────────────────────────────────────────────────
static void train(
    const std::string& data_path,
    const std::string& model_path = "model.bin",
    int   max_iters   = 5000,
    int   eval_every  = 100,
    float lr          = 3e-3f)
{
    // Load data
    std::ifstream f(data_path);
    if (!f) { std::fprintf(stderr, "Cannot open %s\n", data_path.c_str()); return; }
    std::string text((std::istreambuf_iterator<char>(f)), {});
    std::printf("[data] loaded %zu characters\n", text.size());

    // Build tokenizer
    Tokenizer tok;
    tok.build(text);
    std::printf("[tok]  vocab_size = %d\n", tok.vocab_size());
    tok.save("vocab.bin");

    std::vector<int> data = tok.encode(text);

    // Build model
    Config cfg;
    cfg.vocab_size = tok.vocab_size();
    ModelWeights W;
    W.init(cfg);
    std::printf("[model] parameters = %d\n", W.count_params());

    Adam opt;
    opt.lr = lr;

    ForwardCache cache;
    std::mt19937 data_rng(0);
    int T = cfg.block_size;

    for (int iter = 0; iter < max_iters; iter++) {
        // Random batch of 1 sequence
        std::uniform_int_distribution<int> pos_dist(0, (int)data.size() - T - 2);
        int start_pos = pos_dist(data_rng);
        std::vector<int> seq(data.begin() + start_pos,
                             data.begin() + start_pos + T + 1);
        std::vector<int> ctx(seq.begin(), seq.begin() + T);

        // Forward
        float loss = forward(cfg, W, ctx, cache, true);

        // Backward
        backward(cfg, W, ctx, cache);

        // Optimiser step
        opt.update_all(W);

        if ((iter + 1) % eval_every == 0) {
            auto t = std::chrono::steady_clock::now();
            std::printf("[iter %5d] loss = %.4f\n", iter+1, loss);
        }
    }

    // Save
    save_weights(W, model_path);
    std::printf("[done] model saved to %s\n", model_path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// 20.  Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::printf(
            "Usage:\n"
            "  %s train <data.txt> [model.bin] [iters] [lr]\n"
            "  %s gen   <model.bin> <vocab.bin> \"<seed>\"\n",
            argv[0], argv[0]);
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "train") {
        if (argc < 3) { std::fprintf(stderr, "train: need <data.txt>\n"); return 1; }
        std::string data_path  = argv[2];
        std::string model_path = (argc > 3) ? argv[3] : "model.bin";
        int   iters = (argc > 4) ? std::stoi(argv[4]) : 5000;
        float lr    = (argc > 5) ? std::stof(argv[5]) : 3e-3f;
        train(data_path, model_path, iters, 100, lr);

    } else if (mode == "gen") {
        if (argc < 5) { std::fprintf(stderr, "gen: need <model.bin> <vocab.bin> \"<seed>\"\n"); return 1; }
        std::string model_path = argv[2];
        std::string vocab_path = argv[3];
        std::string seed       = argv[4];
        float temperature      = (argc > 5) ? std::stof(argv[5]) : 0.8f;
        int   max_new          = (argc > 6) ? std::stoi(argv[6]) : 300;

        ModelWeights W;
        if (!load_weights(W, model_path)) { std::fprintf(stderr, "Cannot load %s\n", model_path.c_str()); return 1; }

        Tokenizer tok;
        if (!tok.load(vocab_path)) { std::fprintf(stderr, "Cannot load %s\n", vocab_path.c_str()); return 1; }

        std::string out = generate(W.cfg, W, tok, seed, max_new, temperature);
        std::printf("%s\n", out.c_str());

    } else {
        std::fprintf(stderr, "Unknown mode: %s\n", mode.c_str());
        return 1;
    }

    return 0;
}
