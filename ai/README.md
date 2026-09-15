# Tiny Character-Level GPT — C++ implementation

> Faithful C++ recreation of the architecture visualised in **AI.png**.

---

## Architecture (from AI.png)

| Hyper-parameter | Value |
|---|---|
| `n_embd`     | 16 |
| `n_head`     | 4 (d_k = 4 per head) |
| `n_layer`    | 4 transformer blocks |
| `vocab_size` | 65 characters |
| `block_size` | 55 tokens (context window) |
| **Total params** | **14,208** |
| **Trainable matrices** | **27** |

### Parameter breakdown

```
4 blocks × (Q + K + V + wo + fc1 + fc2)
  = 4 × (4×16×16 + 16×64 + 64×16)
  = 4 × 3,072
  = 12,288

wte  (65 × 16) =  1,040
wpe  (55 × 16) =    880
─────────────────────────
Total          = 14,208  ✓

Matrices: 24 (block projections) + wte + wpe + lm_head (tied to wte) = 27  ✓
```

### Forward pass (AI.png section 1)

```
Input (char id)
  → Token Embedding  wte [65×16]
  → Position Embedding wpe [55×16]
  → Add (⊕)
  → RMSNorm
  → Transformer Block × 4
      ├─ [Attention sub-layer]
      │    RMSNorm
      │    Linear → Q  [16×16]
      │    Linear → K  [16×16]
      │    Linear → V  [16×16]
      │    Multi-Head Self-Attention (n_head=4, d_k=4)
      │      • Q·Kᵀ / √d_k, causal softmax, weighted sum of V
      │    Linear → Output wo  [16×16]
      │    residual add (⊕)
      └─ [MLP sub-layer]
           RMSNorm
           Linear → fc1  [16→64]
           ReLU
           Linear → fc2  [64→16]
           residual add (⊕)
  → Linear lm_head [16×65]  (weight-tied to wte)
  → Softmax
  → Cross-Entropy Loss  (training only)
```

### Backward pass (AI.png section 2)

Gradients flow in **reverse order** of forward operations:

```
Loss ← Softmax ← lm_head ← Residual Add (after last MLP)
  ← MLP: fc2 → ReLU → fc1 → RMSNorm
  ← Residual Add (after Attention)
  ← Attention: wo → Softmax → Q/K/V → RMSNorm
  ← … repeat for each block …
  ← RMSNorm (input)
  ← Token Embedding (wte)  +  Position Embedding (wpe)
```

---

## Build

### Option A — CMake (recommended)

```bash
# Linux / macOS / MSYS2
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/gpt

# Windows (MSVC Developer Command Prompt)
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
build\Release\gpt.exe
```

### Option B — single command

```bash
# Linux / macOS / MSYS2 / MinGW
g++ -O2 -std=c++17 -o gpt gpt.cpp

# MSVC Developer Command Prompt
cl /O2 /std:c++17 /EHsc gpt.cpp /Fe:gpt.exe
```

---

## Usage

### Train

```bash
./gpt train data.txt [model.bin] [iterations] [learning_rate]
```

- `data.txt` — plain-text corpus (any character file, recommended ≥ 50 KB)
- `model.bin` — output weights file (default: `model.bin`)
- `iterations` — training steps (default: 5000)
- `learning_rate` — Adam LR (default: 0.003)

Produces `model.bin` + `vocab.bin` in the working directory.

### Generate

```bash
./gpt gen model.bin vocab.bin "seed text" [temperature] [max_new_tokens]
```

- `temperature` — 0 = greedy, 0.8 = default (recommended), 1.0 = maximum entropy
- `max_new_tokens` — characters to generate (default: 300)

### Example

```bash
# Generate training data (requires the OpenRouter key in ../.env)
python fetch_data.py          # optional helper

# Train for 10 000 steps
./gpt train data.txt model.bin 10000 3e-3

# Generate 500 new characters
./gpt gen model.bin vocab.bin "the sun rose" 0.8 500
```

---

## File layout

```
ai/
├── gpt.cpp          ← this implementation (single-file, zero dependencies)
├── CMakeLists.txt   ← cross-platform build
├── README.md        ← this file
├── .env.example     ← OpenRouter key template for data fetching
├── AI.png           ← original architecture diagram
├── model.bin        ← (generated) trained weights
└── vocab.bin        ← (generated) character vocabulary
```

---

## Design notes

| Choice | Rationale (matches AI.png) |
|---|---|
| **RMSNorm** instead of LayerNorm | Explicitly labelled in every norm box |
| **ReLU** in MLP | `fc1, ReLU, fc2` shown in MLP sub-layer |
| **No biases** | Confirmed by parameter count: 14,208 = 12,288 + 1,040 + 880 with no bias terms |
| **Weight-tied lm_head** | `lm_head = wteᵀ`; the only way 27 matrices yield 14,208 scalars |
| **Causal (masked) attention** | Standard autoregressive GPT; implied by the next-char output |
| **Adam optimiser** | β₁=0.9, β₂=0.999 — standard for small transformers |
