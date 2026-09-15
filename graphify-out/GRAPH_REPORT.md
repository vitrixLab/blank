# Graph Report - blank  (2026-09-15)

## Corpus Check
- 6 files · ~61,061 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 139 nodes · 232 edges · 13 communities (9 shown, 3 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a404c38e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- gpt.cpp
- AttentionWeights
- ModelWeights
- Tokenizer
- ForwardCache
- Tiny Character-Level GPT — C++ implementation
- MLPWeights
- AttentionCache
- Adam
- rules/graphify.md
- workflows/graphify.md
- README.md

## God Nodes (most connected - your core abstractions)
1. `AttentionWeights` - 22 edges
2. `ModelWeights` - 22 edges
3. `Config` - 16 edges
4. `MLPWeights` - 14 edges
5. `AttentionCache` - 14 edges
6. `Tokenizer` - 12 edges
7. `forward()` - 11 edges
8. `train()` - 11 edges
9. `MLPCache` - 10 edges
10. `ForwardCache` - 10 edges

## Surprising Connections (you probably didn't know these)
- `generate()` --references--> `Config`  [EXTRACTED]
  ai/gpt.cpp → ai/gpt.cpp  _Bridges community 0 → community 3_
- `ModelWeights` --references--> `Config`  [EXTRACTED]
  ai/gpt.cpp → ai/gpt.cpp  _Bridges community 0 → community 2_
- `AttentionCache` --references--> `vector`  [EXTRACTED]
  ai/gpt.cpp →   _Bridges community 0 → community 7_
- `AttentionWeights` --references--> `vector`  [EXTRACTED]
  ai/gpt.cpp →   _Bridges community 0 → community 1_
- `ForwardCache` --references--> `vector`  [EXTRACTED]
  ai/gpt.cpp →   _Bridges community 0 → community 4_

## Import Cycles
- None detected.

## Communities (13 total, 3 thin omitted)

### Community 0 - "gpt.cpp"
Cohesion: 0.23
Nodes (20): at(), attention_bwd(), attention_fwd(), backward(), Config, block_size, n_embd, n_head (+12 more)

### Community 1 - "AttentionWeights"
Cohesion: 0.12
Nodes (17): AttentionWeights, dwk, dwo, dwq, dwv, mwk, mwo, mwq (+9 more)

### Community 2 - "ModelWeights"
Cohesion: 0.12
Nodes (14): BlockWeights, attn, mlp, ModelWeights, blocks, cfg, dwpe, dwte (+6 more)

### Community 3 - "Tokenizer"
Cohesion: 0.25
Nodes (10): generate(), load_weights(), main(), save_weights(), Tokenizer, c2i, i2c, train() (+2 more)

### Community 4 - "ForwardCache"
Cohesion: 0.13
Nodes (15): BlockCache, attn, mlp, ForwardCache, blocks, embed, embed_n, logits (+7 more)

### Community 5 - "Tiny Character-Level GPT — C++ implementation"
Cohesion: 0.13
Nodes (14): Architecture (from AI.png), Backward pass (AI.png section 2), Build, Design notes, Example, File layout, Forward pass (AI.png section 1), Generate (+6 more)

### Community 6 - "MLPWeights"
Cohesion: 0.17
Nodes (10): MLPWeights, dfc1, dfc2, fc1, fc2, mfc1, mfc2, vfc1 (+2 more)

### Community 7 - "AttentionCache"
Cohesion: 0.20
Nodes (10): AttentionCache, att, att_pre, K, Q, sv, V, x_in (+2 more)

### Community 8 - "Adam"
Cohesion: 0.29
Nodes (6): Adam, b1, b2, eps, lr, step

## Knowledge Gaps
- **82 isolated node(s):** `vocab_size`, `block_size`, `n_embd`, `n_head`, `n_layer` (+77 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 88 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AttentionWeights` connect `AttentionWeights` to `gpt.cpp`, `ModelWeights`, `MLPWeights`?**
  _High betweenness centrality (0.188) - this node is a cross-community bridge._
- **Why does `ModelWeights` connect `ModelWeights` to `gpt.cpp`, `Adam`, `Tokenizer`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **Why does `AttentionCache` connect `AttentionCache` to `gpt.cpp`, `ForwardCache`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **What connects `vocab_size`, `block_size`, `n_embd` to the rest of the system?**
  _82 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AttentionWeights` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._
- **Should `ModelWeights` be split into smaller, more focused modules?**
  _Cohesion score 0.125 - nodes in this community are weakly interconnected._
- **Should `ForwardCache` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._