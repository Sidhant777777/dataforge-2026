# Memory vs. State — DataForge 2026 Submission

**Track:** Pathway Track  
**Concept:** Linear Attention / Recurrent State vs. KV Cache  
Live demo lik-https://sidhant777777.github.io/dataforge-2026/

---

## One-sentence claim

> A fixed-size recurrent state can process a sequence of any length without growing its memory — but it pays by losing the ability to perfectly retrieve any past token on demand.

---

## What this submission covers

Standard Transformers store a key and value vector for every token they process. That KV cache grows linearly: O(n·d) memory and O(n) time per new token. Linear attention collapses this into a single d×d state matrix — constant size regardless of sequence length — by replacing the softmax kernel with a linear feature map. Pathway's Dragon Hatchling (BDH) treats this recurrent state as the primary computational substrate, and BDH-CQ extends it to perform in-context learning entirely through state updates at inference, with no backward pass.

The submission teaches this concept through three interactive demos, then ties the mechanics to BDH's architectural choices.

---

## Audience

Students and practitioners familiar with the basics of attention (query-key-value) but who have not studied linear attention or state-space models in depth. No GPU, no API calls, no install — everything runs in the browser.

---

## Learning objectives

By the end of the interactive explainer, a reader should be able to:

1. Explain why the KV cache grows with sequence length and why that matters for deployment.
2. Describe the linear attention recurrence (S ← S + k⊗v, o = Sq / z·q) and what each variable represents.
3. Identify the core trade-off: constant memory at the cost of approximate retrieval.
4. Define *interference* and explain when it occurs in a shared state matrix.
5. Connect BDH's sparse activations (~5% fire rate) to the interference problem.

---

## Structure of the interactive explainer

### Section 1 — The memory problem  
A slider (1–20 tokens) animates two side-by-side panels: a growing KV list on the left, and a fixed 4×4 state matrix on the right that updates in place. A memory-bar comparison at the bottom makes the O(n·d) vs. O(d²) contrast visually immediate.

### Section 2 — The retrieval trade-off  
Five color→fruit associations are always stored. A toggleable sixth pair (orange, with a key vector nearly identical to yellow/banana) demonstrates interference. Six query buttons show side-by-side retrieval bars for softmax attention vs. linear attention. Querying yellow or orange when orange is toggled on reveals the blending that linear attention is prone to; softmax stays sharp because exponentiation amplifies small key differences.

### Section 3 — The math, live  
Three equation blocks (write, normalizer, read) are displayed alongside a live step demo: six token pairs are written one at a time into a 4×6 heatmap, so the reader can watch S update with each Hebbian write. A closing paragraph ties sparse activations (BDH) to reduced interference.

---

## What is live vs. pre-computed

Everything is live. No pre-stored answers. The state matrix S and normalizer z are built in JavaScript from the current set of stored pairs on every interaction. The heatmap colours are computed from the actual matrix values (positive → blue, negative → orange, scaled to max absolute value). Softmax scores use dot-product × 4 to match the sharpening effect of real softmax attention at typical temperatures.

---

## Files

| File | Purpose |
|---|---|
| `memory-vs-state.html` | Complete self-contained interactive explainer (single file, no CDN) |
| `concept_summary.pdf` | One-page concept summary for submission |
| `blog_post.pdf` | Narrative blog post |
| `make_summary.py` | Source script that generated `concept_summary.pdf` |
| `README.md` | This file |

---

## Citations

1. Katharopoulos et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020. arXiv:2006.16236.  
2. Pathway (2024). *Dragon Hatchling (BDH): A Brain-Inspired Post-Transformer Architecture.* Pathway Technical Report.  
3. Pathway (2024). *BDH-CQ: In-Context Learning via Recurrent State.* Pathway Technical Report.  
4. Dao & Gu (2024). *Transformers are SSMs: Generalized Models with Structured State Spaces.* NeurIPS 2024. arXiv:2405.21060.  
5. Schlag et al. (2021). *Linear Transformers are Secretly Fast Weight Programmers.* ICML 2021. arXiv:2102.11174.  
6. Sun et al. (2023). *Retentive Network: A Successor to Transformer for LLMs.* arXiv:2307.08621.
