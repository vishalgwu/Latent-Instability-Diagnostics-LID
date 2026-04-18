# Theory Synthesis: Transformer Dynamics — Metastable States
**Paper:** "Transformer Dynamics: A Neuroscientific Approach to Interpretability of LLMs"  
**Reference from:** Refecrences_list.docx — "Metastable Dynamical Systems in Language Models"  
**Authors:** Geshkovski et al. / Bhattacharjee & Lee (2024)  
**Status:** Priority 1 — Validates multi-cluster dynamics hypothesis  
**Synthesis author:** LID Research Team  
**Date:** Week 1

---

## 1. Core Claim (One Sentence)

Transformer hidden states exhibit metastable dynamics — sequences of quasi-stable plateaus interrupted by rapid transitions — and these transitions are mechanistically linked to semantic shifts in the generated content.

---

## 2. Key Mathematics

**Metastable trajectory model:**
```
h(t) ∈ S_k    for t ∈ [t_k, t_{k+1})   (metastable plateau k)
||h(t_{k+1}) - h(t_k)|| >> ||h(t) - h(t-1)||   for t ∉ transition
```

The system spends most time near "metastable states" (local energy minima) and transitions rapidly between them. Transition speed:
```
τ_transition << τ_plateau
```

**Cluster count:**  The number of active semantic clusters K* at each layer follows a monotonic increase with depth: early layers are undifferentiated, late layers are highly clustered.

---

## 3. Connection to Neuroscience Framework

The paper imports the "neural manifold" concept from computational neuroscience:
- Population codes in the brain cluster around attractor states during cognitive tasks
- Transitions between states correspond to changes in "cognitive mode" or decision
- The same dynamics appear in transformer residual streams

**Geodesic distance measure:**
```
d_geo(h_1, h_2) = length of shortest path on the manifold surface
```
(as opposed to Euclidean L2 distance — the manifold is not flat)

---

## 4. Relevance to LID

| Paper Finding | LID Implication |
|---|---|
| Metastable plateaus + fast transitions | Z(t) should be LOW during plateaus, SPIKE during transitions — explains peak structure |
| Transitions are rapid (1-2 time steps) | Justifies 1-2 token lead time — we catch the transition onset |
| Semantic transitions precede output | "Commitment" to wrong semantic cluster happens BEFORE token is emitted |
| Layer-dependent cluster count | Different layers have different "resolution" — supports our layer selection strategy |

**Key Insight for LID:**
> "Hallucination is a metastable transition. The hidden state commits to a wrong semantic attractor 1-2 steps before the token is emitted. Our perturbation test detects when the system is mid-transition (sensitive to perturbation) vs. settled in a plateau (insensitive)."

---

## 5. Why q(t) Drops During Hallucination

Metastable transitions involve the hidden state moving **orthogonally** to its current direction — it leaves one attractor basin for another. This predicts:
- During plateau: perturbation returns to same basin → high q (aligned direction)
- During transition: perturbation goes to a different basin → **low q** (direction changes)
- This is the core mechanistic explanation for why q(t) ↓ distinguishes hallucination from creative generation

---

## 6. Creative vs. Error Prediction

Creative generation is also a transition — but:
- Creative: transition to a different-but-valid attractor → I ↑, q remains moderate-high
- Hallucination: transition to an invalid attractor with no grounding → I ↑, q ↓ dramatically

This directly predicts the dual-metric signature that is our main novelty claim.

---

## 7. What We Should NOT Claim

- "Metastable" is a specific dynamical systems term — use it precisely
- We measure Euclidean distance and cosine similarity, not geodesic distance — acknowledge this simplification
- The neuroscience analogy is illustrative, not a formal derivation

---

## 8. Citation Plan

Cite in:
- Related Work (position LID within dynamical systems / neuroscience-inspired interpretability)
- Hypothesis development (explain why instability precedes hallucination)
- Discussion (interpret low q as orthogonal manifold transition)
