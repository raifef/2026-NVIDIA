> **Note to Students:** > The questions and examples provided in the specific sections below are **prompts to guide your thinking**, not a rigid checklist. 
> * **Adaptability:** If a specific question doesn't fit your strategy, you may skip or adapt it.
> * **Depth:** You are encouraged to go beyond these examples. If there are other critical technical details relevant to your specific approach, please include them.
> * **Goal:** The objective is to convince the reader that you have employed AI agents in a thoughtful way.

**Required Sections:**

1. **The Workflow:** How did you organize your AI agents? (e.g., "We used a Cursor agent for coding and a separate ChatGPT instance for documentation").
*I split the work into two “agent roles” so I could move fast without losing control of correctness:

*Builder (code-writing agent): Used an AI coding agent (ChatGPT/Codex-style workflow) to generate and refactor the main implementation in phase_ab_outliers.py: CUDA-Q sampling wrappers for DAQO/DCQO, population construction, and a GPU-capable LABS energy evaluator for batched scoring. I kept prompts narrowly scoped (one subsystem at a time) and always requested patch-style outputs so I could diff changes cleanly.

*Verifier (analysis + QA agent): Used a separate ChatGPT session/persona as a skeptical reviewer to (i) audit my metrics vs the PRD definition (outliers as a low-quantile TTS phenomenon), (ii) sanity-check statistical logic (baseline-defined thresholds, two-proportion test, binomial CIs), and (iii) design a test plan targeting likely hallucinations (wrong energy formula, wrong symmetry assumptions, GPU/CPU mismatch, silently broken CSVs).

*Storyteller (documentation agent): Used ChatGPT to translate technical choices into PRD/README language: explaining compute-matching, describing success metrics (tail ECDF, Q0.04), and writing judge-facing interpretations of plots while explicitly calling out limitations (e.g., when wall-clock TTS is dominated by sampling runtime rather than MTS dynamics).

*The Builder produced code, the Verifier tried to break it with tests and counterexamples, and the Storyteller wrote up what we did and why it is credible and how it fit into my declared PRD.
   
2. **Verification Strategy:** How did you validate code created by AI?
*Because AI-generated code is most likely to fail in silent, plausible ways (off-by-one errors, wrong spin mapping, wrong symmetry, incorrect batching, dtype bugs on GPU), I validated at three levels:

Unit tests for mathematical invariants of LABS and conserved properties (e.g. symmetries), cross-implementation equivalence tests (scalar vs batch; CPU vs GPU) to catch hallucinated optimizations or incorrect vectorization and pipeline output tests for Phase-A and B CSVs to catch schema drift, inconsistent statistics and otherwise infeasible results.

**Specific unit tests written**

*All of these are implemented in test_phase_ab_sanity.py:

* **LABS Energy invariances (symmetry tests)**

*Global flip of all bits should not change energy. Reversal should not change energy. Alternating sign should not change energy.

*Why: These are known invariances for the LABS objective. If AI hallucinates the spin mapping or correlation sum, these break immediately.

* **Physical bounds**
*Enforce: $0≤E≤∑_{k=1}^{N−1}(N−k)^2$

*Why: This catches negative energies, overflow, or wrong correlation accumulation.

* **Scalar vs batch correctness**

*Check labs_energy_batch(pop, use_gpu=False) matches labs_energy(bits) for each item in the batch. If GPU is available, also check labs_energy_batch(pop, use_gpu=True) matches scalar results exactly.

*Why: Most AI errors happen when “optimizing” loops into vectorized kernels, especially with dtype conversions and indexing.

* **Search convergence sanity**

*Tabu search should never return a worse best energy than the start state (best-found energy is non-increasing). Memetic tabu search (MTS) should never worsen the best energy relative to the best in the initial population.

*Why: AI often writes search logic that accidentally overwrites the incumbent best or mis-tracks the best index.

* **Output CSV sanity checks**

*Phase-A schema exists and has required columns. Quantiles are monotone: bestE_sample <= q10 <= q50 <= q90. shots_per_s ≈ shots / sample_elapsed_s (within tolerance). Energies are within bounds. Phase-B sanity: mts_bestE <= seed_bestE, mts_outer_iters >= 1, finite times.

*Why: AI can “make the code run” but write inconsistent stats or malformed outputs that corrupt downstream analysis.


3. **The "Vibe" Log:**
*Debugging the PRD mismatch (“outliers” weren’t actually outliers):
*AI quickly noticed that my original Phase-B outlier logic was selecting outliers based on sampling runtime rather than time-to-hit-target (TTS), and that MTS couldn’t terminate early if TARGET_E was unreachable. That would have invalidated the entire tail statistics claim. The AI’s critique led directly to instrumenting tts_hit_s and rethinking outlier thresholds, saving a lot of time that I would otherwise have spent analysing meaningless outliers.

* *Learn - where I changed prompting to get better results*
*Switching from “write code” to “write tests + patch diffs”:
*Early prompts like “optimize this” tended to produce plausible-but-risky refactors. I improved results by: Providing explicit invariants (symmetry, bounds) and demanding tests first. Asking for patch-style diffs rather than full rewrites. Supplying a “definition of done” for each step (e.g., “batch and scalar energies must match exactly” + “CSV quantiles must be monotone”). This kept the agent constrained, reduced hallucinated changes, and made review feasible.

* *Fail - where AI hallucinated (and how I fixed it)*
*Hallucinated performance/meaning from plots:
*At one point the agent inferred “DAQO is much better” from a tail-ECDF where DAQO’s wall-clock was smaller. But the underlying data showed that the difference was dominated by sampling runtime (DCQO circuit simulation slower), while seed quality often favored DCQO (higher seed-hit rate). 
* Fix: I changed analysis to separate components: report sample_elapsed_s and mts_t_hit_s separately, define outliers in the metric aligned with the PRD hypothesis, and compute outlier thresholds based on DCQO baseline for the relevant metric.


