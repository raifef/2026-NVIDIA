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

Unit tests for mathematical invariants of LABS and “must-never-break” properties (symmetries, bounds).

Cross-implementation equivalence tests (scalar vs batch; CPU vs GPU) to catch hallucinated optimizations or incorrect vectorization.

Pipeline output tests for Phase-A/Phase-B CSVs to catch schema drift, inconsistent stats (quantiles not monotone), and “impossible” results.

Specific unit tests written (to catch hallucinations / logic errors)

All of these are implemented in test_phase_ab_sanity.py:

A) LABS Energy invariances (symmetry tests)

Global flip of all bits should not change energy.

Reversal should not change energy.

Staggering (alternating sign / bit xor pattern) should not change energy.

Why: These are known invariances for the LABS objective. If AI hallucinates the spin mapping or correlation sum, these break immediately.

B) Physical bounds
Enforce: $0≤E≤∑_{k=1}^{N−1}(N−k)^2$

Why: This catches negative energies, overflow, or wrong correlation accumulation.

C) Scalar vs batch correctness

Check labs_energy_batch(pop, use_gpu=False) matches labs_energy(bits) for each item in the batch.

If GPU is available, also check labs_energy_batch(pop, use_gpu=True) matches scalar results exactly.

Why: Most AI errors happen when “optimizing” loops into vectorized kernels, especially with dtype conversions and indexing.

D) Search convergence sanity

Tabu search should never return a worse best energy than the start state (best-found energy is non-increasing).

Memetic tabu search (MTS) should never worsen the best energy relative to the best in the initial population.

Why: AI often writes search logic that accidentally overwrites the incumbent best or mis-tracks the best index.

E) Output CSV sanity checks

Phase-A schema exists and has required columns.

Quantiles are monotone: bestE_sample <= q10 <= q50 <= q90.

shots_per_s ≈ shots / sample_elapsed_s (within tolerance).

Energies are within bounds.

Phase-B sanity: mts_bestE <= seed_bestE, mts_outer_iters >= 1, finite times.

Why: AI can “make the code run” but write inconsistent stats or malformed outputs that corrupt downstream analysis.
   
* *Requirement:* You must describe specific **Unit Tests** you wrote to catch AI hallucinations or logic errors.


3. **The "Vibe" Log:**
* *Win:* One instance where AI saved you hours.
* *Learn:* One instance where you altered your prompting strategy (provided context, created a skills.md file, etc) to get better results from your interaction with the AI agent.
* *Fail:* One instance where AI failed/hallucinated, and how you fixed it.
* *Context Dump:* Share any prompts, `skills.md` files, MCP etc. that demonstrate thoughtful prompting.


