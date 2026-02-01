# Product Requirements Document (PRD)

**Project Name:** Statistical Analysis on Fast-Outliers in DCQO and DAQO-based QE-MTS for the LABS Problem
**Team Name:** QuackingOn
**GitHub Repository:** https://github.com/raifef/2026-NVIDIA.git

---

> **Note to Students:** > The questions and examples provided in the specific sections below are **prompts to guide your thinking**, not a rigid checklist. 
> * **Adaptability:** If a specific question doesn't fit your strategy, you may skip or adapt it.
> * **Depth:** You are encouraged to go beyond these examples. If there are other critical technical details relevant to your specific approach, please include them.
> * **Goal:** The objective is to convince the reader that you have a solid plan, not just to fill in boxes.

---

## 1. Team Roles & Responsibilities [You can DM the judges this information instead of including it in the repository]

| Role | Name | GitHub Handle | Discord Handle
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | Raife Foulkes | @raifef | @raifef |
| **GPU Acceleration PIC** (Builder) | Raife Foulkes | @raifef | @raifef |
| **Quality Assurance PIC** (Verifier) | Raife Foulkes | @raifef | @raifef |
| **Technical Marketing PIC** (Storyteller) | Raife Foulkes | @raifef | @raifef | (Solo Team)

---

## 2. The Architecture
**Raife Foulkes:** Project Lead

### Choice of Quantum Algorithm
* **Algorithm:** 
* Digitized Adiabatic Quantum Optimization (DAQO) seeding implemented in CUDA-Q. The paper reports that QE-MTS exhibits low-quantile outlier replicate-median TTS points; we seek to investigate the source of these very fast outliers. 

* **Motivation:** 
* Fig. 2 + Sec. IV.1 of Alejandro Gomez Cadavid et al reports QE-MTS outliers (very low TTS replicate-medians), visible as points far below the MTS distribution and discussed as low-quantile outliers. These are captured around 
Q_0.04 in their analysis. We want to investigate whether the CD term does not merely improve average seeding quality; it changes the tail behavior (frequency of very fast solves) by placing probability mass into rare good basins that MTS exploits quickly. We test whether DAQO produces a statistically different fast-outlier rate than DCQO under compute-matched budgets. Understanding this would provide insight into what it is about the specific structure of the quantum algorithm which provides extensive speedup for this task. DAQO is a digitized quantum annealing protocol: it prepares candidate bitstrings by evolving under a scheduled combination of a mixing Hamiltonian and the LABS cost Hamiltonian for a fixed number of discrete steps. DCQO is a different protocol that augments this evolution with counterdiabatic terms designed to suppress non-adiabatic transitions and reshape the sampling distribution. QAOA has been shown (in the literature) to not be immediately amenable to this problem, VQE's heavy training requirements are a poor fit for this investigation and QITE is likely a very good competitor with regard to expected outlier probability, but I am working alone and this may require too much overhead within the constraints of the hackathon. By investigating DAQO we can investigate how the structure of a quantum algorithm affects its statistics, and this could provide insight into faster algorithmic avenues. We analyse with the same N, same number of Trotter steps, same shots, same MTS wall-time budget per run and seek only the statistical variations in outliers.
  
   

### Literature Review
* **Reference:** “Scaling advantage with quantum-enhanced memetic tabu search for LABS”, Alejandro Gomez Cadavid et al., https://arxiv.org/pdf/2511.04553 (2025, preprint).
* **Relevance:** Baseline paper extensively referred to in tutorial. This motivates my core question: whether the CD term changes the tail of the TTS distribution rather than only the median.
* **Reference:** “Digitized-counterdiabatic quantum optimization”, Narendra N. Hegade, X. Chen, and E. Solano, https://link.aps.org/doi/10.1103/PhysRevResearch.4.L042030 (2022, Physical Review Research 4, L042030).
* **Relevance:** Foundational reference for DCQO itself. Explains origin of CD term and why it can accelerate ground-state targeting. This is the key theory of what we are ablation testing.
* **Reference:** “Shortcuts to adiabaticity: Concepts, methods, and applications", David Guéry-Odelin et al., https://link.aps.org/doi/10.1103/RevModPhys.91.045001 (2019, Reviews of Modern Physics 91, 045001).
* **Relevance:** Broad overview of shortcuts to adiabaticity and counterdiabatic driving. Explains why removing CD term produces a principled baseline (DAQO) rather than an arbitrary parameter tuning
* **Reference:** “New Improvements in Solving Large LABS Instances Using Massively Parallelizable Memetic Tabu Search”, Zhiwei Zhang et al., https://arxiv.org/html/2504.00987v1 (2025, preprint arXiv:2504.00987).
* **Relevance:** Describes MTS implementation using large parallelisation on NVIDIA A100 systems, directly applicable to the GPU acceleration aspect of this problem and gives me good insight for GPU acceleration approach to take.
* **Reference:** “Evidence of scaling advantage for the quantum approximate optimization algorithm on a classically intractable problem”, Ruslan Shaydulin et al., https://www.science.org/doi/10.1126/sciadv.adm6761 (2024, Science Advances; also preprint arXiv:2308.02342).
* **Relevance:** Explains why QAOA is a good alternative but may require substantial depth in practice. 
* **Reference:** “Combinatorial optimization with quantum imaginary time evolution”, Nora M. Bauer et al., https://link.aps.org/doi/10.1103/PhysRevA.109.052430 (2023, preprint arXiv:2312.16664).
* **Relevance:** applies QITE to PUBO problems comparing with QAOA-depth regimes. This supports my statement that QITE could be a compelling competitor, but may be too ambitious to properly investigate.
---

## 3. The Acceleration Strategy
**Raife Foulkes:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)
* **Strategy:**
    * We test with a single L4. We batch shots and keep circuit execution inside GPU loops to avoid CPU to GPU overhead I have encountered in previous GPU accelerated workflows. We benchmark shots/sec for DAQO vs DCQO at same N and depth. For this task we don't seek to push the limits of high N, but instead repeatedly sample in the paper’s highlighted regime (N=33–37), where low-quantile outliers were reported, to determine the statistical significance of these fast outliers. We could optionally distribute the circuit simulation across multiple L4s to allow multiple runs to be sampled quickly.
 

### Classical Acceleration (MTS)
* **Strategy:** 
* We will port the energy evaluation to GPU, allowing us to evaluate energies for whole populations in parallel, massively reducing classical overhead due to the sums in these terms. To do this, we implement GPU batch energy evaluation for a batch of K sequences (population + candidates) using CuPy and replace per-candidate Python loops with vectorized GPU kernels. GPU batch energy is used for ranking the seed population, evaluating offspring during memetic recombination, and optionally evaluating a batch of neighbor moves per tabu step.


### Hardware Targets
* **Dev Environment:** Logic verified for small samples on QBraid CPU runs, later initial GPU testing will be performed on low-demand, low-power Brev GPUs to ensure GPU workflow is error-free and the GPU workflow is actually capable of achieving speedup vs CPU-based logic. Final benchmarks perform on A100 for short, controlled runs.
* **Production Environment:** Brev A100-80GB to run repeated runs at relatively high N. We will target the quantum-advantage region N=33–37 highlighted by the paper for tail statistics, we are not chasing the maximum N possible. The goal is not to push limits of high N, but instead to investigate whether the high frequency of fast outliers is affected by the presence of the counterdiabatic term.

---

## 4. The Verification Plan
**Raife Foulkes:** Quality Assurance PIC

### Unit Testing Strategy
* **Framework:** pytest (plus optional hypothesis property tests).
* **AI Hallucination Guardrails:** 
* We will ensure all generated results pass low-cost tests, all bitstrings of length N, data types consistent and values within reasonable bounds (energy is deterministic for identical bitstring input, energy is invariant under: global flip, reversal, staggering negatives)
    

### Core Correctness Checks
* **Check 1 (Symmetry):** 
    * LABS sequence $S$ and its negation $-S$, bitstring reversal and staggering negatives ($00000 -> 01010$) must have identical energies. We will sample a number of used bitstrings and verify that the energies of all strings within this verification sample which share one of these symmetries are identical. 
* **Check 2 (Brute Force):**
    * For small N, solutions can be directly verified against a brute force algorithm, and for very small N both can be verified against hand calculations. We verify for all N < 8 against a brute force algorithm to ensure correctness.
* **Check 3 (GPU vs CPU Equivalence):**
* Results are expected to be identical whether they are run on the GPU or CPU, this will allow us to find errors when porting over to GPU accelerated workflow. We will test whether GPU and CPU energy evaluators match exactly for a fixed set of random bitstrings.
---

## 5. Execution Strategy & Success Metrics
**Raife:** Technical Marketing PIC

### Agentic Workflow
* **Plan:** 
* We are using ChatGPT Pro and Codex to speed up the workflow. We have compiled all of the CUDA-Q documentation, relevant literature from the literature review and earlier code to provide context to the agents. Codex provides higher quality code but sometimes lacks reasoning passing back through ChatGPT 5.2 Thinking can help diagnose bigger-picture errors. I have experience using this combination to port from CPU to GPU accelerated workflows for numerical simulation so I am aware of a number of mistakes it may try to make, and how ti fix them.

### Success Metrics
* **Metric 1 (GPU Speedup):** Achieve faster wall-time-to-solution with GPU vs CPU
* **Metric 2 (Reproducibility):** Reproduce the high-speed outliers observed in the paper. We primarily define a fast outlier to be a run in which the TTS is in the lowest 4% of the DCQO TTS distribution, to align with the paper, or it may secondarily be defined as a run which reaches energy ≤ E_target within τ seconds, where τ might be 10% of DCQO median TTS. We aim for R repeats in the range of R = 50-200 depending on runtime, with a minimum of R=50.
* **Metric 3 (Statistical analysis):** Perform a statistical hypothesis test to determine whether the CD term is responsible for these high-speed outliers. We will compare the probability for a fast outlier between methods using a two-proportion test and report binomial confidence intervals.

### Visualization Plan
* **Plot 1:** Time-to-Solution vs. Problem Size (N) comparing CPU vs. GPU
* **Plot 2:** Compare $Q_{0.04}$, $Q_{0.10}$ of TTS distributions between DAQO-MTS and DCQO-MTS. (Outliers expected $\sim Q_{0.04}$)
* **Plot 3:** Empirical distribution function of TTS for DAQO vs DCQO to visually distinguish the tails
---

## 6. Resource Management Plan
**Raife:** GPU Acceleration PIC 

* **Plan:** 
* All development to remain on Qbraid CPU runs until logic fully implemented. Low-demand, low-cost L4 runs will then be utilised to ensure GPU acceleration is working as intended and finally A100 will be used only for polished final results after GPU acceleration has been shown to be error free and promising in its implementation. I will shut down all instances during breaks, there will be no idle time greater than 10 minutes.  
