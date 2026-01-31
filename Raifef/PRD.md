# Product Requirements Document (PRD)

**Project Name:** [e.g., LABS-Solv-V1]
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
* **Algorithm:** [Identify the specific algorithm or ansatz]
* Digitized Adiabatic Quantum Optimization (dAQO) seeding implemented in CUDA-Q. The paper reports that QE-MTS exhibits low-quantile outlier replicate-median TTS points, by comparing against an algorithm without the counterdiabatic correction term, we seek to investigate the source of these very fast outliers.
    * *Example:* "Quantum Approximate Optimization Algorithm (QAOA) with a hardware-efficient ansatz."
    * *Example:* "Variational Quantum Eigensolver (VQE) using a custom warm-start initialization."

* **Motivation:** [Why this algorithm? Connect it to the problem structure or learning goals.]
* Fig. 2 + Sec. IV.1 reports  QE-MTS outliers (very low TTS replicate-medians), visible as points far below the MTS distribution and discussed as low-quantile outliers. These are captured around 
Q_0.04 in their analysis. We want to investigate whether the CD term does not merely improve average seeding quality; it changes the tail behavior (frequency of very fast solves) by placing probability mass into rare good basins that MTS exploits quickly. We test whether dAQO produces a statistically different fast-outlier rate than DCQO under compute-matched budgets. Understanding this would privide insight into what it is about the specific structure of the quantum algorithm which provides extensive speedup for this task. QAOA has been shown (in the literature) to not be immediately amenable to this problem, VQE's heavy training requirements are a poor fit for this investigation and QITE is likely a very good competitor with regard to expected outlier probability, but I am working alone and this may require too much overhead within the constraints of the hackathon. By investigating dAQO we can investigate how the structure of a quantum algorithm affects its statistics, and this could provide insight into faster algorithmic avenues.
  
   

### Literature Review
* **Reference:** [Title, Author, Link]
* **Relevance:** [How does this paper support your plan?]
    * *Example:* "Reference: 'QAOA for MaxCut.' Relevance: Although LABS is different from MaxCut, this paper demonstrates how parameter concentration can speed up optimization, which we hope to replicate."

---

## 3. The Acceleration Strategy
**Raife Foulkes:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)
* **Strategy:** [How will you use the GPU for the quantum part?]
    * We test with a single L4, then distribute the circuit simulation across multiple L4s to allow multiple runs to be sampled quickly. We batch shots and keep circuit execution inside GPU loops to avoid CPU to GPU overhead I have encountered in previous GPU accelerated workflows. We benchmark shots/sec for dAQO vs DCQO at same N and depth. For this task we don't seek to push the limits of high N, but instead repeatedly sample in the quantum advantageous region to determine the statistical significance of these fast outliers.
 

### Classical Acceleration (MTS)
* **Strategy:** [The classical search has many opportuntities for GPU acceleration. What will you chose to do?]
* We will port the energy evaluation to GPU, allowing us to evaluate energies for whole populations in parallel, massively reducing classical overhead due to the sums in these terms.


### Hardware Targets
* **Dev Environment:** Logic verified for small samples on QBraid CPU runs, later initial GPU testing will be performed on low-demand, low-power Breb GPUs to ensure GPU workflow is error-free and the GPU workflow is actually capable of achieving speedup vs CPU-based logic. Final benchmarks perform on A100 for short, controlled runs.
* **Production Environment:** Brev A100-80GB to run repeated runs at relatively high N, but not pushing limits. The goal is not to push limits of high N, but instead to investigate whether the high frequency of fast outliers is affected by the presence of the  counterdiabatic term

---

## 4. The Verification Plan
**Raife Foulkes:** Quality Assurance PIC

### Unit Testing Strategy
* **Framework:** [e.g., `pytest`, `unittest`]
* **AI Hallucination Guardrails:** [How do you know the AI code is right?]
* We will ensure all generated results pass low-cost tests, e.g. all bitstrings of length N, data types consistent and values within reasonable bounds
    

### Core Correctness Checks
* **Check 1 (Symmetry):** 
    * LABS sequence $S$ and its negation $-S$, bitstring reversal and staggering negatives ($00000 -> 01010$) must have identical energies. We will sample a number of used bitstrings and verify that the energies of all strings within this verification sample which share one of these symmetries are identical. 
* **Check 2 (Brute Force):**
    * For small N, solutions can be directly verified against a brute force algorithm, and for very small N both can be verified against hand calculations

---

## 5. Execution Strategy & Success Metrics
**Raife:** Technical Marketing PIC

### Agentic Workflow
* **Plan:** [How will you orchestrate your tools?]
* We are using ChatGPT Pro and Codex to speedup workflow. We have compiled all of the CUDA-Q documentation, relevant literature from the literature review and earlier code to provide context to the agents. Codex provides higher quality code but sometimes lacks reasoninsg passing back through ChatGPT 5.2 Thinking can help diagnose bigger-picture errors. I have experience using this combination to port from CPU to GPU accelerated workflows for numerical simulation so I am aware of a number of mistakes it may try to make, and how ti fix them.

### Success Metrics
* **Metric 1 (GPU Speedup):** Achieve faster wall-time-to-solution with CPU vs GPU
* **Metric 1 (Reproducibility):** Reproduce the high-speed outliers observed in the paper
* **Metric 2 (Statistical analysis):** Perform a statistical hypothesis test to determine whether the CD term is responsible for these high-speed outliers

### Visualization Plan
* **Plot 1:** [e.g., "Time-to-Solution vs. Problem Size (N)" comparing CPU vs. GPU]
* **Plot 2:** Compare $Q_{0.04}$, $Q_{0.10}$ of TTS distributions between dAQO-MTS and DCQO-MTS. (Outliers expected $\sim Q_{0.04}$)

---

## 6. Resource Management Plan
**Raife:** GPU Acceleration PIC 

* **Plan:** [How will you avoid burning all your credits?]
* All development to remain on Qbraid CPU runs until logic fully implemented. Low-demand, low-cost L4 runs will then be utilised to ensure GPU acceleration is working as intended and finally A100 will be used only for polished final results after GPU acceleration has been shown to be error free and promising in its implementation
  
