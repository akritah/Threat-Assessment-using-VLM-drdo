# Qualitative Comparative Evaluation Report: Gemma 3 Base vs. Fine-Tuned (ActivityNet Adapter) on XD-Violence Dataset

**Prepared by:** Graduate Research Assistant  
**Target Reviewer:** DRDO Research Project Guide  
**Project Title:** Parameter-Efficient VLM Domain Adaptation for Surveillance Threat Identification  

---

## 1. Objective
This research project evaluates whether domain adaptation of the **Gemma 3 4B Vision-Language Model (VLM)** using **QLoRA** on the **ActivityNet Captions** dataset improves its scene understanding, temporal activity description capabilities, and downstream threat-level analysis in real-world surveillance videos. The evaluation contrasts the base model against the fine-tuned adapter on anomalous and normal categories from the **XD-Violence** surveillance dataset.

---

## 2. Dataset Description
The evaluation utilizes a stratified sample of the **XD-Violence dataset**, which comprises raw, unedited CCTV footage containing normal security footage and diverse anomalous threats. 
*   **Total Evaluated Videos**: {len_records} videos.
*   **Class Distribution**:
{dist_lines}
*   **Video Format**: Untrimmed `.mp4` video clips at varying aspect ratios.

---

## 3. Evaluation Methodology
For each surveillance video, a key midpoint frame representing the primary activity is extracted at 1 FPS using the project's native `frame_extractor.py` module. Both models are queried with the following surveillance-oriented prompt:
```text
Analyze this surveillance scene.
Describe:
* What is happening?
* Which activities are visible?
* Is there any suspicious behaviour?
* Are there any threat indicators?
* Estimate the threat level as Low, Medium, or High.
* Explain your reasoning.
