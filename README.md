# How to Replace TAs: A Comprehensive Study on Letting LLMs Answer Your Questions

## Project Overview
This project investigates the feasibility of utilizing Large Language Models (LLMs) to autonomously answer multiple-choice exam questions (MCQA). While general question-answering tasks are relatively straightforward, challenges arise when dealing with more domain-specific inquiries, particularly those requiring mathematical reasoning.

To address these challenges, we experimented with different training methodologies, including Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). Our findings indicate that models fine-tuned with DPO align significantly better with user preferences and demonstrate slight improvements in performance. Additionally, we explored quantization techniques, reducing model precision to 8-bit and 4-bit representations, and analyzed their impact on performance.

Our study highlights the difficulties associated with automated answering of STEM-related MCQAs, emphasizing the limitations and potential areas for improvement in LLM-based educational assistance.

## Repository Structure
This repository is structured as follows:

- `_templates/` – Contains the LaTeX template for the final report.
- `_tests/` – Includes scripts for automated validation and formatting. 
- `model/` – Contains all relevant implementation files, including:
  - Training scripts
  - Inference code
  - Quantization scripts
  - RAG-related files (if applicable)
  - Any other necessary functions for evaluation
- `pdfs/` – Stores the final project report.

## Summary of Findings
- **Training Approaches**: DPO significantly improves model alignment with preference data, resulting in better responses.
- **Performance Evaluation**: While the model effectively handles general Q&A tasks, its accuracy drops when dealing with highly specialized MCQAs.
- **Quantization Effects**: Reducing precision to 8-bit and 4-bit representations leads to minor performance trade-offs, which are analyzed in detail.
- **Challenges**: Answering STEM MCQAs remains an open problem, requiring further advancements in LLM reasoning and contextual understanding.
