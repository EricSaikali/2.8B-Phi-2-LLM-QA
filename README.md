# CS-552 - Final submission

Welcome to the final step of your MNLP project! As you can read in the [project description](https://docs.google.com/document/d/1SP8SCHPOZZGEhs2ay-38FjedRE1bS9Q99VJb28eHoYk/edit?usp=sharing), you have 2 main deliverables: 
1. Your final model - including its augmented counterpart(s)
3. The final report


## Repo Structure

The repo has 4 folders, 2 of which serve for you to submit the deliverables:
1. `_templates` contains the latex template for your final report. You MUST use this template.
2. `_tests` contains some scripts which run automated tests so you can be sure your submission is correctly formatted (e.g., that the right files exist in the right place). **Importantly, if your team is NOT implementing RAG, you should change the first line of `_tests/model_rag_validator.py` into `IMPLEMENTING_RAG = False`.**
3. `model` should contain your final models and your model-related implementation files (this includes any file for training, inference, quantization, RAG, and other necessary functions needed for the evaluator to execute successfully). Your implementation should be compatible with the [provided code template](https://github.com/CS-552/project-code-2024).
4. `pdfs` should contain a single pdf file, your final report (named `<YOUR-GROUP-NAME>.pdf`).

## Running the tests manually
The autograding tests run automatically with every commit to the repo. Please check M1's repo for instructions on running the tests manually if you wish to do so.
