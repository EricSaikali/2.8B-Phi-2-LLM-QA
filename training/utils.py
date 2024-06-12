# This file contains utility functions for training the model, especially for formatting the data for SFTTrainer


def format_function(dataset_name="EPFL_SFT"):
    def format_EPFL_SFT(input):
        output_texts = []
        for i in range(len(input['prompt'])):
            text = f"{input['prompt'][i]}\n\nExplanation: {input['chosen'][i]}"
            output_texts.append(text)
        return output_texts

    def format_MCQA(input):
        output_texts = []
        for i in range(len(input['question'])):
            text = f"{input['question'][i]}\n\nExplanation: {input['answer_text'][i]}\n\nAnswer: {input['answer'][i]}"
            output_texts.append(text)
        return output_texts

    def format_helpSteer():
        raise NotImplementedError("Will never be used, press 'F' to pay respect.")

    if dataset_name == "EPFL_SFT":
        return format_EPFL_SFT
    elif dataset_name in ["MMLU", "openQA", "mathQA", "scienceQA", "tal"]:
        return format_MCQA
    elif dataset_name == "helpSteer":
        return format_helpSteer
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")