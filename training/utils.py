# This file contains utility functions for training the model, especially for formatting the data for SFTTrainer


def format_function(dataset_name="EPFL_SFT"):
    def format_EPFL_SFT(input):
        output_texts = []
        for i in range(len(input['prompt'])):
            text = f"### {input['prompt'][i]}\n\n\n ### Answer: {input['chosen'][i]}"
            output_texts.append(text)
        return output_texts

    def format_MCQA(input):
        output_texts = []
        for i in range(len(input['question'])):
            text = f"### {input['question'][i]}\n\n\n ### Answer: {input['answer'][i]}"
            output_texts.append(text)
        return output_texts

    def format_helpSteer():
        raise NotImplementedError("Not implemented yet")

    if dataset_name == "EPFL_SFT":
        return format_EPFL_SFT
    elif (dataset_name == "MMMLU" or dataset_name == "openQA" or dataset_name == "mathQA"):
        return format_MCQA
    elif dataset_name == "helpSteer":
        return format_helpSteer
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")