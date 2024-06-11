import os
import pandas as pd

# nlp imports
import datasets
from transformers import AutoTokenizer


def formatting_func(example):
    text = f"{example[FIELD_NAME]}"
    return text

def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))


import matplotlib.pyplot as plt

def plot_data_lengths(tokenized_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()


def load_dataset(data_path):
    dataset = pd.read_json(data_path, lines=True)
    dataset = datasets.Dataset.from_pandas(dataset)

    return dataset


def tokenize_dataset(dataset):
    tokenized_dataset = dataset.map(generate_and_tokenize_prompt)

    return tokenized_dataset

def fix_dataset(data_path, max_length=512, suffix="_clipped"):
    dataset = load_dataset(data_path)
    tokenized_dataset = tokenize_dataset(dataset)
    lengths = [len(x['input_ids']) for x in tokenized_dataset]
    filter_and_save_datasets(data_path, lengths, max_length, suffix)


def filter_and_save_datasets(dataset_path, lengths, max_length, suffix="_clipped"):
    dataset = pd.read_json(dataset_path, lines=True)
    dataset["lengths"] = lengths
    dataset = dataset[dataset["lengths"] <= max_length]
    new_dataset_path = dataset_path.replace(".json", f"{suffix}.json")
    dataset.to_json(new_dataset_path, orient="records", lines=True)


if __name__ == "__main__":
    # change working directory to file directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2",
                                              trust_remote_code=True,
                                              padding_side='left',
                                              add_eos_token=True,
                                              add_bos_token=True,
                                              use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    FIELD_NAME = "question"
    dataset_name = "MMLU"

    parent_folder = "./"
    processed = "processed/"

    train = f"{dataset_name}_train.jsonl"
    test = f"{dataset_name}_test.jsonl"
    val = f"{dataset_name}_val.jsonl"

    prefix = f"{parent_folder}{dataset_name}/{processed}"

    train_path = f"{prefix}{train}"
    test_path = f"{prefix}{test}"
    val_path = f"{prefix}{val}"

    max_length = 512

    fix_dataset(val_path, max_length, suffix="_clipped")
    fix_dataset(test_path, max_length, suffix="_clipped")
    fix_dataset(train_path, max_length, suffix="_clipped")
