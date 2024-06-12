import time
import pandas as pd
import configparser

from typing import List, Dict, Tuple, Any

# nlp imports
import torch
import datasets
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

from utils import format_function

import gc
import argparse

RANDOM_STATE = 42


def read_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def load_train_test(**kwargs):
    train_dataset = pd.read_json(f"../{DATA_FOLDER}/{kwargs['path']}/{kwargs['train_data']}", lines=True)
    test_dataset = pd.read_json(f"../{DATA_FOLDER}/{kwargs['path']}/{kwargs['test_data']}", lines=True)

    max_data_points = kwargs["max_data_points"]
    if max_data_points != "all":
        max_data_points = int(max_data_points)

        if train_dataset.shape[0] > max_data_points:
            train_dataset = train_dataset.sample(n=int(max_data_points), random_state=RANDOM_STATE)

    train_dataset = datasets.Dataset.from_pandas(train_dataset)
    test_dataset = datasets.Dataset.from_pandas(test_dataset)

    return train_dataset, test_dataset


def train_DPO(model, tokenizer, num_epochs, output_folder, dpo_config, peft_config, **kwargs):
    # loading the datasets
    train_dataset, test_dataset = load_train_test(**kwargs)

    dataset_name = kwargs["name"]

    # setting the training parameters
    training_args = dpo_config
    training_args.output_dir = f"./{output_folder}/models/{dataset_name}"
    training_args.logging_dir = f"./{output_folder}/runs/{dataset_name}"
    training_args.max_length = eval(kwargs["max_seq_length"]) + 500
    training_args.max_prompt_length = eval(kwargs["max_seq_length"])
    training_args.num_train_epochs = num_epochs

    early_stopping_callback = transformers.EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.01)

    learning_rate = kwargs["learning_rate"]
    if learning_rate != "default":
        learning_rate = float(learning_rate)
        training_args.learning_rate = learning_rate

    # creating the DPO trainer
    trainer = DPOTrainer(
        model,
        args=training_args,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback],
    )

    # training the model
    trainer.train()
    trainer.model.save_pretrained(f"./{output_folder}/models/{dataset_name}/final")
    tokenizer.save_pretrained(f"./{output_folder}/models/{dataset_name}/final")

    del train_dataset, test_dataset

    for i in range(2):
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(30)

    return 0


def train_SFT(model, tokenizer, num_epochs, output_folder, sft_config, peft_config, **kwargs):
    # loading the datasets
    train_dataset, test_dataset = load_train_test(**kwargs)

    dataset_name = kwargs["name"]
    prompt_formatting_function = format_function(dataset_name)

    response_template = "### Explanation: "
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = sft_config
    training_args.output_dir = f"./{output_folder}/models/{dataset_name}"
    training_args.logging_dir = f"./{output_folder}/runs/{dataset_name}"
    training_args.max_seq_length = eval(kwargs["max_seq_length"])
    training_args.num_train_epochs = num_epochs

    early_stopping_callback = transformers.EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.01)

    learning_rate = kwargs["learning_rate"]
    if learning_rate != "default":
        learning_rate = float(learning_rate)
        training_args.learning_rate = learning_rate

    # creating the SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        formatting_func=prompt_formatting_function,
        data_collator=collator,
        callbacks=[early_stopping_callback],
    )

    # training the model
    trainer.train()
    trainer.model.save_pretrained(f"./{output_folder}/models/{dataset_name}/final")
    tokenizer.save_pretrained(f"./{output_folder}/models/{dataset_name}/final")

    del train_dataset, test_dataset

    for i in range(2):
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(30)

    return 0


def train(config_list_names: List[str], model_configs: configparser.ConfigParser,
          data_configs: configparser.ConfigParser, peft_config: LoraConfig,
          sft_config: SFTConfig, dpo_config: DPOConfig) -> int:
    """
    Trains the base model on different configurations and saves them.
    For each configuration, the model is trained succesively on the datasets specified in its config file. The training is done in the order of the datasets in the config file.
    The model can be trained on SFT or DPO datasets. The type of dataset is specified in the config file, and the specificity of the dataset is specified in the data_info file.

    Inputs:
        model: a transformer model
        config_list_names: a list of the names of the configurations in the config file
        model_configs: a configparser object containing the model configurations
        data_configs: a configparser object containing the data configurations

    Outputs:
        0: if the training is successful
    """

    # for each configuration, train the dataset based on the configuration
    for config_name in config_list_names:

        # loading the base model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True, padding_side='left',
                                                  device_map="auto")

        # model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", trust_remote_code=True, padding_side='left', device_map="auto")

        tokenizer.pad_token = tokenizer.eos_token

        # loading training configuration of the configuration name
        config = dict(model_configs[config_name].items())
        dataset_list = eval(config["datasets"])
        epochs_per_dataset = eval(config["epochs_per_dataset"])

        # iterating on the configuration datasets
        for n, dataset_name in enumerate(dataset_list):

            # loading the dataset informations
            dataset_config = dict(data_configs[dataset_name].items())
            type = dataset_config["type"]

            epochs = epochs_per_dataset[n]

            # training the model based on the train type
            if type == "DPO":
                train_DPO(model, tokenizer, epochs, config_name, dpo_config, peft_config, **dataset_config)
            elif type == "SFT":
                train_SFT(model, tokenizer, epochs, config_name, sft_config, peft_config, **dataset_config)
            else:
                raise ValueError("Invalid training type")

            # saving the model
            # model.save_pretrained(f"{model_configs[config_name]['save_dir']}/{config_name}_{str(n)}")
            # tokenizer.save_pretrained(f"{model_configs[config_name]['save_dir']}/{config_name}_{str(n)}")

            torch.cuda.empty_cache()
            gc.collect()

        del (model, tokenizer)
    return 0


def get_configs(param_dict: Dict[str, Any]) -> (SFTConfig, DPOConfig, LoraConfig):
    sft_config = SFTConfig(
        output_dir="./output",

        per_device_train_batch_size=int(param_dict["per_device_train_batch_size_sft"]),
        per_device_eval_batch_size=int(param_dict["per_device_eval_batch_size_sft"]),
        eval_accumulation_steps=int(param_dict["eval_accumulation_steps_sft"]),
        gradient_accumulation_steps=int(param_dict["gradient_accumulation_steps_sft"]),

        bf16=True,

        learning_rate=float(param_dict["learning_rate_sft"]),
        lr_scheduler_type=param_dict["lr_scheduler_type_sft"],
        warmup_ratio=float(param_dict["warmup_ratio_sft"]),

        logging_steps=1,
        report_to="tensorboard",

        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,

        save_strategy="steps",
        eval_strategy="steps",
        save_steps=250,
        eval_steps=250,

        do_train=True,
        do_eval=True,
    )

    dpo_config = DPOConfig(
        output_dir="./output",

        beta=0.1,
        per_device_train_batch_size=int(param_dict["per_device_train_batch_size_dpo"]),
        per_device_eval_batch_size=int(param_dict["per_device_eval_batch_size_dpo"]),
        eval_accumulation_steps=int(param_dict["eval_accumulation_steps_dpo"]),
        gradient_accumulation_steps=int(param_dict["gradient_accumulation_steps_dpo"]),

        bf16=True,

        learning_rate=float(param_dict["learning_rate_dpo"]),
        lr_scheduler_type=param_dict["lr_scheduler_type_dpo"],
        warmup_ratio=float(param_dict["warmup_ratio_dpo"]),

        logging_steps=1,
        report_to="tensorboard",

        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,

        save_strategy="steps",
        eval_strategy="steps",
        save_steps=250,
        eval_steps=250,

        do_train=True,
        do_eval=True,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["Wqkv", "fc1", "fc2"]  # ["Wqkv", "fc1", "fc2" ] # ["Wqkv", "out_proj", "fc1", "fc2" ]
    )

    return peft_config, sft_config, dpo_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_list_names", type=str, nargs="+", help="List of the names of the configurations in the config file")
    args = parser.parse_args()

    DATA_FOLDER = "datasets"
    config_file = "config.ini"
    data_info_file = "dataset_info.ini"
    training_config_file = "training.ini"

    model_configs = read_config(config_file)
    data_configs = read_config(data_info_file)

    config_name = "DEFAULT"
    training_configs = read_config(training_config_file)
    peft_config, sft_config, dpo_config = get_configs(dict(training_configs[config_name].items()))

    config_list_names = args.config_list_names
    # config_list_names = ["temp"]
    train(config_list_names, model_configs, data_configs, peft_config, sft_config, dpo_config)
