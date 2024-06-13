import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# pip install git+https://github.com/huggingface/accelerate.git
# pip install bitsandbytes
# TODO check if this is what is demanded to compare.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path")
    args = parser.parse_args()

    model_path = args.model_path

    model_parent_folder = '/'.join(model_path.split("/")[:-3])
    model_name = model_path.split("/")[-2]

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    bnb_quantization_config = BitsAndBytesConfig(load_in_4bit=True,  bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_quantization_config)

    model.save_pretrained(f"{model_parent_folder}/quantized/{model_name}")
    tokenizer.save_pretrained(f"{model_parent_folder}/quantized/{model_name}")
