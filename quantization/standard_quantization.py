import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# pip install git+https://github.com/huggingface/accelerate.git
# pip install bitsandbytes
# TODO check if this is what is demanded to compare.

if __name__ == "__main__":

    model_path = "../model/checkpoints/training_12_06_MERGED_MCQA/models/EPFL_DPO/final"
    model_name = "training_12_06_MERGED_MCQA"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    bnb_quantization_config = BitsAndBytesConfig(load_in_4bit=True,  bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_quantization_config)

    model.save_pretrained(f"./model/checkpoints/{model_name}-quantized")
    tokenizer.save_pretrained(f"./model/checkpoints/{model_name}-quantized")
