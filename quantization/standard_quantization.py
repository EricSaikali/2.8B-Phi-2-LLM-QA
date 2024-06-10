import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
from accelerate.utils import BnbQuantizationConfig
from accelerate.utils import load_and_quantize_model
from accelerate import Accelerator

# pip install git+https://github.com/huggingface/accelerate.git
# pip install bitsandbytes
# TODO check if this is what is demanded to compare.

if __name__ == "__main__":

    model_path = "microsoft/phi-2"
    model_name = "phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype='bf16',
                                                    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

    quantized_model = load_and_quantize_model(model, bnb_quantization_config=bnb_quantization_config, device_map="auto")

    accelerate = Accelerator()
    new_weights_location = f"./model/checkpoints/{model_name}-quantized-2"
    try:
        accelerate.save_model(quantized_model, new_weights_location)
    except:
        pass
    try:
        quantized_model.save_pretrained(f"./model/checkpoints/{model_name}-quantized-1")
    except:
        pass
    try:
        tokenizer.save_pretrained(f"./model/checkpoints/{model_name}-quantized-1")
    except:
        pass