from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.optim as optim

from quantization.quantizer import Quantizer


def apply_quantization_to_model(model, n, learning_rate, epochs):
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param.data
            quantizer = Quantizer(weights, n)
            outliers = quantizer.compute_outliers()

            # Optimize quantization range
            s = quantizer.optimize_quantization_range(learning_rate, epochs)

            quantized_weights = quantizer.quantize(s)
            dequantized_weights = quantizer.dequantize(quantized_weights, outliers)

            # Update model weights with de-quantized weights
            param.data.copy_(dequantized_weights)


if __name__ == "__main__":
    model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    n = 2.5
    learning_rate = 0.01
    epochs = 100

    # Apply quantization to the pretrained model
    apply_quantization_to_model(model, n, learning_rate, epochs)
    # Save the quantized model
    quantized_model_path = "quantized_phi_2_model.pth"
    torch.save(model.state_dict(), quantized_model_path)
    print("Quantized model saved successfully.")

    # Load the quantized model
    loaded_model = AutoModelForCausalLM.from_pretrained(model_id)
    loaded_model.load_state_dict(torch.load(quantized_model_path))
    print("Quantized model loaded successfully.")
