import torch
import torch.nn as nn
import torch.optim as optim


class Quantizer:

    def __init__(self, weights, n):
        self.weights = weights
        self.n = n

    def compute_outliers(self):
        mean_weights = torch.mean(self.weights)
        var_weights = torch.var(self.weights)
        threshold = self.n * var_weights
        outliers = torch.abs(self.weights - mean_weights) >= threshold
        return outliers

    def quantize(self, s):
        quantized_weights = torch.round(self.weights / s) * s
        return quantized_weights

    def compute_gradient(self, weights, s):
        Q_weights = self.quantize(s)
        loss = torch.norm(Q_weights - weights, p=2) ** 2
        loss.backward()
        return loss

    def optimize_quantization_range(self, learning_rate, epochs):
        s = torch.tensor(1.0, requires_grad=True)
        optimizer = optim.Adam([s], lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.compute_gradient(self.weights, s)
            optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}, s: {s.item()}')

        return s

    def dequantize(self, quantized_weights, outliers):
        mask = outliers.float()
        dequantized_weights = mask * self.weights + (1 - mask) * quantized_weights
        return dequantized_weights
