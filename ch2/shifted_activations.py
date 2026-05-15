import torch
import torch.nn as nn

# ============================================================
# Quantizer classes (Listings 2.8 and 2.9)
# ============================================================

class SymmetricQuantizer:
    def __init__(self, bits=8):
        self.q_max = (1 << (bits - 1)) - 1
        self.q_min = -self.q_max
        self.scale = 1.0
        self.zero_point = 0

    def calibrate(self, tensor):
        abs_max = tensor.abs().max()
        self.scale = (abs_max / self.q_max).item() if abs_max > 0 else 1.0

    def quantize(self, tensor):
        scaled = tensor / self.scale
        rounded = torch.round(scaled)
        return torch.clamp(rounded, self.q_min, self.q_max).to(torch.int8)

    def dequantize(self, q_tensor):
        return q_tensor.float() * self.scale


class AffineQuantizer:
    def __init__(self, bits=8):
        self.q_min = 0
        self.q_max = (1 << bits) - 1
        self.scale = 1.0
        self.zero_point = 0

    def calibrate(self, tensor):
        r_min = tensor.min().item()
        r_max = tensor.max().item()
        r_min = min(r_min, 0.0)
        r_max = max(r_max, 0.0)

        real_range = r_max - r_min
        int_range = self.q_max - self.q_min

        if real_range == 0:
            self.scale, self.zero_point = 1.0, 0
        else:
            self.scale = real_range / int_range
            initial_z = self.q_min - (r_min / self.scale)
            self.zero_point = int(round(initial_z))
            self.zero_point = max(self.q_min, min(self.q_max, self.zero_point))

    def quantize(self, tensor):
        scaled = (tensor / self.scale) + self.zero_point
        rounded = torch.round(scaled)
        return torch.clamp(rounded, self.q_min, self.q_max).to(torch.uint8)

    def dequantize(self, q_tensor):
        return self.scale * (q_tensor.float() - self.zero_point)


# ============================================================
# Linear8bit layer (Listing 2.10)
# ============================================================

class Linear8bit(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.weight_quantizer = SymmetricQuantizer(bits=8)
        self.input_quantizer = AffineQuantizer(bits=8)

    def forward(self, x):
        self.input_quantizer.calibrate(x)
        x_q = self.input_quantizer.quantize(x)

        self.weight_quantizer.calibrate(self.weight)
        w_q = self.weight_quantizer.quantize(self.weight)

        term1 = torch.mm(x_q.float(), w_q.float().t())
        sum_w_q = torch.sum(w_q.float(), dim=1)
        term2 = self.input_quantizer.zero_point * sum_w_q

        y_int = term1 - term2
        scale = self.weight_quantizer.scale * self.input_quantizer.scale
        y_out = y_int * scale

        if self.bias is not None:
            y_out = y_out + self.bias
        return y_out


# ============================================================
# Set up weights once, shared across both experiments
# ============================================================

torch.manual_seed(42)
weights_fp32 = torch.randn(32, 64) * 0.1


# ============================================================
# (Context) Original ReLU experiment — Listings 2.11–2.12
# ============================================================

inputs_relu = torch.relu(torch.randn(128, 64)) * 2.5

layer_relu = Linear8bit(in_features=64, out_features=32)
layer_relu.weight.data = weights_fp32.clone()

layer_relu_fp32 = nn.Linear(64, 32, bias=True)
layer_relu_fp32.weight.data = weights_fp32.clone()
layer_relu_fp32.bias.data = layer_relu.bias.data.clone()

with torch.no_grad():
    output_relu_fp32 = layer_relu_fp32(inputs_relu)
    output_relu_8bit = layer_relu(inputs_relu)

mae_relu = (output_relu_8bit - output_relu_fp32).abs().mean().item()
print("=" * 60)
print("ReLU experiment (for reference)")
print("=" * 60)
print(f"Zero-point Zx: {layer_relu.input_quantizer.zero_point}")
print(f"MAE vs FP32:   {mae_relu:.4f}")
print()


# ============================================================
# Listing 2.13 — Shifted activations
# ============================================================

torch.manual_seed(123)
inputs_shifted = torch.randn(128, 64) * 2.0 + 5.0       # A

layer_shifted = Linear8bit(in_features=64, out_features=32)
layer_shifted.weight.data = weights_fp32.clone()

layer_shifted_fp32 = nn.Linear(64, 32, bias=True)
layer_shifted_fp32.weight.data = weights_fp32.clone()
layer_shifted_fp32.bias.data = layer_shifted.bias.data.clone()

with torch.no_grad():
    output_fp32_ref = layer_shifted_fp32(inputs_shifted)
    output_correct  = layer_shifted(inputs_shifted)     # B

    # Naive: drop the term2 = Zx · sum(q_w) correction
    x_q = layer_shifted.input_quantizer.quantize(inputs_shifted)
    w_q = layer_shifted.weight_quantizer.quantize(layer_shifted.weight)
    term1 = torch.mm(x_q.float(), w_q.float().t())
    scale = (layer_shifted.weight_quantizer.scale *
             layer_shifted.input_quantizer.scale)
    output_naive = term1 * scale + layer_shifted.bias   # C

mae_correct = (output_correct - output_fp32_ref).abs().mean().item()
mae_naive   = (output_naive   - output_fp32_ref).abs().mean().item()

print("=" * 60)
print("Listing 2.13 — Shifted activations")
print("=" * 60)
print(f"Zero-point Zx:         {layer_shifted.input_quantizer.zero_point}")
print(f"MAE vs FP32 (correct): {mae_correct:.4f}")
print(f"MAE vs FP32 (naive):   {mae_naive:.4f}")
print(f"Ratio (naive/correct): {mae_naive / mae_correct:.1f}x")