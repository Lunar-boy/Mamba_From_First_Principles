import torch
import einops

torch.manual_seed(42)

device = torch.device("cpu")
dtype = torch.float64

x = torch.randn(
    2, 16, 4, 8,
    device=device,
    dtype=dtype,
    requires_grad=True,
)

# Tensor contraction
y = torch.einsum("bthp,bthp->bth", x, x)

# Tensor rearrangement
z = einops.rearrange(x, "b t h p -> b t (h p)")

# Backpropagation
loss = y.mean()
loss.backward()

print("Device:", x.device)
print("Dtype:", x.dtype)
print("Einsum shape:", y.shape)
print("Rearrange shape:", z.shape)
print("Gradient shape:", x.grad.shape)
print("Gradient finite:", torch.isfinite(x.grad).all().item())
