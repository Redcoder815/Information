import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Learnable weight and bias
        self.weight = nn.Parameter(torch.randn(1, dtype=torch.float, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float, requires_grad=True))

    def forward(self, x):
        return x * self.weight + self.bias

# Example usage
model = MyModel()
print("Weight:", model.weight.item())
print("Bias:", model.bias.item())
