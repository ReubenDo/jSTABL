import torch 
class LeakyRelu(torch.nn.Module):
    def __init__(self,half_alpha=0.01,negative_slope= 1e-2, inplace=False):
        self.half_alpha = half_alpha
        super(LeakyRelu, self).__init__()
    def forward(self,x):
        return (0.5 + self.half_alpha) * x + (0.5 - self.half_alpha) * abs(x)