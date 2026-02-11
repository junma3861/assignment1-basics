import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.linear3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        out = self.linear1(x)
        out = out * self.sigmoid(out)
        out = self.linear3(x) * out
        out = self.linear2(out)
        return out