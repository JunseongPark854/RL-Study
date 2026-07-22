import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def pair(t):
    if (isinstance(t, tuple)) and len(t) == 2:
        return t
    elif (isinstance(t, int)):
        return (t,t)
    raise ValueError("Input must be an int or a tuple of length 2")

class Classic_CNN(nn.Module):
    def __init__(
            self, 
            *,
            image_size,
            dropout : float,
            channels=3, 
            base_channels=32, 
            num_blocks=3, 
            num_classes=10,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            hidden_dim = 256
                 ):
        super().__init__()
        is_poll = True
        in_ch = channels
        self.blocks = nn.ModuleList([])
        # Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True)
        # w = floor( (W_in + 2*padding - kernel_size) / stride ) + 1
        image_high, image_width = pair(image_size)
        for i in range(num_blocks): # input (B, 3, h, w)
            out_ch = base_channels * (2 ** i)
            self.blocks.append(nn.ModuleList([
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ]))
            image_high = (image_high + 2*padding - kernel_size)//stride + 1
            image_width = (image_high + 2*padding - kernel_size)//stride + 1
            if is_poll:
                image_high = image_high // 2
                image_width = image_width // 2
            in_ch = out_ch

        in_dim = image_high*image_width*in_ch
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, img):
        x = img
        for conv, act, poll in self.blocks:
            x = conv(x)
            x = act(x)
            x = poll(x)
        x = self.head(x)
        return x