from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

# search z score in SKLearn

pixel_norm = PixelNorm()
image = Image.open('/home/morzh/work/checker_3x3.png')
x = TF.to_tensor(image)
x.unsqueeze_(0)
norm = pixel_norm.forward(x)

print(x)
print(x.shape)
print(torch.mean(x ** 2, dim=1, keepdim=True))
print(torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8))
print(norm)
