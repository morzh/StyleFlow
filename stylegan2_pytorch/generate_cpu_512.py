import torch
from torchvision import utils
from model import Generator
import matplotlib.pyplot as plt
import time



latent = 512
n_mlp = 8
size = 512
channel_multiplier = 2
truncation = 0.75
truncation_mean = 4096
rand_sample = 1

ckpt = '/home/morzh/work/stylegan2-pytorch/checkpoint/ffhq-512-avg-tpurun1.pt'

device = torch.device("cpu")
g_ema = Generator(size, latent, n_mlp, channel_multiplier=channel_multiplier).to(device)
checkpoint = torch.load(ckpt)

g_ema.load_state_dict(checkpoint["g_ema"])

if truncation < 1:
    with torch.no_grad():
        mean_latent = g_ema.mean_latent(truncation_mean)
else:
    mean_latent = None

samples_number = 2e2
overall_time = 0.0

with torch.no_grad():
    g_ema.eval()
    for i in range(samples_number):
        sample_z = torch.randn(rand_sample, latent, device=device)
        time_start = time.time()
        sample, _ = g_ema([sample_z], truncation=truncation, truncation_latent=mean_latent)
        time_end = time.time()
        inference_time = time_end - time_start
        print('time for inference is', inference_time, 'seconds')
        overall_time += inference_time
        img = sample[0].permute(1, 2, 0)
        img = (img + 1.0) / 2.0
        img = torch.clamp(img, 0.0, 1.0)
        plt.figure(figsize=(11, 7))
        plt.imshow(img)
        plt.tight_layout()
        plt.show()

print('mean inference time is', overall_time / samples_number)