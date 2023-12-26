import tensorflow
import torch
from pytorch_pretrained_biggan import BigGAN
from torchvision.models import resnet50
from typing import Text

def load_gan(mode,device='cpu',path: Text='.'):
    if mode=="biggan":
        gan = BigGAN.from_pretrained('biggan-deep-256').to(device)
        gan_layers = []
        for name, layer in gan.named_modules():
            if "conv" in name:
                gan_layers.append(name)
    
    elif mode=="stylegan-lsun_horse":
        pass
    return gan,gan_layers