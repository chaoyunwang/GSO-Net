import torch

path="../Task-pretrained_model/checkpoint_1000.pth.tar"

model=torch.load(path)

torch.save(model["Net"], "../Task-pretrained_model/pretrained.pth")