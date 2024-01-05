import torch
from new_model import ResUNet
from thop import profile

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResUNet().to(device)

input = torch.randn(1, 3, 64, 64).cuda()
flops, params = profile(model, inputs=(input,))

print("FLOPs=", str(flops / 1e6) + '{}'.format("G"))
print("params=", str(params / 1e6) + '{}'.format("M"))