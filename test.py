import torch
from new_model import ResUNet,ResUNet_flat
import os
import numpy as np
import torchvision as tv
import argparse
from data_process import normalize,initial_projection

def parse_args():
    parser = argparse.ArgumentParser(description="parameter for test")
    parser.add_argument("--task", type=str, default="Developable", choices=["Developable", "Flatten", "Denoise"],
                        help="model type")
    parser.add_argument("--model_path", type=str, default="../Task-pretrained_model/Developable/Developable_Net-C.pth",
                        help="model type")
    parser.add_argument("--input_dir", type=str,default="../Dataset/test/",
                        help="input obj path")
    parser.add_argument("--output_dir", type=str,default="../Task-test_result/Developable/Developable_Net-C",
                        help="output obj path")
    return parser.parse_args()

transform = tv.transforms.Compose([tv.transforms.ToTensor()])

def load_model(model,model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

if __name__ == '__main__':
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.task == "Flatten":
        net=ResUNet_flat().to(device)
    else:
        net=ResUNet().to(device)

    net = load_model(net,args.model_path, device)
    with torch.no_grad():
        path = args.input_dir
        for i in os.listdir(path):
            noisy = np.load(os.path.join(path, i))
            x1, m, cen = normalize(noisy)
            x = transform(x1).float().unsqueeze(0).to(device)
            y = net.forward(x)
            out = y[0].cpu().numpy().transpose((1, 2, 0))
            if args.task == "Flatten":
                out = out + initial_projection(x1)
                predict = out * m
                predict = np.dstack((predict, np.zeros((64, 64))))
            else:
                predict = out * m + cen
            np.save(args.output_dir + "/" + i, predict)
            print(i)

# Net-CF is optimized using Net-C and then Net-F model
# python test.py --task Developable  --model_path ../Task-pretrained_model/Developable/Developable_Net-S.pth --input_dir ../Dataset/test/ --output_dir ../Task-test_result/Developable/Developable_Net-S
# python test.py --task Developable  --model_path ../Task-pretrained_model/Developable/Developable_Net-C.pth --input_dir ../Dataset/test/ --output_dir ../Task-test_result/Developable/Developable_Net-C
# python test.py --task Developable  --model_path ../Task-pretrained_model/Developable/Developable_Net-F.pth --input_dir ../Task-test_result/Developable/Developable_Net-C --output_dir ../Task-test_result/Developable/Developable_Net-CF

# python test.py --task Flatten  --model_path ../Task-pretrained_model/Flatten/Flatten_Net.pth --input_dir ../Dataset/test/ --output_dir ../Task-test_result/Flatten/Flatten_Net
# python test.py --task Flatten  --model_path ../Task-pretrained_model/Flatten/Flatten_Net-W.pth --input_dir ../Dataset/test/ --output_dir ../Task-test_result/Flatten/Flatten_Net-W

# python test.py --task Denoise  --model_path ../Task-pretrained_model/Denoise/Denoise_0.001.pth --input_dir ../Task-test_result/Denoise/evaluate_testset/noise-0.001 --output_dir ../Task-test_result/Denoise/test_result/noise-0.001
# python test.py --task Denoise  --model_path ../Task-pretrained_model/Denoise/Denoise_0.005.pth --input_dir ../Task-test_result/Denoise/evaluate_testset/noise-0.005 --output_dir ../Task-test_result/Denoise/test_result/noise-0.005
# python test.py --task Denoise  --model_path ../Task-pretrained_model/Denoise/Denoise_0.010.pth --input_dir ../Task-test_result/Denoise/evaluate_testset/noise-0.010 --output_dir ../Task-test_result/Denoise/test_result/noise-0.010
# python test.py --task Denoise  --model_path ../Task-pretrained_model/Denoise/Denoise_0.015.pth --input_dir ../Task-test_result/Denoise/evaluate_testset/noise-0.015 --output_dir ../Task-test_result/Denoise/test_result/noise-0.015
