import torch
from new_model import ResUNet,ResUNet_flat
import numpy as np
import torchvision as tv
import pywavefront
import argparse
from data_process import normalize,initial_projection,quads,save_mesh

def parse_args():
    parser = argparse.ArgumentParser(description="parameter for inference")
    parser.add_argument("--task", type=str, default="Developable", choices=["Developable", "Flatten", "Denoise"],
                        help="model type")
    parser.add_argument("--model_path", type=str, default="../Task-pretrained_model-test_result/Developable/pretrained_model/Developable_Net-C.pth",
                        help="model type")
    parser.add_argument("--input_obj", type=str,default="./opt_example/Developable.obj",
                        help="input obj path")
    parser.add_argument("--output_obj", type=str,default="./opt_example/opt-Developable.obj",
                        help="output obj path")
    return parser.parse_args()

transform = tv.transforms.Compose([tv.transforms.ToTensor()])

def load_model(model,model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def process_obj(input_obj):
    obj = pywavefront.Wavefront(input_obj)
    vertices = obj.vertices
    points = np.reshape(np.array(vertices), (64, 64, 3))
    return points

if __name__ == '__main__':
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.task == "Flatten":
        net = ResUNet_flat().to(device)
    else:
        net = ResUNet().to(device)

    net = load_model(net, args.model_path, device)
    with torch.no_grad():
        noisy=process_obj(args.input_obj)
        x, m, cen = normalize(noisy)
        x = transform(x).float().unsqueeze(0).to(device)
        y = net.forward(x)
        out = y[0].cpu().numpy().transpose((1, 2, 0))
        if args.task == "Flatten":
            predict = out * m
            predict = predict + initial_projection(noisy)
            predict = np.dstack((predict, np.zeros((64, 64))))
        else:
            predict = out * m + cen
        predict=np.reshape(predict, (64 * 64, 3))
        quad = quads()
        save_mesh(args.output_obj, predict, quad)


# Net-CF is optimized using Net-C and then Net-F model

# python inference.py --task Developable --model_path ../Task-pretrained_model/Developable/Developable_Net-C.pth --input_obj ./opt_example/Developable.obj --output_obj ./opt_example/opt-Developable.obj

# python inference.py --task Flatten  --model_path ../Task-pretrained_model/Flatten/Flatten_Net-W.pth --input_obj ./opt_example/Flatten.obj --output_obj ./opt_example/opt-Flatten.obj

# python inference.py --task Denoise  --model_path ../Task-pretrained_model/Denoise/Denoise_0.010.pth --input_obj ./opt_example/Denoise.obj --output_obj ./opt_example/opt-Denoise.obj

