import torch
from new_model import ResUNet
import numpy as np
import torchvision as tv
from data_process import normalize
import os

transform = tv.transforms.Compose([tv.transforms.ToTensor()])

if __name__ == '__main__':
##########################use NET-C to optimize the dataset,train to NET-CF###############################
    
#parameters
    ori_path = "../Dataset/"
    output_dir = "../Dataset_opt2/"
    model_path = "../Task-pretrained_model\Developable/Developable_Net-C.pth"

#model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ResUNet().to(device)
    checkpoint = torch.load(model_path, map_location=net.device)
    net.load_state_dict(checkpoint)
    net.eval()

#forward
    for root, dirs, files in os.walk(ori_path):
       for file in files:
            file_path=os.path.join(root, file)
            noisy = np.load(file_path)

            x, m, cen = normalize(noisy)
            x = transform(x).float().unsqueeze(0).to(device)
            y = net.forward(x)
            out = y[0].cpu().numpy().transpose((1, 2, 0))
            predict, m, cen=normalize(out)

            opt_file_path=file_path.replace(ori_path, output_dir)
            opt_file_dir_path = os.path.dirname(opt_file_path)
            if not os.path.exists(opt_file_dir_path):
                os.makedirs(opt_file_dir_path)

            np.save(opt_file_path, predict)

