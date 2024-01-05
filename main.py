import torch
from data import NoisyBSDSDataset
import nntools as nt
from new_model import ResUNet ,ResUNet_flat
import argparse

#add arguments
def parse():
    parser = argparse.ArgumentParser(description='GSO-Net')
    parser.add_argument('--task', type=str,default="Developable", choices=["Developable","Flatten","Denoise"],help='task type')
    parser.add_argument('--root_dir', type=str,default="../Dataset", help='root directory of dataset')
    parser.add_argument('--pretrain_model', type=str,default=None,help="model of pretrain")#"./pretrained/pretrained_auto_en-de.pth"
    parser.add_argument('--output_dir', type=str,default="../Developable",help='directory of saved checkpoints')
    parser.add_argument('--num_epochs', type=int,default=4000, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,help='learning rate for training')
    parser.add_argument('--batch_size', type=int, default=256)
    return parser.parse_args()

def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #DATASET
    train_set = NoisyBSDSDataset(args.root_dir,task=args.task,mode='train')
    test_set = NoisyBSDSDataset(args.root_dir,task=args.task, mode='test')

    # net
    if args.task == "Flatten":
        net = ResUNet_flat().to(device)
    else:
        net = ResUNet().to(device)

    # optimizer
    optimzer = torch.optim.AdamW(net.parameters(), lr=args.lr)

    # experiment
    exp = nt.Experiment(args.task,net, train_set, test_set, optimzer, batch_size=args.batch_size,
                        output_dir=args.output_dir,load_model=args.pretrain_model)
    exp.run(num_epochs=args.num_epochs)

if __name__ == '__main__':
    args = parse()
    print(args)
    run(args)
    # os.system("/usr/bin/shutdown")

#python main.py --task Developable --root_dir ../Dataset --pretrain_model ./pretrained/pretrained_auto_en-de.pth --output_dir ../Developable/ --num_epochs 4000 --lr 1e-4 --batch_size 256
#python main.py --task Flatten --root_dir ../Dataset --output_dir ../Flatten/ --num_epochs 1000 --lr 1e-4 --batch_size 256
#python main.py --task Denoise --root_dir ../Dataset --pretrain_model ./pretrained/pretrained_auto_en-de.pth --output_dir ../Denoise/ --num_epochs 1000 --lr 1e-4 --batch_size 256