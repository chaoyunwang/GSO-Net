import os
from torchvision import transforms
import numpy as np
import torch.utils.data as data_utils
from data_process import normalize,initial_projection,random_rotation_matrix_diag_qr

class NoisyBSDSDataset(data_utils.Dataset):
    def __init__(self, root_dir, task,mode):
        super(NoisyBSDSDataset, self).__init__()
        self.task = task
        self.mode = mode
        self.images_dir = os.path.join(root_dir, mode)
        self.files = os.listdir(self.images_dir)
        self.data_noisy = []
        for file in self.files:
            noisy_path = os.path.join(self.images_dir, file)
            noisy = np.load(noisy_path)
            self.data_noisy.append(noisy)
        self.transforms = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        noisy = self.data_noisy[idx]
        if self.mode =='train':
            #dataaug
            random_matrix = random_rotation_matrix_diag_qr(low=0,high=0.3)#set parameter
            noisy = np.dot(noisy, random_matrix)

        #normalize
        noisy, m, centroid = normalize(noisy)
        if self.task == "Denoise":
            noise = np.random.normal(loc=0, scale=0.010, size=noisy.shape)#add noise
            noisy = noisy + noise
        if self.task == "Flatten":
            flatten_init=initial_projection(noisy)
            flatten_init= self.transforms(flatten_init)

        noisy = self.transforms(noisy)
        #return
        if self.task == "Flatten":
            return noisy.float(),flatten_init.float()
        else:
            return noisy.float()