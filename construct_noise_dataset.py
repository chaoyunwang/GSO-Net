import os
import numpy as np
from data_process import normalize

if __name__ == '__main__':
##########################add noise to the dataset as a testset for different noise level###############################
#parameters
    path = "../Dataset/test"
    save_path="../Task-test_result\Denoise\evaluate_testset/"
    level = 0.001#noise intensity: 001,005,0.010,0.015

#operation
    for i in os.listdir(path):
        noisy = np.load(os.path.join(path, i))
        x, m, cen = normalize(noisy)

        noise = np.random.normal(loc=0, scale=level, size=noisy.shape)
        noisy_noise = x + noise
        noisy_noise = noisy_noise * m + cen

        save_dir_path=save_path+"noise-"+str(level)+"/"
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        np.save(os.path.join(save_dir_path, i), noisy_noise)
