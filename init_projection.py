import numpy as np
import os
from data_process import initial_projection

##########################use init3d-3d to generate test result##############################
#parameters
input_dir="../Dataset/test"
output_dir="../Task-test_result/Flatten/Flatten_Init/"

for i in os.listdir(input_dir):
    new=initial_projection(np.load(os.path.join(input_dir,i)))
    result= np.dstack((new, np.zeros((64, 64))))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir,i),result)