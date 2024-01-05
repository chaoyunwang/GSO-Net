# Function: evaluate the metric of the result
import numpy as np
import os
import torch
from scipy.spatial.distance import cdist

def criterion_gauss_curvature_2pi_6_area(y):
    y = torch.tensor(y)

    left = y[:-2, 1:-1, :]
    right = y[2:, 1:-1, :]
    top = y[1:-1, :-2, :]
    down = y[1:-1, 2:, :]
    center = y[1:-1, 1:-1, :]
    left_top = y[:-2, :-2, :]
    right_down = y[2:, 2:, :]

    V1 = torch.reshape(left - center, (-1, 3))
    V2 = torch.reshape(left_top - center, (-1, 3))
    V3 = torch.reshape(top - center, (-1, 3))
    V4 = torch.reshape(right - center, (-1, 3))
    V5 = torch.reshape(right_down - center, (-1, 3))
    V6 = torch.reshape(down - center, (-1, 3))

    #acos-angle
    theta1 = torch.acos(torch.nn.functional.cosine_similarity(V1, V2, dim=1))
    theta2 = torch.acos(torch.nn.functional.cosine_similarity(V2, V3, dim=1))
    theta3 = torch.acos(torch.nn.functional.cosine_similarity(V3, V4, dim=1))
    theta4 = torch.acos(torch.nn.functional.cosine_similarity(V4, V5, dim=1))
    theta5 = torch.acos(torch.nn.functional.cosine_similarity(V5, V6, dim=1))
    theta6 = torch.acos(torch.nn.functional.cosine_similarity(V6, V1, dim=1))
    #area
    area1 = 0.5 * torch.norm(torch.cross(V1, V2, dim=-1), dim=-1)
    area2 = 0.5 * torch.norm(torch.cross(V2, V3, dim=-1), dim=-1)
    area3 = 0.5 * torch.norm(torch.cross(V3, V4, dim=-1), dim=-1)
    area4 = 0.5 * torch.norm(torch.cross(V4, V5, dim=-1), dim=-1)
    area5 = 0.5 * torch.norm(torch.cross(V5, V6, dim=-1), dim=-1)
    area6 = 0.5 * torch.norm(torch.cross(V6, V1, dim=-1), dim=-1)
    area_all = area1 + area2 + area3 + area4+area5+area6

    pi_tensor = torch.full_like(theta1, torch.tensor(np.pi * 2))
    gauss_arr = (pi_tensor - theta1 - theta2 - theta3 - theta4-theta5-theta6)/area_all
    residual = torch.mean(torch.abs(gauss_arr))
    return residual.item()

def new_hausdorff_distance(set1, set2):
    """
    Calculates the Hausdorff distance between two sets of points.

    Parameters:
    set1 (ndarray): First set of points.
    set2 (ndarray): Second set of points.

    Returns:
    float: The Hausdorff distance between set1 and set2 as a percentage.
    """
    set1_reshaped = np.reshape(set1, (-1, 3))
    set2_reshaped = np.reshape(set2, (-1, 3))
    #
    euclidean_distance = cdist(set1_reshaped, set2_reshaped)
    h2 = np.mean(np.min(euclidean_distance, axis=0))
    # Calculate the diagonal length of the bounding box of set1
    min_vals = np.min(set1_reshaped, axis=0)
    max_vals = np.max(set1_reshaped, axis=0)
    bounding_box_diag = np.sqrt(np.sum((max_vals - min_vals) ** 2))
    # Convert Hausdorff distance to percentage
    hausdorff_pct = h2 / bounding_box_diag * 100
    return hausdorff_pct

def mesh_normal(y):
    """
    Calculate the normal vectors of a mesh.

    Parameters:
    y (ndarray): Input mesh coordinates.

    Returns:
    ndarray: Normal vectors of the mesh.
    """
    left_top = y[:-1, :-1, :]
    right_top = y[1:, :-1, :]
    left_down = y[:-1, 1:, :]
    right_down = y[1:, 1:, :]

    v1 = left_top - right_down
    v2 = right_top - left_down

    normal = np.cross(v1, v2, axis=-1)
    return normal

def cross_matrix(array1, array2):
    """
    Calculate the average angle in degrees between two arrays of vectors.

    Parameters:
    array1 (numpy.ndarray): First array of vectors.
    array2 (numpy.ndarray): Second array of vectors.

    Returns:
    float: Average angle in degrees between the vectors.
    """
    dot_product = np.sum(array1 * array2, axis=-1)
    norm_array1 = np.linalg.norm(array1, axis=-1)
    norm_array2 = np.linalg.norm(array2, axis=-1)
    cosine_angles = dot_product / (norm_array1 * norm_array2)
    cosine_angles = np.clip(cosine_angles, -1, 1)
    radian_angles = np.arccos(cosine_angles)
    degree_angles = np.rad2deg(radian_angles)
    return np.mean(degree_angles)

def criterion_2d_3d(y1, y2):
    """
    Calculate the criterion for 2D-3D evaluation.

    Args:
        y1 (list): The first input tensor.
        y2 (list): The second input tensor.

    Returns:
        float: The calculated criterion value.

    """
    y1 = torch.tensor(y1).unsqueeze(0)
    y2 = torch.tensor(y2).unsqueeze(0)

    left_top_array_1 = y1[:, :-1, :-1, :]
    right_top_array_1 = y1[:, 1:, :-1, :]
    left_bottom_array_1 = y1[:, :-1, 1:, :]
    right_bottom_array_1 = y1[:, 1:, 1:, :]

    left_top_array_2 = y2[:, :-1, :-1, :]
    right_top_array_2 = y2[:, 1:, :-1, :]
    left_bottom_array_2 = y2[:, :-1, 1:, :]
    right_bottom_array_2 = y2[:, 1:, 1:, :]

    v02_1 = left_top_array_1 - right_bottom_array_1
    v13_1 = left_bottom_array_1 - right_top_array_1

    v02_2 = left_top_array_2 - right_bottom_array_2
    v13_2 = left_bottom_array_2 - right_top_array_2

    resdiual1 = torch.norm(v02_1, p=2, dim=3) - torch.norm(v02_2, p=2, dim=3)  # equal
    resdiual2 = torch.norm(v13_1, p=2, dim=3) - torch.norm(v13_2, p=2, dim=3)  # equal
    resdiual3 = (v02_1 * v13_1).sum(-1) - (v02_2 * v13_2).sum(-1)
    resdiual = torch.mean(torch.abs(resdiual1) + torch.abs(resdiual2) + torch.abs(resdiual3))

    return resdiual.item()

def init_2d(img):
    """
    Initialize a 2D meshgrid based on the given image.

    Parameters:
    img (ndarray): Input image.

    Returns:
    ndarray: 2D meshgrid.
    """
    left=img[:-1,:,:]
    right=img[1:,:,:]
    top=img[:,:-1,:]
    down=img[:,1:,:]
    x=right-left
    y=down-top

    x1=np.mean((np.linalg.norm(x,axis=2).sum(0)))
    y1=np.mean((np.linalg.norm(y,axis=2).sum(1)))
    x2 = np.linspace(-x1 / 2, x1 / 2, num=64)
    y2 = np.linspace(-y1 / 2, y1 / 2, num=64)
    new=np.meshgrid(x2,y2)
    add=np.zeros((64,64))
    new_mesh=np.dstack((new[0],new[1],add))
    return new_mesh

def loss_gc(opt_path):
    results = []
    for i in os.listdir(opt_path):
        opt_i=np.load(os.path.join(opt_path, i))
        loss_gauss = criterion_gauss_curvature_2pi_6_area(opt_i)
        results.append(loss_gauss)
    return sum(results)/len(results)

def K_A_r_a(opt_path,input_path):
    results = []
    for i in os.listdir(opt_path):
        opt_i=np.load(os.path.join(opt_path, i))
        input_i= np.load(os.path.join(input_path, i))
        input_ag=np.sum(abs(criterion_gauss_curvature_2pi_6_area(input_i)))
        opt_ag=np.sum(abs(criterion_gauss_curvature_2pi_6_area(opt_i)))
        rate=(input_ag-opt_ag)/input_ag
        results.append(rate)
    return sum(results)/len(results)

def d_H_a(opt_path,input_path):
    results = []
    for i in os.listdir(opt_path):
        opt_i = np.load(os.path.join(opt_path, i))
        input_i = np.load(os.path.join(input_path, i))
        rate=new_hausdorff_distance(input_i,opt_i)
        results.append(rate)
    return sum(results)/len(results)

def loss_iso_cell(opt_path,input_path):
    results=[]
    for i in os.listdir(opt_path):
        opt_i = np.load(os.path.join(opt_path, i))
        input_i = np.load(os.path.join(input_path, i))
        iso2 = criterion_2d_3d(input_i, opt_i)
        results.append(iso2)
    return  sum(results) / len(results)

def normal_angle_difference(opt_path,input_path,ori_path):
    results_noisy = []
    results_opt = []
    results_rate = []
    for i in os.listdir(opt_path):
        ori = np.load(os.path.join(ori_path, i))
        noisy = np.load(os.path.join(input_path, i))
        opt = np.load(os.path.join(opt_path, i))

        y_ori= mesh_normal(ori)
        y_noisy = mesh_normal(noisy)
        y_opt = mesh_normal(opt)

        degree_noisy = cross_matrix(y_ori, y_noisy)
        degree_opt = cross_matrix(y_ori, y_opt)
        degree_rate = (degree_noisy - degree_opt) / degree_noisy

        results_noisy.append(degree_noisy)
        results_opt.append(degree_opt)
        results_rate.append(degree_rate)
    return sum(results_noisy) / len(results_noisy), sum(results_opt) / len(results_opt), sum(results_rate) / len(results_rate)

if __name__ == '__main__':

# ###########Developable###########
#     input_path = "../Dataset/test/"
#     opt_path = "../Task-test_result/Developable/Developable_Net-S" #Developable_TNO,Developable_Net-S,,Developable_Net-C,Developable_Net-F
# # #loss_gc
#     gc=loss_gc(opt_path)
#     print("opt_loss_gc:", gc)
# # #K_A_r_a
#     rate=K_A_r_a(opt_path,input_path)
#     print("opt_K_A_r_a:", rate)
# #d_H_a
#     dh=d_H_a(opt_path,input_path)
#     print("opt_d_H_a:", dh)

# # ###########Flatten###########
#     input_path = "../Dataset/test/"
#     opt_path = "../Task-test_result/Flatten/Flatten_Net-W" #Flatten_Init,Flatten_TNO,Flatten_Net,Flatten_Net-W
# #loss_iso_cell
#     lic=loss_iso_cell(opt_path, input_path)
#     print("opt_loss_iso_cell:", lic)

# ##########Denoise###########
    ori_path = "../Dataset/test/"#ori smmoth mesh
    input_path = "../Task-test_result/Denoise/evaluate_testset/noise-0.010"#add noise,noise-0.001,noise-0.005,noise-0.010,noise-0.015
    opt_path = "../Task-test_result/Denoise/test_result/noise-0.010" #denoise, noise-0.001,noise-0.005,noise-0.010,noise-0.015
#noise angle difference
    noise,opt,opt_rate=normal_angle_difference(ori_path,input_path,opt_path)
    print("avg-noisy:", noise)
    print("avg-opt:", opt)
    print("avg-rate:", opt_rate)