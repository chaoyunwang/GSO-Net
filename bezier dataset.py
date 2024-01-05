import numpy as np
# from matplotlib import pyplot as plt
import random
import torch

def select_segment(min_length, max_length):
    start_point = random.uniform(0.1, 0.9 - min_length)
    end_point = start_point + random.uniform(min_length, max_length)
    return [start_point, end_point]

def getBezierInterp(p, t):
    if len(p) == 1:
        return p[0]
    return getBezierInterp([p[i]*(1-t) + p[i+1]*t for i in range(len(p)-1)], t)

# Calculate the location of points in the mesh
def location(control_points,div,n_split):
    t_list = np.linspace(select_segment(0.3,0.8)[0],select_segment(0.3,0.8)[1], num=64)
    tt_list = np.linspace(select_segment(0.3,0.8)[0],select_segment(0.3,0.8)[1], num=64)
    for i in range(div):
        t=t_list[i]
        q = [getBezierInterp(control_points[j], t) for j in range(n_split)]
        for j in range(div):
            tt=tt_list[j]
            qq = getBezierInterp(q, tt)
            xs[i][j] = qq[0]
            ys[i][j] = qq[1]
            zs[i][j] = qq[2]
    return xs,ys,zs

# #plot surface
# def plot_surface(xs, ys, zs):
#     fig = plt.figure(1, figsize=(20, 20))
#     ax = fig.add_subplot(1, 1, 1, projection='3d')
#     ax.set_top_view()
#
#     ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='jet')
#     plt.show()

#normalize
def pc_normalize(pc):
    centroid = np.mean(pc, axis=(0,1))
    pc = pc - centroid
    m=np.max(np.linalg.norm(pc, axis=2))
    pc = pc / m
    return pc

#Calculate the Gaussian curvature of the mesh
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

    # 计算余弦相似度-角度
    theta1 = torch.acos(torch.nn.functional.cosine_similarity(V1, V2, dim=1))
    theta2 = torch.acos(torch.nn.functional.cosine_similarity(V2, V3, dim=1))
    theta3 = torch.acos(torch.nn.functional.cosine_similarity(V3, V4, dim=1))
    theta4 = torch.acos(torch.nn.functional.cosine_similarity(V4, V5, dim=1))
    theta5 = torch.acos(torch.nn.functional.cosine_similarity(V5, V6, dim=1))
    theta6 = torch.acos(torch.nn.functional.cosine_similarity(V6, V1, dim=1))
    # 计算面积
    area1 = 0.5 * torch.norm(torch.cross(V1, V2, dim=-1), dim=-1)
    area2 = 0.5 * torch.norm(torch.cross(V2, V3, dim=-1), dim=-1)
    area3 = 0.5 * torch.norm(torch.cross(V3, V4, dim=-1), dim=-1)
    area4 = 0.5 * torch.norm(torch.cross(V4, V5, dim=-1), dim=-1)
    area5 = 0.5 * torch.norm(torch.cross(V5, V6, dim=-1), dim=-1)
    area6 = 0.5 * torch.norm(torch.cross(V6, V1, dim=-1), dim=-1)
    area_all = area1 + area2 + area3 + area4+area5+area6

    pi_tensor = torch.full_like(theta1, torch.tensor(np.pi * 2))
    gauss_arr = (pi_tensor - theta1 - theta2 - theta3 - theta4-theta5-theta6)/area_all

    gauss_arr_neg = gauss_arr.clone()
    gauss_arr_pos = gauss_arr.clone()
    gauss_arr_neg[gauss_arr_neg >= 0] = 0
    gauss_arr_pos[gauss_arr_pos <= 0] = 0

    residual_pos = torch.sum(gauss_arr_pos)
    residual_neg = torch.sum(gauss_arr_neg)
    residual_abs = torch.sum(torch.abs(gauss_arr))
    return residual_pos.item(), residual_neg.item(), residual_abs.item()

#Calculate the boundary lengeth of the surface array
def caltuator_eage(y):
    top=np.linalg.norm(y[:-1,0,:]-y[1:,0,:],axis=-1).sum()
    down=np.linalg.norm(y[:-1, -1, :] - y[1:, -1, :],axis=-1).sum()
    left = np.linalg.norm(y[0, :-1, :] - y[0, 1:, :],axis=-1).sum()
    right = np.linalg.norm(y[-1, :-1, :] - y[-1, 1:, :],axis=-1).sum()
    var=np.var([top,down,left,right])
    return var

if __name__ == '__main__':
#####################generate grid surface dataset with metric fliter and control point parameter ############################
# Parameters
    N=64
    min_split = 4
    max_split = 20
    var_threshold = 0.5
    gauss_pos_threshold = 10000
    gauss_neg_threshold = -10000
    gauss_abs_threshold = 15000
    num_meshes = 100
    output_path="./dataset_example/"

#generate mesh
    xs=np.zeros([N,N])
    ys = np.zeros([N, N])
    zs = np.zeros([N, N])
    j=0
    while j < num_meshes:
        n_split = random.randint(min_split, max_split)
        n_split1 = random.randint(n_split, max_split)
        control_points = np.array([[np.array([i + np.random.uniform(-1, 1),
                                    j + np.random.uniform(-1, 1),
                                    random.uniform(0,1) * (n_split + n_split1) / 2])
                                    for j in range(n_split)]
                                    for i in range(n_split1)])
        xs, ys, zs = location(control_points, N, n_split)
        surface_array = pc_normalize(np.dstack([xs, ys, zs]))

        #metric
        gauss_pos, gauss_neg, gauss_abs=criterion_gauss_curvature_2pi_6_area(surface_array)
        var=caltuator_eage(surface_array)
        
        #optionally fliter 
        if var > var_threshold or \
            gauss_pos > gauss_pos_threshold or \
            gauss_neg < gauss_neg_threshold or \
            gauss_abs > gauss_abs_threshold:
            continue
        mesh_path = output_path \
                    + str(j)+"_" \
                    + str(round(var,3)) +"_" \
                    + str(int(gauss_pos))+"_" \
                    + str(int(gauss_neg)) + "_" \
                    + str(int(gauss_abs)) + "_"\
                    +".npy"
        np.save(mesh_path,surface_array)
        j+=1