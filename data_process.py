"数据预处理"

import numpy as np

def normalize(pc):
    centroid = np.mean(pc, axis=(0, 1))
    pc -= centroid
    m = np.max(np.linalg.norm(pc, axis=2))
    pc /= m
    return pc, m, centroid

#init 3d-2d
def initial_projection(noisy):
    left = noisy[:-1, :, :]
    right = noisy[1:, :, :]
    top = noisy[:, :-1, :]
    down = noisy[:, 1:, :]
    x = right - left
    y = down - top
    x1 = np.mean(np.linalg.norm(x, axis=2).sum(0))
    y1 = np.mean(np.linalg.norm(y, axis=2).sum(1))
    x2 = np.linspace(-x1 / 2, x1 / 2, num=64)
    y2 = np.linspace(-y1 / 2, y1 / 2, num=64)
    new = np.meshgrid(x2, y2)
    new_mesh=np.dstack((new[0],new[1]))
    return new_mesh

#use for dataaug
def random_rotation_matrix_diag_qr(low,high):
    random_matrix = np.random.uniform(low, high, size=(3, 3))  # 默认为0.5
    np.fill_diagonal(random_matrix, 1)  # 保证主分量的值
    Q, R = np.linalg.qr(np.random.rand(3, 3))  # 这里是正交矩阵和上三角矩阵，QR分解
    random_matrix = np.dot(random_matrix, Q)
    return random_matrix

#save mesh
def save_mesh(name,points_n,quads):
    with open(name, "w") as f:
        for p in points_n:
            f.write("v {} {} {}\n".format(p[0], p[1], p[2]))
        for q in quads:
            f.write("f {} {} {} {}\n".format(q[0], q[1], q[2], q[3]))
#mesh quads
def quads():
    quads = []
    N = 64
    for i in range(0, N - 1):
        for j in range(1, N):
            quads.append((i * N + j, i * N + j + 1, (i + 1) * N + j + 1, (i + 1) * N + j))
    return quads