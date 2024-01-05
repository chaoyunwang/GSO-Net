import numpy as np
import torch
from torch import nn

class Criterion_loss():
    def __init__(self):
        super(Criterion_loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.mae = nn.L1Loss(reduction="sum")
        self.huber=nn.HuberLoss(reduction="sum",delta=0.0001)#默认为1，可以设置为0.1，0.001

    def criterion(self, y, d):
        return self.mse(y, d)/y.size(0)

    def criterion_huber(self,y,d):
        return self.huber(y, d)

    def criterion_logcos(self,y,d):

        loss = torch.log(torch.cosh(y - d))
        return torch.sum(loss)

    def criterion_mae(self, y, d):
        return self.mae(y, d)/y.size(0)

#fairness
    def criterion_fairness_pow2(self, y):
        y1 = y.permute(0, 2, 3, 1)
        y2 = y.permute(0, 3, 2, 1)

        resdiual = torch.tensor(0.).cuda()
        for y3 in [y1, y2]:
            left_array = y3[:, :-2, :, :]
            right_array = y3[:, 2:, :, :]
            left_right = y3[:, 1:-1, :, :]
            resdiual += torch.sum(torch.pow(left_array + right_array - 2 * left_right,2))
        return resdiual/y.size(0)

    def criterion_fairness_pow_conv_1x3(self,y):  # (16, 3, 64, 64)
        conv_filter = torch.tensor([[-1., 2., -1.]]).unsqueeze(0).unsqueeze(0)
        # 复制滤波器以应用于所有通道
        conv_filter = conv_filter.repeat(y.size(1), 1, 1, 1)

        # 创建卷积层
        conv = nn.Conv2d(y.size(1), y.size(1), kernel_size=(1, 3), padding=(0, 0), groups=y.size(1), bias=False)
        conv.weight.data = conv_filter
        conv.weight.requires_grad = False

        conv = conv.cuda()

        # 计算卷积
        residual_x = conv(y)
        residual_y = conv(y.permute(0, 1, 3, 2))
        # 计算损失
        residual = torch.pow(residual_x, 2) + torch.pow(residual_y, 2)
        residual_loss = torch.sum(residual) / y.size(0)
        return residual_loss

    def criterion_fairness_pow2_conv_3X3(self,y):  # (16, 3, 64, 64)
        conv_filter = torch.tensor([
            [0., -1., 0.],
            [-1., 4., -1.],
            [0., -1., 0.]
        ]).unsqueeze(0).unsqueeze(0)
        # 复制滤波器以应用于所有通道
        conv_filter = conv_filter.repeat(y.size(1), 1, 1, 1)
        # 创建卷积层
        conv = nn.Conv2d(y.size(1), y.size(1), kernel_size=(3, 3), padding=(0, 0), groups=y.size(1), bias=False)
        conv.weight.data = conv_filter
        conv.cuda()
        # 计算卷积
        residual = conv(y)
        # 计算损失
        residual_loss = torch.sum(residual**2)
        return residual_loss/y.size(0)

#isometry3d-2d ,y1:3d,y2:2d
    def criterion_2d_3d_pow2_dim2(self, y1,y2):
        y1 = y1.permute(0, 2, 3, 1)
        y2 = y2.permute(0, 2, 3, 1)

        y3=torch.zeros((y2.shape[0],y2.shape[1],y2.shape[2],1)).cuda()
        y2=torch.cat((y2,y3),dim=3)

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
        resdiual3 = (v02_1 * v13_1).sum(-1)-(v02_2 * v13_2).sum(-1)
        resdiual=torch.mean(torch.pow(resdiual1,2)+torch.pow(resdiual2,2)+torch.pow(resdiual3,2))

        return resdiual

    def compute_gauss_curvature_weights(self, y):
        # y = y.permute(0, 2, 3, 1)
        left = y[:, :-2, 1:-1, :]
        right = y[:, 2:, 1:-1, :]
        top = y[:, 1:-1, :-2, :]
        down = y[:, 1:-1, 2:, :]
        center = y[:, 1:-1, 1:-1, :]
        left_top = y[:, :-2, :-2, :]
        right_down = y[:, 2:, 2:, :]

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
        area_all = area1 + area2 + area3 + area4 + area5 + area6

        pi_tensor = torch.full_like(theta1, torch.tensor(np.pi * 2))
        gauss_arr = (pi_tensor - theta1 - theta2 - theta3 - theta4 - theta5 - theta6) / area_all
        gauss_arr = gauss_arr.reshape(left.shape[0], 1, left.shape[1], left.shape[2])

        #pading
        gauss_arr = torch.nn.functional.pad(gauss_arr, (0, 1, 0, 1))
        # normalize
        min_val = gauss_arr.reshape(gauss_arr.shape[0], gauss_arr.shape[1], -1).min(-1)[0]
        max_val = gauss_arr.reshape(gauss_arr.shape[0], gauss_arr.shape[1], -1).max(-1)[0]
        gauss_weights = (gauss_arr - min_val.unsqueeze(-1).unsqueeze(-1)) / (max_val - min_val).unsqueeze(-1).unsqueeze(
            -1)
        # # 逆归一化
        gauss_weights = 1 - gauss_weights
        gauss_weights = gauss_weights.squeeze(1)

        return gauss_weights

    def criterion_2d_3d_pow2_dim2_gaussweight(self, y1, y2):
        y1 = y1.permute(0, 2, 3, 1)
        # (16,2,64,64)
        y2 = y2.permute(0, 2, 3, 1)

        y3 = torch.zeros((y2.shape[0], y2.shape[1], y2.shape[2], 1)).cuda()
        y2 = torch.cat((y2, y3), dim=3)

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

        gauss_weights = self.compute_gauss_curvature_weights(y1)
        resdiual = torch.sum(gauss_weights * (torch.pow(resdiual1, 2) + torch.pow(resdiual2, 2) + torch.pow(resdiual3, 2)))
        return resdiual / y1.size(0)

    def criterion_gauss_curvature_egf_simplfy(self,pred):  # (16, 3,64, 64)
        # First Derivatives
        Xu, Xv = torch.gradient(pred, dim=(2, 3))

        # Normal vector
        N_normalized = nn.functional.normalize(torch.cross(Xu, Xv, dim=1), p=2, dim=1)

        # First fundamental form coefficients
        E = (Xu * Xu).sum(dim=1)
        F = (Xu * Xv).sum(dim=1)
        G = (Xv * Xv).sum(dim=1)

        # Second Derivatives
        Xuu, Xuv = torch.gradient(Xu, dim=(2, 3))
        Xvu, Xvv = torch.gradient(Xv, dim=(2, 3))

        # Second fundamental form coefficients
        L = (N_normalized * Xuu).sum(dim=1)
        M = (N_normalized * Xuv).sum(dim=1)
        N = (N_normalized * Xvv).sum(dim=1)

        # Gaussian curvature
        K = (L * N - M * M) / (E * G - F * F)
        residual = torch.mean(torch.abs(K))
        # residual = torch.mean(torch.pow(K,2))
        return residual

    def criterion_gauss_curvature_2pi_triangel_noarea_conv3x3(self,y):
        # y = y.permute(0, 2, 3, 1)
        conv_filter_1 = torch.tensor([
            [0., 0., 0.],
            [-1., 1., 0.],
            [0., 0., 0.]
        ]).unsqueeze(0).unsqueeze(0).repeat(y.size(1), 1, 1, 1)
        conv_filter_2 = torch.tensor([
            [-1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.]
        ]).unsqueeze(0).unsqueeze(0).repeat(y.size(1), 1, 1, 1)
        conv_filter_3 = torch.tensor([
            [0., -1., 0.],
            [0., 1., 0.],
            [0., 0., 0.]
        ]).unsqueeze(0).unsqueeze(0).repeat(y.size(1), 1, 1, 1)
        conv_filter_4 = torch.tensor([
            [0., 0., 0.],
            [0., 1., -1.],
            [0., 0., 0.]
        ]).unsqueeze(0).unsqueeze(0).repeat(y.size(1), 1, 1, 1)
        conv_filter_5 = torch.tensor([
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., -1.]
        ]).unsqueeze(0).unsqueeze(0).repeat(y.size(1), 1, 1, 1)
        conv_filter_6 = torch.tensor([
            [0., 0., 0.],
            [0., 1., 0.],
            [0., -1., 0.]
        ]).unsqueeze(0).unsqueeze(0).repeat(y.size(1), 1, 1, 1)

        conv1 = nn.Conv2d(y.size(1), y.size(1), kernel_size=(3, 3), padding=(0, 0), groups=y.size(1), bias=False)
        conv1.weight.data = conv_filter_1
        conv2 = nn.Conv2d(y.size(1), y.size(1), kernel_size=(3, 3), padding=(0, 0), groups=y.size(1), bias=False)
        conv2.weight.data = conv_filter_2
        conv3 = nn.Conv2d(y.size(1), y.size(1), kernel_size=(3, 3), padding=(0, 0), groups=y.size(1), bias=False)
        conv3.weight.data = conv_filter_3
        conv4 = nn.Conv2d(y.size(1), y.size(1), kernel_size=(3, 3), padding=(0, 0), groups=y.size(1), bias=False)
        conv4.weight.data = conv_filter_4
        conv5 = nn.Conv2d(y.size(1), y.size(1), kernel_size=(3, 3), padding=(0, 0), groups=y.size(1), bias=False)
        conv5.weight.data = conv_filter_5
        conv6 = nn.Conv2d(y.size(1), y.size(1), kernel_size=(3, 3), padding=(0, 0), groups=y.size(1), bias=False)
        conv6.weight.data = conv_filter_6

        conv1.cuda()
        conv2.cuda()
        conv3.cuda()
        conv4.cuda()
        conv5.cuda()
        conv6.cuda()

        V1 = torch.reshape(conv1(y).permute(0, 2, 3, 1), (-1, 3))
        V2 = torch.reshape(conv2(y).permute(0, 2, 3, 1), (-1, 3))
        V3 = torch.reshape(conv3(y).permute(0, 2, 3, 1), (-1, 3))
        V4 = torch.reshape(conv4(y).permute(0, 2, 3, 1), (-1, 3))
        V5 = torch.reshape(conv5(y).permute(0, 2, 3, 1), (-1, 3))
        V6 = torch.reshape(conv6(y).permute(0, 2, 3, 1), (-1, 3))

        # acos-angle
        theta1 = torch.acos(torch.nn.functional.cosine_similarity(V1, V2, dim=1))
        theta2 = torch.acos(torch.nn.functional.cosine_similarity(V2, V3, dim=1))
        theta3 = torch.acos(torch.nn.functional.cosine_similarity(V3, V4, dim=1))
        theta4 = torch.acos(torch.nn.functional.cosine_similarity(V4, V5, dim=1))
        theta5 = torch.acos(torch.nn.functional.cosine_similarity(V5, V6, dim=1))
        theta6 = torch.acos(torch.nn.functional.cosine_similarity(V6, V1, dim=1))

        pi_tensor = torch.full_like(theta1, torch.tensor(np.pi * 2))
        gauss_arr = pi_tensor - theta1 - theta2 - theta3 - theta4 - theta5 - theta6
        residual = torch.sum(gauss_arr**2)
        return residual/y.size(0)

    def criterion_gauss_curvature_2pi_triangel_area(self,y):
        y = y.permute(0, 2, 3, 1)

        left = y[:, :-2, 1:-1, :]
        right = y[:, 2:, 1:-1, :]
        top = y[:, 1:-1, :-2, :]
        down = y[:, 1:-1, 2:, :]
        center=y[:, 1:-1, 1:-1, :]
        left_top = y[:, :-2, :-2, :]
        right_down = y[:, 2:, 2:, :]

        V1 = torch.reshape(left-center, (-1, 3))
        V2 = torch.reshape(left_top - center, (-1, 3))
        V3 = torch.reshape(top-center, (-1, 3))
        V4 = torch.reshape(right-center, (-1, 3))
        V5 = torch.reshape(right_down - center, (-1, 3))
        V6 = torch.reshape(down-center, (-1, 3))

        #arcos-angle
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
        return residual