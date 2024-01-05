import os
import time
import torch
import numpy as np
import torch.utils.data as data_utils
from utils import Criterion_loss
from torch.utils.tensorboard import SummaryWriter

class StatsManager:
    def __init__(self):
        self.init()

    def init(self):
        self.running_loss = [0.0] * 4
        self.number_update = 0

    def accumulate(self, loss):
        self.running_loss = list(np.add(self.running_loss, loss))
        self.number_update += 1

    def summarize(self):
        return [loss / self.number_update for loss in self.running_loss] if self.number_update > 0 else [0] * 4

class Experiment:
    def __init__(self,task, net, train_set, val_set, optimizer, output_dir=None, batch_size=256, load_model=None):
        self.task, self.net, self.train_set, self.val_set, self.optimizer = task, net, train_set, val_set, optimizer
        self.batch_size = batch_size
        self.output_dir = output_dir or 'experiment_{}'.format(time.time())
        os.makedirs(self.output_dir, exist_ok=True)
        self.loss_criterion = Criterion_loss()
        self.stats_manager = StatsManager()
        self.train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        self.val_loader = data_utils.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.test_loss = 1e6
        self.load_model(load_model)
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard_logs'))

    def load_model(self, model_path):
        if model_path is not None and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.net.device)
            self.net.load_state_dict(checkpoint)

    def save_model(self, name):
        torch.save(self.net.state_dict(), os.path.join(self.output_dir, f"{self.task}_{name}.pth"))

    def run(self, num_epochs):
        self.net.train()
        for epoch in range(num_epochs):
            self.stats_manager.init()
            start_time = time.time()
            weight_2 = 1.0 * np.exp(-3 * (epoch) / 4000.0)
            weight_3 = 1.0
            weight_4 = 1.0
            if self.task == "Flatten":
                for x,init_2d in self.train_loader:
                    x ,init_2d= x.to(self.net.device),init_2d.to(self.net.device)
                    self.optimizer.zero_grad()
                    y = self.net(x)+init_2d
                    loss2 = self.loss_criterion.criterion_2d_3d_pow2_dim2_gaussweight(x, y)#criterion_2d_3d_pow2_dim2,criterion_2d_3d_pow2_dim2_gaussweight
                    loss3 = self.loss_criterion.criterion_fairness_pow_conv_1x3(y)
                    loss4 = 0
                    loss_opt=1e6*loss2+loss3

                    loss_opt.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.01)
                    self.optimizer.step()
                    loss_all = [loss_opt.item(), loss2.item(), loss3.item(), loss4]
                    self.stats_manager.accumulate(loss_all)
            else:
                for x in self.train_loader:
                    print(11)
                    x = x.to(self.net.device)
                    self.optimizer.zero_grad()
                    y = self.net(x)
                    # loss1 = self.loss_criterion.criterion_mae(y, x)
                    loss2 = self.loss_criterion.criterion(y, x)
                    loss3 = self.loss_criterion.criterion_fairness_pow_conv_1x3(y)
                    loss4 = self.loss_criterion.criterion_gauss_curvature_2pi_triangel_area(y)
                    if self.task == "Denoise":
                        loss_opt = loss2 + 10*loss3
                    elif self.task == "Developable":
                        loss_opt = weight_2 * loss2 + weight_3 * loss3 + weight_4 * loss4

                    loss_opt.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.01)
                    self.optimizer.step()
                    loss_all = [loss_opt.item(), loss2.item(), loss3.item(), loss4.item()]
                    self.stats_manager.accumulate(loss_all)

            training_summary = self.stats_manager.summarize()
            evaluation_summary = self.evaluate(weight_2,weight_3,weight_4)
            #TensorBoard logging
            for idx, loss in enumerate(training_summary):
                self.writer.add_scalar(f'Training/Loss_{idx + 1}', loss, epoch)
            for idx, loss in enumerate(evaluation_summary):
                self.writer.add_scalar(f'Evaluation/Loss_{idx + 1}', loss, epoch)

            print("Epoch {} | Time: {:.2f}s | Training Loss: (loss_opt: {:.6f}, loss2: {:.6f}, loss3: {:.6f}, loss4: {:.6f})".format(
                    epoch + 1, time.time() - start_time, *training_summary))
            print("                         | Evaluation Loss: (loss_opt: {:.6f}, loss2: {:.6f}, loss3: {:.6f}, loss4: {:.6f})".format(
                                                        *evaluation_summary))
            #save model 
            if self.test_loss > evaluation_summary[0]:
                self.test_loss = evaluation_summary[0]
                self.save_model("best")
            if epoch % 500 == 0:
                self.save_model(epoch)

        self.writer.close()

    def evaluate(self,weight_2,weight_3,weight_4):
        self.stats_manager.init()
        self.net.eval()
        with torch.no_grad():
            if self.task == "Flatten":
                for x,init_2d in self.val_loader:
                    x ,init_2d= x.to(self.net.device),init_2d.to(self.net.device)
                    y = self.net(x)+init_2d
                    loss2 = self.loss_criterion.criterion_2d_3d_pow2_dim2(x, y)
                    loss3 = self.loss_criterion.criterion_fairness_pow_conv_1x3(y)
                    loss4 = 0
                    loss_opt=1e6*loss2+loss3
                    loss_all = [loss_opt.item(), loss2.item(), loss3.item(), loss4]
                    self.stats_manager.accumulate(loss_all)
            else:
                for x in self.val_loader:
                    x = x.to(self.net.device)
                    y = self.net(x)
                    # loss1 = self.loss_criterion.criterion_mae(y, x)
                    loss2 = self.loss_criterion.criterion(y, x)
                    loss3 = self.loss_criterion.criterion_fairness_pow_conv_1x3(y)
                    loss4 = self.loss_criterion.criterion_gauss_curvature_2pi_triangel_area(y)
                    if self.task == "Denoise":
                        loss_opt = loss2 + 10*loss3
                    elif self.task == "Developable":
                        loss_opt = weight_2 * loss2 + weight_3 * loss3 + weight_4 * loss4
                    loss_all = [loss_opt.item(), loss2.item(), loss3.item(), loss4.item()]
                    self.stats_manager.accumulate(loss_all)
        self.net.train()
        return self.stats_manager.summarize()
