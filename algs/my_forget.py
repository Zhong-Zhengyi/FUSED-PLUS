import time
import math
import pandas as pd
import torch

from models.Model_base import *
from models import LeNet_FashionMNIST, CNN_Cifar10, CNN_Cifar100, Model_adults, Model_purchase
from utils import init_network, test_class_forget, test_client_forget
from dataset.data_utils import *
from algs.fl_base import Base
import torch.optim as optim
import copy
import logging
# import objgraph
import matplotlib.pyplot as plt
from utils import *
import random
from models.Model_base import *

class FUSED(Base):
    def __init__(self, args):
        super(FUSED, self).__init__(args)
        self.args = args
        self.log_dir = f"logs/moe_{self.args.data_name}_{self.args.alpha}"
        self.param_change_dict = {}
        self.param_size = {}

    def train_normal(self, global_model, client_all_loaders, test_loaders):
        print('\n')
        print(5 * "#" + "  FUSED Federated Training Start  " + 5 * "#")

        checkpoints_ls = []
        result_list = []
        param_list = []
        for name, param in global_model.named_parameters():
            # print(name)
            self.param_change_dict[name] = 0
            self.param_size[name] = 0

        for epoch in range(self.args.global_epoch):
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user*self.args.fraction), replace=False))
            select_client_loaders = [client_all_loaders[idx] for idx in selected_clients]
            client_models = self.global_train_once(epoch, global_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)

            global_model = self.fedavg(client_models)

            all_idx = list(range(self.args.num_user))

            client_test_acc = []

            if self.args.forget_paradigm == 'sample':
            
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model,
                                                                                              self.args, test_loaders)
                print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero,
                                                                                 avg_test_acc))
                result_list.extend(test_result_ls)
            
            elif self.args.forget_paradigm == 'client':
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, global_model, self.args, test_loaders)
                print('Epoch={}, avg_f_acc={}, avg_r_acc={}'.format(epoch, avg_f_acc, avg_r_acc))
                result_list.extend(test_result_ls)
            
            elif self.args.forget_paradigm == 'class':
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args, test_loaders)
                print('Epoch={}, avg_f_acc={}, avg_r_acc={}'.format(epoch, avg_f_acc, avg_r_acc))
                result_list.extend(test_result_ls)

            if self.args.paradigm == 'fused':
                diff_ls = list(self.param_change_dict.values())
                name = list(self.param_change_dict.keys())
                diff_ls_ = [float(i) for i in diff_ls]
                param_list.append(diff_ls_)
                # diff_ls_.append(list(self.param_size.values()))
        df = pd.DataFrame(param_list, columns=name)
        df.to_csv('./results/param_change_{}_distri_{}.csv'.format(self.args.data_name, self.args.alpha))

        torch.save(global_model.state_dict(), 'save_model/global_model_{}.pth'.format(self.args.data_name))

        if self.args.forget_paradigm == 'sample':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc'])
        elif self.args.forget_paradigm == 'client':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss'])
        elif self.args.forget_paradigm == 'class':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss'])
        if self.args.save_normal_result:
            df.to_csv('./results/Acc_loss_fl_{}_data_{}_distri_{}.csv'.format(self.args.forget_paradigm, self.args.data_name, self.args.alpha))

        return global_model, client_models

    def forget_client_train(self, global_model, client_all_loaders, test_loaders):
        global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(self.args.data_name)))
        avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, 1, global_model, self.args,
                                                                  test_loaders)
        print('FUSED-epoch-{}-client forget, Avg_r_acc: {}, Avg_f_acc: {}'.format('xxxx', avg_r_acc,
                                                                                 avg_f_acc))
        if self.args.data_name == 'text':
            fused_model = Loratext(self.args, global_model)
        else:
            fused_model = Lora(self.args, global_model)
        torch.save(fused_model.state_dict(), 'save_model/global_loramodel_{}.pth'.format(self.args.data_name))
        print('\n')
        print(5 * "#" + "  FUSED Federated Client Unlearning Start  " + 5 * "#")

        checkpoints_ls = []
        result_list = []
        consume_time = 0

        for epoch in range(self.args.global_epoch):
            fused_model.train()
            selected_clients = [i for i in range(self.args.num_user) if i not in self.args.forget_client_idx]
            select_client_loaders = select_part_sample(self.args, client_all_loaders, selected_clients)

            std_time = time.time()
            client_models = self.global_train_once(epoch, fused_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            avg_model = self.fedavg(client_models)
            consume_time += end_time - std_time
            fused_model.load_state_dict(avg_model.state_dict())

            fused_model.eval()

            avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, fused_model, self.args,
                                                                      test_loaders)

            result_list.extend(test_result_ls)

            print('FUSED-epoch-{}-client forget, Avg_r_acc: {}, Avg_f_acc: {}'.format(epoch, avg_r_acc,
                                                                                    avg_f_acc))


        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss'])
        df['Comsume_time'] = consume_time

        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv('./results/{}/Acc_loss_lora_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm,
                                                                                              self.args.forget_paradigm,
                                                                                              self.args.data_name,
                                                                                              self.args.alpha,
                                                                                              len(self.args.forget_client_idx)))
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/Acc_loss_lora_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'.format(
                        self.args.forget_paradigm,
                        self.args.forget_paradigm,
                        self.args.data_name,
                        self.args.alpha,
                        len(self.args.forget_client_idx), self.args.cut_sample))

        print(5 * "#" + "  FUSED Federated Client Unlearning End  " + 5 * "#")

        return fused_model

    def forget_class(self, global_model, client_all_loaders, test_loaders):
        print('\n')
        print(5 * "#" + "  FUSED Federated Class Unlearning Start  " + 5 * "#")
        num_selected_clients = self.args.num_user * self.args.forget_client_idx
        checkpoints_ls = []
        result_list = []
        consume_time = 0
        if self.args.data_name == 'text':
            fused_model = Loratext(self.args, global_model)
        else:
            fused_model = Lora(self.args, global_model)
        for epoch in range(self.args.global_epoch):
            fused_model.train()
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user * self.args.fraction), replace=False))

            select_client_loaders = select_part_sample(self.args, client_all_loaders, selected_clients)
            std_time = time.time()

            client_models = self.global_train_once(epoch, fused_model,  select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            fused_model = self.fedavg(client_models)
            consume_time += end_time-std_time

            avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, fused_model, self.args, test_loaders)
            result_list.extend(test_result_ls)
            print('Epoch={}, Remember Test Acc={}, Forget Test Acc={}'.format(epoch, avg_r_acc, avg_f_acc))

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss'])
        df['Comsume_time'] = consume_time

        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                './results/{}/Acc_loss_lora_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm, self.args.forget_paradigm, self.args.data_name, self.args.alpha, len(self.args.forget_class_idx)))
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/Acc_loss_lora_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'.format(self.args.forget_paradigm,
                                                                                     self.args.forget_paradigm,
                                                                                     self.args.data_name,
                                                                                     self.args.alpha,
                                                                                     len(self.args.forget_class_idx), self.args.cut_sample))


        print(5 * "#" + "  FUSED Federated Class Unlearning End  " + 5 * "#")
        return fused_model

    def forget_sample(self, global_model, client_all_loaders, test_loaders):
        print('\n')
        print(5 * "#" + "  FUSED Federated Sample Unlearning Start  " + 5 * "#")

        checkpoints_ls = []
        result_list = []
        consume_time = 0
        if self.args.data_name == 'text':
            fused_model = Loratext(self.args, global_model)
        else:
            fused_model = Lora(self.args, global_model)
        for epoch in range(self.args.global_epoch):
            fused_model.train()
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user * self.args.fraction), replace=False))# 将需要遗忘的客户端排除在外

            self.select_forget_idx = list()
            select_client_loaders = list()
            record = -1
            for idx in selected_clients:
                select_client_loaders.append(client_all_loaders[idx])
                record += 1
                if idx in self.args.forget_client_idx:
                    self.select_forget_idx.append(record)
            std_time = time.time()
            client_models = self.global_train_once(epoch, fused_model,  select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            fused_model = self.fedavg(client_models)
            consume_time += end_time-std_time

            avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, fused_model, self.args, test_loaders)
            result_list.extend(test_result_ls)
            print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero, avg_test_acc))

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc'])
        df['Comsume_time'] = consume_time
        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                './results/{}/Acc_loss_lora_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm, self.args.forget_paradigm, self.args.data_name, self.args.alpha, self.args.backdoor_frac))
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/Acc_loss_lora_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'.format(self.args.forget_paradigm,
                                                                                        self.args.forget_paradigm,
                                                                                        self.args.data_name,
                                                                                        self.args.alpha,
                                                                                        self.args.backdoor_frac, self.args.cut_sample))

        print(5 * "#" + "  FUSED Federated Sample Unlearning End  " + 5 * "#")
        return fused_model

    # def pretrain_before_distill(self, proxy_loader, model):
    #     model.to(self.args.device)
    #     optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #     criteria = nn.CrossEntropyLoss()
    #     for epoch in range(self.args.distill_pretrain_epoch):
    #         for batch_idx, (data, target) in enumerate(proxy_loader):
    #             model.zero_grad()
    #             data = data.to(self.args.device)
    #             target = target.to(self.args.device)
    #             optimizer.zero_grad()
    #             pred, _ = model(data)
    #             loss = criteria(pred, target)
    #             loss.backward()
    #             optimizer.step()
    #         scheduler.step()

    def distill(self, proxy_data, teacher_model, student_model, test_loaders):
        student_model.to(self.args.device)
        optimizer = optim.SGD(student_model.gate_model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        all_idx = [idx for idx in range(self.args.num_user)]

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        avg_acc = 0
        result_list = []
        std_time = time.time()
        for server_epoch in range(self.args.distill_epoch):
            last_avg_acc = avg_acc
            for batch_idx, (data, target) in enumerate(proxy_data):
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                loss_all = []
                loss_all_f = []
                loss_all_r = []
                z_r_ls = []
                z_f_ls = []
                weights_f = []
                weights_r = []
                student_outputs, j = student_model(data)
                hard_loss = nn.CrossEntropyLoss()(student_outputs, target)
                for k, teacher in enumerate(teacher_model):
                    teacher.to(self.args.device)
                    teacher_outputs, i = teacher(data)

                    # if self.args.if_unlearning == True:
                    #     if k in self.args.forget_client_idx:
                    #         loss_all_f.append(-distillation_loss)
                    #     else:
                    #         loss_all_r.append(hard_loss * 0.3 -distillation_loss)
                    # else:
                    #     loss_all.append(hard_loss * 0.3 + distillation_loss * 0.7)

                    if self.args.if_unlearning == True:
                        # if k in self.args.forget_client_idx:
                        #     z_f_ls.append(teacher_outputs)
                        #     weights_f.append(self.args.datasize_ls[k]/sum(self.args.datasize_ls))
                        # else:
                        #     z_r_ls.append(teacher_outputs)
                        #     weights_r.append(self.args.datasize_ls[k] / sum(self.args.datasize_ls))
                        if k not in self.args.forget_client_idx:
                            distillation_loss = nn.KLDivLoss(reduction="batchmean")(
                                nn.functional.log_softmax(student_outputs / self.args.temperature, dim=1),
                                nn.functional.softmax(teacher_outputs / self.args.temperature, dim=1))
                            # distillation_loss = nn.CrossEntropyLoss(student_outputs, teacher_outputs)
                            loss_all.append(hard_loss * 0.5 + distillation_loss * 0.5)
                        # elif k in self.args.forget_client_idx:
                        #     distillation_loss = nn.KLDivLoss(reduction="batchmean")(
                        #         nn.functional.log_softmax(student_outputs / self.args.temperature, dim=1),
                        #         nn.functional.softmax(teacher_outputs / self.args.temperature, dim=1))
                        #     # distillation_loss = nn.CrossEntropyLoss(student_outputs, teacher_outputs)
                        #     loss_all.append(hard_loss * 0.3 - distillation_loss * 0.7)
                    else:
                        distillation_loss = nn.KLDivLoss(reduction="batchmean")(
                            nn.functional.log_softmax(student_outputs / self.args.temperature, dim=1),
                            nn.functional.softmax(teacher_outputs / self.args.temperature, dim=1))
                        loss_all.append(hard_loss * 0.3 + distillation_loss * 0.7)

                # if self.args.if_unlearning == True:
                #     optimizer.zero_grad()
                #     student_outputs = student_outputs.detach().cpu().float()
                #     z_f_ls = [t.detach().cpu().float()*weights_f[idx] for idx, t in enumerate(z_f_ls)]
                #     z_f_ls = torch.stack(z_f_ls)
                #     # weights_f = torch.tensor(weights_f, dtype=torch.float32)
                #     z_f_avg = torch.sum(z_f_ls, dim=0)
                #
                #     z_r_ls = [t.detach().cpu().float()*weights_r[idx] for idx, t in enumerate(z_r_ls)]
                #     z_r_ls = torch.stack(z_r_ls)
                #     # weights_r = torch.tensor(weights_r, dtype=torch.float32)
                #     z_r_avg = torch.sum(z_r_ls, dim=0)
                #     # z_r_avg = torch.mean(z_r_ls, dim=0)
                #
                #     distillation_loss = self.calculate_expression(student_outputs, z_r_avg, z_f_avg)
                #     distillation_loss = torch.tensor(distillation_loss, requires_grad=True)
                #     # if self.args.data_name == 'purchase':
                #     #     distillation_loss = torch.tensor(distillation_loss, requires_grad=True)
                #     # else:
                #     #     distillation_loss = distillation_loss.clone().detach().requires_grad_(True)
                #     # distillation_loss = torch.tensor(distillation_loss, requires_grad=True)
                #     distillation_loss = distillation_loss.to(self.args.device)
                #     # if server_epoch == 0:
                #     print('server_epoch-loss', server_epoch, distillation_loss)
                #     loss_avg = distillation_loss
                #     # loss_avg_f = sum(loss_all_f)/len(loss_all_f)
                #     # loss_avg_r = sum(loss_all_r)/len(loss_all_r)
                #     # loss_avg = loss_avg_r + loss_avg_f
                #     loss_avg.backward()
                #     optimizer.step()

                # else:
                optimizer.zero_grad()
                loss_avg = sum(loss_all)/len(loss_all)
                loss_avg.backward()
                optimizer.step()
            scheduler.step()
            end_time = time.time()
            consume_time = end_time-std_time

            avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, server_epoch, student_model, self.args,
                                                                      test_loaders)
            result_list.extend(test_result_ls)
            if self.args.if_unlearning == True:
                print('MoE-epoch-{}-client forget, Avg_r_acc: {}, Avg_f_acc: {}'.format(server_epoch, avg_r_acc, avg_f_acc))
            else:
                avg_acc = np.array([row[2] for row in test_result_ls])
                avg_acc = np.mean(avg_acc)
                print('Distill Acc: {}'.format(avg_acc))

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss'])
        df['Comsume_time'] = consume_time

        df.to_csv(
                './results/Acc_loss_normalmoe_after_distill_{}_distri_{}.csv'.format(self.args.data_name, self.args.alpha))

        return student_model, avg_acc, df

    def cos_similarity(self, vec1, vec2):
        batch_size = vec1.shape[0]
        sum = 0
        for k in range(batch_size):
            dot_product = np.dot(vec1[k], vec2[k])
            norm_vec1 = np.linalg.norm(vec1[k])
            norm_vec2 = np.linalg.norm(vec2[k])
            similarity = dot_product / (norm_vec1 * norm_vec2)
            sum += similarity
        similarity = sum / batch_size
        return similarity

    def calculate_expression(self, z, z_r_avg, z_f_avg, T=0.5):
        z_r_avg.detach().cpu().float()
        if self.args.data_name == 'purchase':
            similarity1 = self.cos_similarity(z, z_r_avg)
            similarity2 = self.cos_similarity(z, z_f_avg)
        else:
            batch_size = z.shape[0]
            similarity1 = 0
            similarity2 = 0
            for k in range(batch_size):
                prob = F.softmax(z[k], dim=0)
                prob_r = F.softmax(z_r_avg[k], dim=0)

                # sim_z_r = self.cos_similarity(z, z_r_avg)
                # term1 = np.exp(sim_z_r / T)
                term1 = F.kl_div(prob.log(), prob_r, reduction='batchmean')

                # sim_z_f = self.cos_similarity(z, z_f_avg)
                # term2 = np.exp(sim_z_f / T)
                z_f_avg.detach().cpu().float()
                prob_f = F.softmax(z_f_avg[k], dim=0)
                term2 = F.kl_div(prob.log(), prob_f, reduction='batchmean')
                similarity1 += term1
                similarity2 += term2
            similarity1 = similarity1 / batch_size
            similarity2 = similarity2 / batch_size


        # loss = -np.log(similarity1/(similarity1 + similarity2))
        loss = similarity1-0.5*similarity2
        # loss = -similarity2

        return loss

    def calculate_kl(self, z_s, z_t, T=0.5):
        batch_size = z_s.shape[0]
        similarity1 = 0
        for k in range(batch_size):
            prob_s = F.softmax(z_s[k], dim=0)
            prob_t = F.softmax(z_t[k], dim=0)
            term1 = F.kl_div(prob_s.log(), prob_t, reduction='batchmean')
            similarity1 += term1
        loss = similarity1/batch_size

        return loss

    def relearn_unlearning_knowledge(self, unlearning_model, client_all_loaders, test_loaders):
        checkpoints_ls = []
        all_global_models = list()
        all_client_models = list()
        global_model = unlearning_model
        result_list = []

        all_global_models.append(global_model)
        std_time = time.time()
        for epoch in range(self.args.global_epoch):
            if self.args.forget_paradigm == 'client':
                select_client_loaders = list()
                for idx in self.args.forget_client_idx:
                    select_client_loaders.append(client_all_loaders[idx])
            elif self.args.forget_paradigm == 'class':
                select_client_loaders = list()
                client_loaders = select_forget_class(self.args, copy.deepcopy(client_all_loaders))
                for v in client_loaders:
                    if v is not None:
                        select_client_loaders.append(v)
            elif self.args.forget_paradigm == 'sample':
                select_client_loaders = list()
                client_loaders = select_forget_sample(self.args, copy.deepcopy(client_all_loaders))
                for v in client_loaders:
                    if v is not None:
                        select_client_loaders.append(v)
            client_models = self.global_train_once(epoch, global_model, select_client_loaders, test_loaders, self.args,
                                                   checkpoints_ls)

            all_client_models += client_models
            global_model = self.fedavg(client_models)
            all_global_models.append(copy.deepcopy(global_model).to('cpu'))
            end_time = time.time()

            consume_time = end_time - std_time

            if self.args.forget_paradigm == 'client':
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, global_model, self.args,
                                                                          test_loaders)
                for item in test_result_ls:
                    item.append(consume_time)
                result_list.extend(test_result_ls)
                df = pd.DataFrame(result_list,
                                  columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss',
                                           'Comsume_time'])
            elif self.args.forget_paradigm == 'class':
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args,
                                                                         test_loaders)
                for item in test_result_ls:
                    item.append(consume_time)
                result_list.extend(test_result_ls)
                df = pd.DataFrame(result_list,
                                  columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss', 'Comsume_time'])
            elif self.args.forget_paradigm == 'sample':
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model, self.args, test_loaders)
                for item in test_result_ls:
                    item.append(consume_time)
                result_list.extend(test_result_ls)
                df = pd.DataFrame(result_list,
                                  columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc', 'Comsume_time'])

            global_model.to('cpu')

            print("Relearn Round = {}".format(epoch))
        
        if self.args.cut_sample == 1.0:
            df.to_csv('./results/{}/relearn_data_{}_distri_{}_fnum_{}_algo_{}.csv'.format(self.args.forget_paradigm,
                                                                                          self.args.data_name,
                                                                                      self.args.alpha,
                                                                                      len(self.args.forget_class_idx),
                                                                                      self.args.paradigm), index=False)
        elif self.args.cut_sample < 1.0:
            df.to_csv('./results/{}/relearn_data_{}_distri_{}_fnum_{}_algo_{}_partdata_{}.csv'.format(self.args.forget_paradigm,
                                                                                          self.args.data_name,
                                                                                      self.args.alpha,
                                                                                      len(self.args.forget_class_idx),
                                                                                      self.args.paradigm, self.args.cut_sample), index=False)
        return

    def train_auxiliary(self, global_model, client_all_loaders, test_loaders):
        print('\n')
        print(5 * "#" + "  Auxiliary Training Start  " + 5 * "#")

        checkpoints_ls = []

        for epoch in range(self.args.global_epoch):
            print('Auxiliary Training epoch {}/{}'.format(epoch + 1, self.args.global_epoch))
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user*self.args.fraction), replace=False))
            select_client_loaders = [client_all_loaders[idx] for idx in selected_clients]
            client_models = self.global_train_once(epoch, global_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)
            global_model = self.fedavg(client_models)

        print(5 * "#" + "  Auxiliary Training End  " + 5 * "#")
        return global_model, client_models