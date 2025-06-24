
import argparse
import copy

from dataset.generate_data import data_init, cross_data_init
from dataset.data_utils import compress_client_loaders
import torch

from algs import federaser, my_forget, fl_base, ExactFun, Infocom22, EraseClient, FedAU
from utils import *
import random
import numpy as np
from models.Model_base import Lora

def get_args():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    # TODO
    parser.add_argument('--model', type=str, required=False, default='TextCNN', help= 'choose a model: LeNet_FashionMNIST,CNN_Cifar10,CNN_Cifar100,TextCNN')
    parser.add_argument('--data_name', type=str, required=False, default='text', help= 'choose: mnist, fashionmnist, purchase, adult, cifar10, text, cifar100, cifar10')
    parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
    parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
    parser.add_argument('--distribution', default=True, type=bool, help='True means iid, while False means non-iid')
    parser.add_argument('--train_with_test', default=True, type=bool, help='')
    parser.add_argument('--temperature', default=0.5, type=float, help='the temperature for distillation loss')
    parser.add_argument('--max_checkpoints', default=3, type=int)

    parser.add_argument('--compression_method', type=str, default=None, 
                        help='choose compression method: clustering, block_mixing, random_mixing, gaussian_noise')
    parser.add_argument('--compression_ratio', type=float, default=0.1,
                        help='compression ratio for data compression')
    parser.add_argument('--noise_std', type=float, default=0.1,
                        help='standard deviation for gaussian noise compression')


    # TODO
    parser.add_argument('--forget_paradigm', default='client', type=str, help='choose from client or class')
    parser.add_argument('--paradigm', default='fedau', type=str,
                        help='choose the training paradigm:fused, federaser, retrain, infocom22, exactfun, fl, eraseclient, fedau')
    parser.add_argument('--forget_client_idx', type=list, default=[0, 1, 2, 3, 4])
    parser.add_argument('--forget_class_idx', type=list, default=[0])
    parser.add_argument('--if_retrain', default=False, type=bool, help='')
    parser.add_argument('--if_unlearning', default=False, type=bool, help='')
    parser.add_argument('--baizhanting', default=True, type=bool, help='')
    parser.add_argument('--backdoor', default=False, type=bool, help='')
    parser.add_argument('--backdoor_frac', default=0.2, type=float, help='')

    # TODO
    parser.add_argument('--MIT', default=False, type=bool, help='whether to use membership inference attack')
    parser.add_argument('--n_shadow', default=1, type=int, help='the number of shadow model')
    parser.add_argument('--cut_sample', default=1.0, type=float, help='using part of the training data')
    parser.add_argument('--relearn', default=False, type=bool, help='whether to relearn the unlearned knowledge')
    parser.add_argument('--save_normal_result', default=True, type=bool, help='whether to save the normal result')

    parser.add_argument('--local_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int)


    # TODO
    parser.add_argument('--global_epoch', default=100, type=int)
    parser.add_argument('--local_epoch', default=1, type=int)
    parser.add_argument('--distill_epoch', default=10, type=int)
    parser.add_argument('--distill_pretrain_epoch', default=2, type=int)
    parser.add_argument('--fraction', default=1.0, type=float, help='the fraction of training data')
    parser.add_argument('--num_user', default=50, type=int)

    parser.add_argument('--niid', default=True, type=bool, help='')
    parser.add_argument('--balance', default=True, type=bool, help='')
    parser.add_argument('--partition', default='dir', type=str, help='choose from pat or dir')
    parser.add_argument('--alpha', default=1.0, type=float, help='for Dirichlet distribution')
    parser.add_argument('--proxy_frac', default=0.2, type=float, help='the fraction of training data')
    parser.add_argument('--seed', default=50, type=int)

    parser.add_argument('--unlearn_interval', default=1, type=int, help='')
    parser.add_argument('--forget_local_epoch_ratio', default=0.2, type=float)


    parser.add_argument('--epoch_unlearn', default=20, type=int, help='')
    parser.add_argument('--num_iterations', default=50, type=int, help='')


    parser.add_argument('--dp', action='store_true', default=False, help='whether dp')
    parser.add_argument('--sigma',  type=float, default= 0.1 , help='the sgd of Gaussian noise')
    parser.add_argument('--ul_client_gamma', type=float, default=0.5, help='ul_client_gamma')
    parser.add_argument('--ul_samples_alpha', type=float, default=0.9, help='ul_samples_alpha')


    args = parser.parse_args()
    return args
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', args.device)

    model = model_init(args)


    client_all_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders = data_init(args)
    print(test_loaders[0])
    # client_all_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders = cross_data_init(args)

    if args.paradigm == 'fused':
        args.if_unlearning = False
        case = my_forget.FUSED(args)

        if args.forget_paradigm == 'client':
            if args.compression_method != 'None' and args.compression_method is not None:
                model, all_client_models = case.train_auxiliary(model, copy.deepcopy(client_all_loaders), copy.deepcopy(test_loaders))
                compressed_client_loaders = compress_client_loaders(client_all_loaders, args, model)
            else:
                compressed_client_loaders = client_all_loaders
            client_all_loaders_process, test_loaders_process = baizhanting_attack(args, copy.deepcopy(client_all_loaders),
                                                                                  copy.deepcopy(test_loaders))
            proxy_client_loaders_process, proxy_test_loaders_process = baizhanting_attack(args, copy.deepcopy(proxy_client_loaders), copy.deepcopy(proxy_test_loaders))
            
            # client_all_loaders_process, test_loaders_process = client_all_loaders, test_loaders
            model, all_client_models = case.train_normal(model, client_all_loaders_process, test_loaders_process)
            
            args.if_unlearning = True
            unlearning_model = case.forget_client_train(copy.deepcopy(model), compressed_client_loaders, test_loaders_process)
            if args.MIT:
                args.save_normal_result = False
                membership_inference_attack(args, unlearning_model, case, copy.deepcopy(model), client_all_loaders_process,
                                            test_loaders, proxy_client_loaders_process, proxy_client_loaders,
                                            proxy_test_loaders_process)
                args.save_normal_result = True
            if args.relearn:
                case.relearn_unlearning_knowledge(unlearning_model, client_all_loaders_process, test_loaders_process)
        elif args.forget_paradigm == 'class':
            
            client_all_loaders_bk = copy.deepcopy(client_all_loaders)
            proxy_client_loaders_bk = copy.deepcopy(proxy_client_loaders)
            model, all_client_models = case.train_normal(model, copy.deepcopy(client_all_loaders), test_loaders)
            if args.compression_method != 'None' and args.compression_method is not None:
                compressed_client_loaders = compress_client_loaders(client_all_loaders, args, model)
            else:
                compressed_client_loaders = client_all_loaders
            print(args.forget_class_idx)
            args.if_unlearning = True
            for user in range(args.num_user):
                train_ls = []
                proxy_train_ls = []
                if args.data_name == 'text':
                    for batch in compressed_client_loaders[user]:
                        data = batch[0]
                        at = batch[1]
                        targets = batch[2]
                        for idx, label in enumerate(targets):
                            if label in args.forget_class_idx:
                                label_ls = [i for i in range(args.num_classes)]
                                label_ls.remove(label)
                                inverse_label = np.random.choice(label_ls)
                                label = inverse_label
                            train_ls.append((torch.tensor(data[idx]), torch.tensor(at[idx]), torch.tensor(label)))
                    for batch in proxy_client_loaders[user]:
                        data = batch[0]
                        at = batch[1]
                        targets = batch[2]
                        for idx, label in enumerate(targets):
                            if label in args.forget_class_idx:
                                label_ls = [i for i in range(args.num_classes)]
                                label_ls.remove(label)
                                inverse_label = np.random.choice(label_ls)
                                label = inverse_label
                            proxy_train_ls.append((torch.tensor(data[idx]), torch.tensor(at[idx]), torch.tensor(label)))
                else:
                    for data, target in compressed_client_loaders[user]:
                        data = data.tolist()
                        targets = target.tolist()
                        for idx, label in enumerate(targets):
                            if label in args.forget_class_idx:
                                label_ls = [i for i in range(args.num_classes)]
                                label_ls.remove(label)
                                inverse_label = np.random.choice(label_ls)
                                label = inverse_label
                            train_ls.append((torch.tensor(data[idx]), torch.tensor(label)))
                    for data, target in proxy_client_loaders[user]:
                        data = data.tolist()
                        targets = target.tolist()
                        for idx, label in enumerate(targets):
                            if label in args.forget_class_idx:
                                label_ls = [i for i in range(args.num_classes)]
                                label_ls.remove(label)
                                inverse_label = np.random.choice(label_ls)
                                label = inverse_label
                            proxy_train_ls.append((torch.tensor(data[idx]), torch.tensor(label)))
                train_loader = DataLoader(train_ls, batch_size=args.test_batch_size, shuffle=True)
                proxy_train_loader = DataLoader(proxy_train_ls, batch_size=args.test_batch_size, shuffle=True)
                compressed_client_loaders[user] = train_loader
                proxy_client_loaders[user] = proxy_train_loader

            # client_all_loaders_process = erase_forget_class(args, client_all_loaders)
            unlearning_model = case.forget_class(copy.deepcopy(model), compressed_client_loaders, test_loaders)

            if args.MIT:
                args.save_normal_result = False
                membership_inference_attack(args, unlearning_model, case, copy.deepcopy(model), copy.deepcopy(client_all_loaders_bk),
                                            test_loaders, proxy_client_loaders_bk, proxy_client_loaders,
                                            proxy_test_loaders)
                args.save_normal_result = True
            if args.relearn:
                case.relearn_unlearning_knowledge(unlearning_model, client_all_loaders_bk, test_loaders)

        elif args.forget_paradigm == 'sample':
            if args.compression_method != 'None' and args.compression_method is not None:
                model, all_client_models = case.train_auxiliary(model, copy.deepcopy(client_all_loaders), copy.deepcopy(test_loaders))
                compressed_client_loaders = compress_client_loaders(client_all_loaders, args, model)
            else:
                compressed_client_loaders = client_all_loaders
            client_all_loaders_attack = backdoor_attack(args, copy.deepcopy(client_all_loaders))
            proxy_client_loaders_attack = backdoor_attack(args, copy.deepcopy(proxy_client_loaders))
            
            client_all_loaders_process = erase_backdoor(args, copy.deepcopy(client_all_loaders))
            proxy_client_loaders_process = erase_backdoor(args, copy.deepcopy(proxy_client_loaders))
            if args.compression_method != 'None' and args.compression_method is not None:
                compressed_client_loaders = compress_client_loaders(client_all_loaders_process, args, model)
            else:
                compressed_client_loaders = client_all_loaders_process
            model, all_client_models = case.train_normal(model, client_all_loaders_attack, test_loaders)

            args.if_unlearning = True
            unlearning_model = case.forget_sample(copy.deepcopy(model), compressed_client_loaders, test_loaders)

            if args.MIT:
                args.save_normal_result = False
                membership_inference_attack(args, unlearning_model, case, copy.deepcopy(model), client_all_loaders_attack,
                                            test_loaders, proxy_client_loaders_attack, proxy_client_loaders_process,
                                            proxy_test_loaders)
                args.save_normal_result = True
            if args.relearn:
                case.relearn_unlearning_knowledge(unlearning_model, client_all_loaders_attack, test_loaders)