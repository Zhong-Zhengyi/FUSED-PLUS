import random

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle
import copy
from sklearn.cluster import KMeans
from transformers import BertTokenizer

train_size = 0.99 # merge original training set and test set, then split it manually.
least_samples = 100 # guarantee that each client must have at least one samples for testing.
# alpha = 0.8 # for Dirichlet distribution


def data_set(data_name):
    # if not data_name in ['mnist', 'purchase', 'adult', 'cifar10']:
    #     raise TypeError('data_name should be a string, including mnist,purchase,adult,cifar10. ')

    # model: 2 conv. layers followed by 2 FC layers
    if (data_name == 'mnist'):
        trainset = datasets.MNIST('./dataset/mnist', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))

        testset = datasets.MNIST('./dataset/mnist', train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
    elif (data_name == 'fashionmnist'):
        trainset = datasets.MNIST('./dataset/fashionmnist', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))

        testset = datasets.MNIST('./dataset/fashionmnist', train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
    # model: ResNet-18
    elif (data_name == 'cifar10'):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

        trainset = datasets.CIFAR10(root='./dataset/cifar10', train=True,
                                    download=True, transform=transform)

        testset = datasets.CIFAR10(root='./dataset/cifar10', train=False,
                                   download=True, transform=transform)

    elif (data_name == 'cifar100'):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR100('./dataset/cifar100', train=True, download=True,
                                  transform=transform)

        testset = datasets.CIFAR100('./dataset/cifar100', train=False, download=True,
                                 transform=transform)

    # model: 2 FC layers
    elif (data_name == 'purchase'):
        labels = []
        features = []
        with open('./dataset/purchase/dataset_purchase', 'r') as f:
            n=0
            for line in f:
                n+=1
                line = line.strip()
                splitted_line = line.split(",")
                labels.append(int(splitted_line[0]) - 1)
                features.append(list(map(int, splitted_line[1:])))
        label_set = set(labels)
        num_classes = len(list(label_set))
        one_hot_label = torch.eye(num_classes)[labels]
        yy = np.array(one_hot_label)
        xx = np.array(features)
        # xx = np.load("./dataset/purchase/purchase_xx.npy")
        # yy = np.load("./dataset/purchase/purchase_y2.npy")
        # yy = yy.reshape(-1,1)
        # enc = preprocessing.OneHotEncoder(categories='auto')
        # enc.fit(yy)
        # yy = enc.transform(yy).toarray()
        X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=42)

        X_train_tensor = torch.Tensor(X_train).type(torch.FloatTensor)
        X_test_tensor = torch.Tensor(X_test).type(torch.FloatTensor)
        y_train_tensor = torch.Tensor(y_train).type(torch.LongTensor)
        y_test_tensor = torch.Tensor(y_test).type(torch.LongTensor)

        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        testset = TensorDataset(X_test_tensor, y_test_tensor)


    # model: 2 FC layers
    elif (data_name == 'adult'):
        # load data
        file_path = "./dataset/adult/"
        data1 = pd.read_csv(file_path + 'adult.data', header=None)
        data2 = pd.read_csv(file_path + 'adult.test', header=None)
        data2 = data2.replace(' <=50K.', ' <=50K')
        data2 = data2.replace(' >50K.', ' >50K')
        train_num = data1.shape[0]
        data = pd.concat([data1, data2])

        # data transform: str->int
        data = np.array(data, dtype=str)
        labels = data[:, 14]
        le = LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        data = data[:, :-1]

        categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]
        # categorical_names = {}
        for feature in categorical_features:
            le = LabelEncoder()
            le.fit(data[:, feature])
            data[:, feature] = le.transform(data[:, feature])
            # categorical_names[feature] = le.classes_
        data = data.astype(float)

        n_features = data.shape[1]
        numerical_features = list(set(range(n_features)).difference(set(categorical_features)))
        for feature in numerical_features:
            scaler = MinMaxScaler()
            sacled_data = scaler.fit_transform(data[:, feature].reshape(-1, 1))
            data[:, feature] = sacled_data.reshape(-1)

        # OneHotLabel
        oh_encoder = ColumnTransformer(
            [('oh_enc', OneHotEncoder(sparse=False), categorical_features), ],
            remainder='passthrough')
        oh_data = oh_encoder.fit_transform(data)

        xx = oh_data
        yy = labels

        xx = preprocessing.scale(xx)
        yy = np.array(yy)

        xx = torch.Tensor(xx).type(torch.FloatTensor)
        yy = torch.Tensor(yy).type(torch.LongTensor)
        xx_train = xx[0:data1.shape[0], :]
        xx_test = xx[data1.shape[0]:, :]
        yy_train = yy[0:data1.shape[0]]
        yy_test = yy[data1.shape[0]:]

        # trainset = Array2Dataset(xx_train, yy_train)
        # testset = Array2Dataset(xx_test, yy_test)
        trainset = TensorDataset(xx_train, yy_train)
        testset = TensorDataset(xx_test, yy_test)

    return trainset, testset

def compress_data_for_lora(X, y, args, model, compression_method='clustering', AT=None):
    """
    1. clustering
    2. block_mixing
    3. random_mixing
    4. gaussian_noise
    """
    if compression_method == 'None' or compression_method is None:
        if args.data_name == 'text':
            return np.array(X), np.array(y), np.array(AT)
        else:
            return np.array(X), np.array(y)
    compressed_X = []
    compressed_y = []
    compressed_AT = [] if args.data_name == 'text' else None

    unique_classes = np.unique(y)

    is_text = args.data_name == 'text'
    is_image_dataset = args.data_name in ['mnist', 'fashionmnist', 'cifar10', 'cifar100']
    
    def split_image_into_blocks(image): #将图像分割成4x4块
        h, w = image.shape[1:3]
        block_h, block_w = h // 4, w // 4
        blocks = []
        for i in range(4):
            for j in range(4):
                block = image[:, i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                blocks.append(block)
        return blocks, block_h, block_w
    
    def merge_blocks_into_image(blocks, block_h, block_w): #将4x4块合并成完整图像
        image = np.zeros((blocks[0].shape[0], block_h*4, block_w*4))
        for i in range(4):
            for j in range(4):
                image[:, i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w] = blocks[i*4 + j]
        return image
    
    def get_model_probability(model, batch, args): 
        model.eval()
        with torch.no_grad():
            if args.data_name == 'text':
                input_ids = batch[0].to(args.device)
                input_ids = input_ids.long()
                attention_mask = batch[1].to(args.device)

                outputs = model(input_ids, attention_mask)
                logits = outputs.logits
            else:
                if isinstance(batch, np.ndarray):
                    batch = torch.from_numpy(batch).float()
                if len(batch.shape) == 3: 
                    batch = batch.unsqueeze(0)
                batch = batch.to(args.device)
                model.to(args.device)

                outputs = model(batch)
                logits = outputs

            prob = torch.nn.functional.softmax(logits, dim=1)
            return prob.cpu().numpy()
    def get_model_accuracy(model, batch, label, args):
        model = model.to(args.device)
        model.eval()
        with torch.no_grad():
            if args.data_name == 'text':
                input_ids = torch.as_tensor(batch[0])
                attention_mask = torch.as_tensor(batch[1])
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                input_ids = input_ids.to(args.device).long()
                attention_mask = attention_mask.to(args.device)
                outputs = model(input_ids, attention_mask)
                logits = outputs.logits
            else:
                if isinstance(batch, np.ndarray):
                    batch = torch.from_numpy(batch).float()
                if len(batch.shape) == 3:
                    batch = batch.unsqueeze(0)
                batch = batch.to(args.device)
                outputs = model(batch)
                logits = outputs

            probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.cpu().numpy()[0][label]
    def compute_kl_divergence(p, q):
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        return np.sum(p * np.log(p / q))
    
    for cls in unique_classes:
        class_indices = np.where(y == cls)[0]
        class_data = X[class_indices]
        class_size = len(class_data)

        print(f"Class {cls}: before compression: {class_size}")
        before_count = len(compressed_X)

        if class_size <= 10:
            compressed_X.extend(class_data)
            compressed_y.extend([cls] * class_size)

        else:
            n_compressed = max(10, int(class_size * args.compression_ratio))         
            if compression_method == 'clustering':
                n_samples = class_data.shape[0]
                n_features = np.prod(class_data.shape[1:])
                reshaped_data = class_data.reshape(n_samples, n_features)

                kmeans = KMeans(n_clusters=n_compressed, random_state=42)
                kmeans.fit(reshaped_data)

                for i in range(n_compressed):
                    cluster_indices = np.where(kmeans.labels_ == i)[0]
                    cluster_data = class_data[cluster_indices]
                    mean_data = np.mean(cluster_data, axis=0)
                    compressed_X.append(mean_data)
                    compressed_y.append(cls)
            
            elif compression_method == 'block_mixing':
                if not is_image_dataset:
                    print(f"Warning: Block mixing compression is only supported for image datasets. Using original data for {args.data_name}.")
                    compressed_X.extend(class_data)
                    compressed_y.extend([cls] * class_size)

                    continue

                kl_divergences = []
                for sample in class_data:
                    original_sample = sample.copy()
                    
                    blocks, block_h, block_w = split_image_into_blocks(original_sample)
                    blocks[0], blocks[10] = blocks[10], blocks[0] 
                    mixed_image = merge_blocks_into_image(blocks, block_h, block_w)

                    if isinstance(original_sample, np.ndarray):
                        sample_for_prob = torch.from_numpy(original_sample).float()
                    else:
                        sample_for_prob = original_sample
                    if len(sample_for_prob.shape) == 3:  
                        sample_for_prob = sample_for_prob.unsqueeze(0)  

                    orig_prob = get_model_probability(model, sample_for_prob, args)
                    mixed_prob = get_model_probability(model, mixed_image, args)
                    kl_div = compute_kl_divergence(orig_prob, mixed_prob)
                    kl_divergences.append(kl_div)

                n_select = max(10, int(class_size * args.compression_ratio))
                selected_indices = np.argsort(kl_divergences)[:n_select]
                selected_data = class_data[selected_indices]
                
                compressed_X.extend(selected_data)
                compressed_y.extend([cls] * n_select)
            
            elif compression_method == 'random_mixing':
                if not is_image_dataset:
                    print(f"Warning: Random mixing compression is only supported for image datasets. Using original data for {args.data_name}.")
                    compressed_X.extend(class_data)
                    compressed_y.extend([cls] * class_size)

                    continue

                kl_divergences = []
                for sample in class_data:
                    original_sample = sample.copy()
                    
                    pixels = original_sample.reshape(original_sample.shape[0], -1)
                    np.random.shuffle(pixels.T)
                    mixed_image = pixels.reshape(sample.shape)
                    if isinstance(original_sample, np.ndarray):
                        sample_for_prob = torch.from_numpy(original_sample).float()
                    else:
                        sample_for_prob = original_sample
                    if len(sample_for_prob.shape) == 3:  
                        sample_for_prob = sample_for_prob.unsqueeze(0)
                    
                    orig_prob = get_model_probability(model, sample_for_prob, args)
                    mixed_prob = get_model_probability(model, mixed_image, args)
                    kl_div = compute_kl_divergence(orig_prob, mixed_prob)
                    kl_divergences.append(kl_div)

                n_select = max(10, int(class_size * args.compression_ratio))
                selected_indices = np.argsort(kl_divergences)[:n_select]
                selected_data = class_data[selected_indices]
                
                compressed_X.extend(selected_data)
                compressed_y.extend([cls] * n_select)
            elif compression_method == 'gaussian_noise':
                if not is_image_dataset:
                    print(f"Warning: Gaussian noise compression is only supported for image datasets. Using original data for {args.data_name}.")
                    compressed_X.extend(class_data)
                    compressed_y.extend([cls] * class_size)
                    continue

                kl_divergences = []
                noise_std = args.noise_std  # 高斯噪声标准差
                
                for sample in class_data:
                    original_sample = sample.copy()
                    noise = np.random.normal(0, noise_std, original_sample.shape)
                    noisy_image = original_sample + noise
                    
                    if args.data_name == 'cifar10' or args.data_name == 'cifar100':
                        noisy_image = np.clip(noisy_image, -2.5, 2.5)  
                    elif args.data_name in ['mnist', 'fashionmnist']:
                        noisy_image = np.clip(noisy_image, -0.5, 0.5)
                    else:
                        noisy_image = np.clip(noisy_image, 0, 1)
                    
                    if isinstance(original_sample, np.ndarray):
                        sample_for_prob = torch.from_numpy(original_sample).float()
                    else:
                        sample_for_prob = original_sample
                    if len(sample_for_prob.shape) == 3:  
                        sample_for_prob = sample_for_prob.unsqueeze(0)  

                    orig_prob = get_model_probability(model, sample_for_prob, args)
                    mixed_prob = get_model_probability(model, noisy_image, args)
                    kl_div = compute_kl_divergence(orig_prob, mixed_prob)
                    kl_divergences.append(kl_div)

                n_select = max(10, int(class_size * args.compression_ratio))
                selected_indices = np.argsort(kl_divergences)[:n_select]
                selected_data = class_data[selected_indices]
                
                compressed_X.extend(selected_data)
                compressed_y.extend([cls] * n_select)
            else:
                raise ValueError(f"Unsupported compression method: {compression_method}")
        after_count = len(compressed_X)
        print(f"Class {cls}: after compression: {after_count - before_count}")

    return np.array(compressed_X), np.array(compressed_y)

def separate_data(data, num_clients, num_classes, args, niid=False, balance=False, partition=None, class_per_client=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]

    statistic = [[] for _ in range(num_clients)]
    if args.data_name == 'text':
        AT = [[] for _ in range(num_clients)]
        dataset_content, at, dataset_label = data
    else:
        dataset_content, dataset_label = data

    dataidx_map = {}
    # if args.paradigm == 'retrain' and args.forget_paradigm == 'class':
    #     classes_ls = [i for i in range(num_classes) if i not in args.forget_class_idx]
    # else:
    classes_ls = [i for i in range(num_classes)]

    if not niid:
        partition = 'pat'
        class_per_client = len(classes_ls)

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for cls in classes_ls:
            idx_for_each_class.append(idxs[dataset_label == cls])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in classes_ls:
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients /len(classes_ls)) *class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients -1)]
            else:
                num_samples = np.random.randint(max(num_per /10, least_samples /len(classes_ls)), num_per, num_selected_clients -1).tolist()
            num_samples.append(int(num_all_samples -sum(num_samples)))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx +num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = len(classes_ls)
        N = len(dataset_label)
        label_ls = []
        if args.data_name == 'purchase':
            for i, item in enumerate(dataset_label):
                ls = [index for index, value in enumerate(item) if value != 0]
                label_ls.append(ls[0])
            dataset_label = np.array(label_ls)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print \
                    (f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(args.alpha, num_clients))
                proportions = np.array([ p *(len(idx_j ) < N /num_clients) for p ,idx_j in zip(proportions ,idx_batch)])
                proportions = proportions /proportions.sum()
                proportions = (np.cumsum(proportions ) *len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j ,idx in zip(idx_batch ,np.split(idx_k ,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        if args.data_name == 'text':
            AT[client] = at[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))#statistic为一个列表，索引为客户端id，元素一个列表，列表记录这客户端每一类数据的数量


    #if args.paradigm == 'lora':
    #   for client in range(num_clients):
    #        if args.data_name == 'text':
    #            X[client], y[client], AT[client] = compress_data_for_lora(X[client], y[client], args, compression_method=args.compression_method, AT=AT[client])
    #        else:
    #            X[client], y[client] = compress_data_for_lora(X[client], y[client], args, compression_method=args.compression_method)
    #    print("Data compressed for LoRA paradigm. Original size: {}, Compressed size: {}".format(
    #        len(dataset_content), sum(len(x) for x in X)))

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)
    if args.data_name == 'text':
        return X, AT, y, statistic
    else:
        return X, y, statistic

def split_test_proxy(test_loader, args):
    test_data_x, test_data_y, proxy_data_x, proxy_data_y = [], [], [], []
    for test_data in test_loader:
        data, label = test_data
    dataset_image = []
    dataset_label = []

    dataset_image.extend(data.cpu().detach().numpy())
    dataset_label.extend(label.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    num_classes = args.num_classes
    idxs = np.array(range(len(dataset_label)))
    idx_for_each_class = []
    for i in range(num_classes):
        idx_for_each_class.append(idxs[dataset_label == i])
        num_class_proxy = len(idx_for_each_class[i])*args.proxy_frac
        idx_class_proxy = np.random.choice(idx_for_each_class[i], int(num_class_proxy))
        idx_class_test = list(set(idx_for_each_class[i])-set(idx_class_proxy))
        proxy_data_x.extend(dataset_image[idx_class_proxy])
        proxy_data_y.extend(dataset_label[idx_class_proxy])
        test_data_x.extend(dataset_image[idx_class_test])
        test_data_y.extend(dataset_label[idx_class_test])
    proxy_data_x = np.array(proxy_data_x)
    proxy_data_y = np.array(proxy_data_y)
    test_data_x = np.array(test_data_x)
    test_data_y = np.array(test_data_y)

    X_proxy = torch.Tensor(proxy_data_x).type(torch.float32)
    y_proxy = torch.Tensor(proxy_data_y).type(torch.int64)
    # X_test = torch.Tensor(test_data_x).type(torch.float32)
    # y_test = torch.Tensor(test_data_y).type(torch.int64)

    data_proxy = [(x, y) for x, y in zip(X_proxy, y_proxy)]
    # data_test = [(x, y) for x, y in zip(X_test, y_test)]
    proxy_loader = DataLoader(data_proxy, batch_size=args.test_batch_size, shuffle=True)
    # test_loader = DataLoader(data_test, batch_size=args.test_batch_size, shuffle=True)
    return test_data_x, test_data_y, proxy_loader


def split_proxy(x, y, args, AT=None):
    client_x, client_at, client_y, proxy_data_x, proxy_data_at, proxy_data_y = [], [], [], [], [], []

    classes_ls = [i for i in range(args.num_classes)]
    if args.data_name == 'text':
        for client in range(args.num_user):
            dataset_image = x[client]
            dataset_at = AT[client]
            dataset_label = y[client]
            idxs = np.array(range(len(dataset_label)))
            idx_for_each_class = {}
            all_class_x = []
            all_class_at = []
            all_class_y = []
            all_class_x_proxy = []
            all_class_at_proxy = []
            all_class_y_proxy = []
            for i in classes_ls:
                idx_for_each_class[i] = idxs[dataset_label == i]
                num_class_proxy = len(idx_for_each_class[i]) * args.proxy_frac
                idx_class_proxy = np.random.choice(idx_for_each_class[i], int(num_class_proxy))
                idx_class_client = list(set(idx_for_each_class[i]) - set(idx_class_proxy))
                # proxy_data_x.extend(dataset_image[idx_class_proxy])
                # proxy_data_at.extend(dataset_at[idx_class_proxy])
                # proxy_data_y.extend(dataset_label[idx_class_proxy])
                all_class_x.extend(dataset_image[idx_class_client])
                all_class_at.extend(dataset_at[idx_class_client])
                all_class_y.extend(dataset_label[idx_class_client])
                all_class_x_proxy.extend(dataset_image[idx_class_proxy])
                all_class_at_proxy.extend(dataset_at[idx_class_proxy])
                all_class_y_proxy.extend(dataset_label[idx_class_proxy])
            client_x.append(all_class_x)
            client_at.append(all_class_at)
            client_y.append(all_class_y)
            proxy_data_x.append(all_class_x_proxy)
            proxy_data_at.append(all_class_at_proxy)
            proxy_data_y.append(all_class_y_proxy)
        # proxy_data_x = np.array(proxy_data_x)
        # proxy_data_at = np.array(proxy_data_at)
        # proxy_data_y = np.array(proxy_data_y)

        client_loaders, test_loaders = split_data(client_x, client_y, args, client_at)
        proxy_client_loaders, proxy_test_loaders = split_data(proxy_data_x, proxy_data_y, args, proxy_data_at)
        # X_proxy = torch.Tensor(proxy_data_x).type(torch.float32)
        # at_proxy = torch.Tensor(proxy_data_at).type(torch.float32)
        # y_proxy = torch.Tensor(proxy_data_y).type(torch.int64)

        # data_proxy = [(x, at, y) for x, at, y in zip(X_proxy, at_proxy, y_proxy)]

        # proxy_loader = DataLoader(data_proxy, batch_size=args.test_batch_size, shuffle=True)
    else:
        for client in range(args.num_user):
            dataset_image = x[client]
            dataset_label = y[client]
            idxs = np.array(range(len(dataset_label)))
            idx_for_each_class = {}
            all_class_x = []
            all_class_y = []
            all_class_x_proxy = []
            all_class_y_proxy = []
            for i in classes_ls:
                idx_for_each_class[i] = idxs[dataset_label == i]
                num_class_proxy = len(idx_for_each_class[i])*args.proxy_frac
                idx_class_proxy = np.random.choice(idx_for_each_class[i], int(num_class_proxy))
                idx_class_client = list(set(idx_for_each_class[i])-set(idx_class_proxy))
                all_class_x_proxy.extend(dataset_image[idx_class_proxy])
                all_class_y_proxy.extend(dataset_label[idx_class_proxy])
                all_class_x.extend(dataset_image[idx_class_client])
                all_class_y.extend(dataset_label[idx_class_client])
            client_x.append(all_class_x)
            client_y.append(all_class_y)
            proxy_data_x.append(all_class_x_proxy)
            proxy_data_y.append(all_class_y_proxy)
        # proxy_data_x = np.array(proxy_data_x)
        # proxy_data_y = np.array(proxy_data_y)

        client_loaders, test_loaders = split_data(client_x, client_y, args)
        proxy_client_loaders, proxy_test_loaders = split_data(proxy_data_x, proxy_data_y, args)

        # X_proxy = torch.Tensor(proxy_data_x).type(torch.float32)
        # y_proxy = torch.Tensor(proxy_data_y).type(torch.int64)

        # data_proxy = [(x, y) for x, y in zip(X_proxy, y_proxy)]
        # proxy_loader = DataLoader(data_proxy, batch_size=args.test_batch_size, shuffle=True)

    return client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders

def split_data(X, y, args, client_at=None):
    
    # Split dataset
    client_loaders, test_loaders = [], []
    if args.forget_paradigm == 'client':
        train_size = 0.7
    else:
        train_size = 0.99
    for i in range(len(y)):
        if args.data_name == 'text':
            X_train, X_test, at_train, at_test, y_train, y_test = train_test_split(
                X[i], client_at[i], y[i], train_size=train_size, shuffle=True)
            train_data = [(x, at, y) for x, at, y in zip(X_train,at_train, y_train)]
            test_data = [(x, at, y) for x, at, y in zip(X_test, at_test, y_test)]
            client_loaders.append(
                DataLoader(train_data, batch_size=args.local_batch_size, shuffle=True, num_workers=2, ))
            test_loaders.append(DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True))
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=train_size, shuffle=True)

            train_data = [(x, y) for x, y in zip(X_train, y_train)]
            test_data = [(x, y) for x, y in zip(X_test, y_test)]
            client_loaders.append(DataLoader(train_data, batch_size=args.local_batch_size, shuffle=True, num_workers=2,))
            test_loaders.append(DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True))

    del X, y
    # gc.collect()
    return client_loaders, test_loaders

def compress_client_loaders(client_loaders, args, model):
    
    compressed_client_loaders = []
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.device == 'cuda' else {}
    
    for loader in client_loaders:
        X = []
        y = []
        AT = [] if args.data_name == 'text' else None
        
        for batch in loader:
            if args.data_name == 'text':
                data, at, labels = batch
                X.append(data.cpu().detach().numpy())
                AT.append(at.cpu().detach().numpy())
                y.append(labels.cpu().detach().numpy())
            else:
                data, labels = batch
                if isinstance(data, torch.Tensor):
                    data = data.cpu().detach().numpy()
                X.append(data)
                y.append(labels.cpu().detach().numpy())
        
        X = np.concatenate(X)
        y = np.concatenate(y)
        if args.data_name == 'text':
            AT = np.concatenate(AT)

        if args.data_name == 'text':
            compressed_X, compressed_y, compressed_AT = compress_data_for_lora(
                X, y, args, model,
                compression_method=args.compression_method,
                AT=AT
            )
            compressed_dataset = TensorDataset(
                torch.tensor(compressed_X).type(torch.float32),
                torch.tensor(compressed_AT).type(torch.float32),
                torch.tensor(compressed_y).type(torch.int64)
            )
        else:
            compressed_X, compressed_y = compress_data_for_lora(
                X, y, args, model,
                compression_method=args.compression_method
            )
            compressed_dataset = TensorDataset(
                torch.tensor(compressed_X).type(torch.float32),
                torch.tensor(compressed_y).type(torch.int64)
            )
        
        compressed_loader = DataLoader(
            compressed_dataset,
            batch_size=args.local_batch_size,
            shuffle=True,
            **kwargs
        )
        compressed_client_loaders.append(compressed_loader)
    
    return compressed_client_loaders


