import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from dataset.data_utils import data_set, separate_data, split_proxy
import numpy as np
import os
import pickle as pkl
import pandas as pd
import tqdm
import random
from transformers import BertTokenizer
import copy

MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'

# Allocate data to users


def data_init(FL_params):
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.device == 'cuda' else {}
    dataset_x = []
    dataset_at = []
    dataset_y = []
    if FL_params.data_name == 'text':
        trainset, testset = data_set_text(FL_params, True)
        test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, num_workers=2,
                                  **kwargs)
        for sample in trainset:
            values = list(sample.values())
            dataset_x.append(np.array(values[0]))
            dataset_at.append(np.array(values[1]))
            dataset_y.append(int(values[2]))
        for sample in testset:
            values = list(sample.values())
            dataset_x.append(np.array(values[0]))
            dataset_at.append(np.array(values[1]))
            dataset_y.append(int(values[2]))

    else:
        trainset, testset = data_set(FL_params.data_name)

        test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, num_workers=2, **kwargs)
        train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, num_workers=2, **kwargs)

        for train_data in train_loader:
            x_train, y_train = train_data
            dataset_x.extend(x_train.cpu().detach().numpy())
            dataset_y.extend(y_train.cpu().detach().numpy())
        if FL_params.forget_paradigm == 'client':
            for test_data in test_loader:
                x_test, y_test = test_data
                dataset_x.extend(x_test.cpu().detach().numpy())
                dataset_y.extend(y_test.cpu().detach().numpy())

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)
    if FL_params.data_name == 'text':
        dataset_at = np.array(dataset_at)

        X, AT, y, statistic = separate_data((dataset_x, dataset_at, dataset_y), FL_params.num_user, FL_params.num_classes, FL_params,
                                        FL_params.niid, FL_params.balance, FL_params.partition, class_per_client=2)
        client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders = split_proxy(X, y, FL_params, AT)
    else:
        X, y, statistic = separate_data((dataset_x, dataset_y), FL_params.num_user, FL_params.num_classes, FL_params,
                                        FL_params.niid, FL_params.balance, FL_params.partition, class_per_client=2)
        client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders = split_proxy(X, y, FL_params)
    
    FL_params.datasize_ls = [len(k) for k in X]
    if FL_params.forget_paradigm == 'client':
        test_loaders = test_loaders
        proxy_test_loaders = proxy_test_loaders
    else:

        proxy_test_x = []
        proxy_test_at = []
        proxy_test_y = []
        for i in range(FL_params.num_user):
            if FL_params.data_name == 'text':
                for data, at, label in test_loaders[i]:
                    proxy_test_x.append(data)
                    proxy_test_at.append(at)
                    proxy_test_y.append(label)
            else:
                for batch in test_loaders[i]:
                    data, label = batch
                    proxy_test_x.append(data)
                    proxy_test_y.append(label)
        proxy_test_x = torch.cat(proxy_test_x).numpy()
        if FL_params.data_name == 'text':
            proxy_test_at = torch.cat(proxy_test_at).numpy()
        proxy_test_y = torch.cat(proxy_test_y).numpy()
        if FL_params.data_name == 'text':
            proxy_test_loader = DataLoader(TensorDataset(torch.tensor(proxy_test_x), torch.tensor(proxy_test_at), torch.tensor(proxy_test_y)), batch_size=FL_params.test_batch_size, shuffle=True)
        else:
            proxy_test_loader = DataLoader(TensorDataset(torch.tensor(proxy_test_x), torch.tensor(proxy_test_y)), batch_size=FL_params.test_batch_size, shuffle=True)
        proxy_test_loaders = [proxy_test_loader for _ in range(FL_params.num_user)]
        
    return client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders

def cross_data_init(FL_params):
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.device == 'cuda' else {}
    dataset_x = []
    dataset_y = []
    if FL_params.data_name == 'text':
        trainset, testset = data_set_text(FL_params, True)
        for (x, y) in trainset:
            dataset_x.append(x)
            dataset_y.append(y)
        for (x, y) in testset:
            dataset_x.append(x)
            dataset_y.append(y)
    else:
        trainset, testset = data_set(FL_params.data_name)

        # # 构建测试数据加载器
        test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, num_workers=2, **kwargs)
        train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, num_workers=2,
                                  **kwargs)

        for train_data in train_loader:
            x_train, y_train = train_data
            dataset_x.extend(x_train.cpu().detach().numpy())
            dataset_y.extend(y_train.cpu().detach().numpy())
        if FL_params.forget_paradigm == 'client':
            for test_data in test_loader:
                x_test, y_test = test_data
                dataset_x.extend(x_test.cpu().detach().numpy())
                dataset_y.extend(y_test.cpu().detach().numpy())

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)

    class_num = int(FL_params.num_classes/FL_params.num_user)
    X = []
    y = []
    idx_ls = []
    for user in range(FL_params.num_user):
        idx = []
        for i in range(class_num):
            item = user*class_num + i
            indices = [idx for idx, label in enumerate(dataset_y) if label == item]
            idx.extend(indices)
        idx_ls.append(idx)
    corss_idx = idx_ls[0][:int(len(idx_ls[0])*0.01)]
    idx_ls[0] = idx_ls[0][int(len(idx_ls[0])*0.01):]
    idx_ls[1] = corss_idx + idx_ls[1]
    remain_idx = []
    for idx in range(1, FL_params.num_user):
        remain_idx.extend(idx_ls[idx])
    random.shuffle(remain_idx)
    sublist_size = len(remain_idx) // (FL_params.num_user-len(FL_params.forget_client_idx))
    remainder = len(remain_idx) % (FL_params.num_user-len(FL_params.forget_client_idx))

    # Create the sublists
    sublists = [remain_idx[i * sublist_size + min(i, remainder):(i + 1) * sublist_size + min(i + 1, remainder)] for i in
                range(9)]

    for idx in range(1, FL_params.num_user):
        idx_ls[idx] = sublists[idx-1]

    for user in range(FL_params.num_user):
        X.append(dataset_x[idx_ls[user]])
        y.append(dataset_y[idx_ls[user]])

    for i in range(FL_params.num_user):
        print('client {} data size {} lable {}'.format(i, len(X[i]),np.unique(y[i])))

    # client_loaders, test_loaders, proxy_loader = split_proxy(X, y, FL_params)
    client_loaders, test_loaders, proxy_loader, proxy_test_loaders = split_proxy(X, y, FL_params)
    FL_params.datasize_ls = [len(k) for k in X]
    if FL_params.forget_paradigm == 'client':
        test_loaders = test_loaders
    else:
        test_loaders = [test_loader for _ in range(FL_params.num_user)]

    return client_loaders, test_loaders, proxy_loader, proxy_test_loaders

def data_set_text(config, ues_word):
    # if ues_word:
    #     tokenizer = lambda x: x.split(' ')  # word-level, split the word with space
    # else:
    #     tokenizer = lambda x: [y for y in x]  # char-level
    # if os.path.exists(config.vocab_path):
    #     vocab = pkl.load(open(config.vocab_path, 'rb'))
    # # else:
    # #     vocab = build_vocab_from_csv(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    # #     pkl.dump(vocab, open(config.vocab_path, 'wb'))
    # print('Vocab size: {}'.format(len(vocab)))
    config.pad_size = 300
    model_path = './models/Bert'
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
    train_data_path = []
    test_data_path = []
    for i in range(config.num_classes):
        train_data_path.append('./dataset/text_data/text_train_{}_class.csv'.format(i))
        test_data_path.append('./dataset/text_data/text_eval_{}_class.csv'.format(i))

    def load_dataset_from_csv(path, pad_size=128):
        dataset = pd.read_csv(path)
        dataset = dataset.values
        contents = []
        labels = []
        num = 0
        for line in dataset:
            num += 1
            if num <= 1000:
                try:
                    content, label = str(line[0]).strip() + str(line[1]).strip(), int(line[2])
                except AttributeError:
                    content, label = line[0].strip(), int(line[2])

                # words_line = []
                # token = tokenizer(content)
                # seq_len = len(token)
                # if pad_size:
                #     if len(token) < pad_size:
                #         token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                #     else:
                #         token = token[:pad_size]
                #         seq_len = pad_size
                # # word to id
                # for word in token:
                #     words_line.append(vocab.get(word, vocab.get(UNK)))
                # contents.append((words_line, int(label)))
                contents.append(content)
                labels.append(label)

        return contents , labels  # [([...], 0), ([...], 1), ...]

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for train_path in train_data_path:
        x_train.extend(load_dataset_from_csv(train_path, config.pad_size)[0])
        y_train.extend(load_dataset_from_csv(train_path, config.pad_size)[1])
    for test_path in test_data_path:
        x_test.extend(load_dataset_from_csv(test_path, config.pad_size)[0])
        y_test.extend(load_dataset_from_csv(test_path, config.pad_size)[1])
    trains = TextDataset(x_train, y_train, tokenizer)
    tests = TextDataset(x_test,  y_test, tokenizer)
    return trains, tests

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }