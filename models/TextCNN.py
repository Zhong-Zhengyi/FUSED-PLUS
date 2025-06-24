#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import numpy as np
from models.Model_base import MyModel
import copy


# class Model(MyModel):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         if config.embedding_pretrained is not None:
#             self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
#         else:
#             self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
#         self.dropout = nn.Dropout(config.dropout)
#         self.fc100 = nn.Linear(config.num_filters * len(config.filter_sizes), 100)
#         self.fc = nn.Linear(100, config.num_classes)
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)).squeeze(3)
#         x = F.max_pool1d(x, x.size(2)).squeeze(2)  # one-dimension max-pooling
#         return x
#
#     def forward(self, x):
#         x = x.long()
#         out = self.embedding(x)
#         out = out.unsqueeze(1)
#         out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
#         out = self.dropout(out)
#         mid_val = self.fc100(out)
#         out = self.fc(mid_val)
#         return out, mid_val

# class Model(MyModel):
#     def __init__(self, config):
#         model_name = 'bert-base-uncased'
#         super(Model, self).__init__()
#         model_path = './Bert'
#         self.tokenizer = BertTokenizer.from_pretrained(model_path)
#         self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=config.num_classes)

#     def forward(self, input_ids, attention_mask):
#         return self.model(input_ids=input_ids, attention_mask=attention_mask)

class Model(MyModel):
    def __init__(self, config):
        super(Model, self).__init__()
        model_name = 'bert-base-uncased'
        model_path = './models/Bert'
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=10, local_files_only=True)
        

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
    
class Model_UL(MyModel):
    def __init__(self, config):
        super(Model_UL, self).__init__()
        model_name = 'bert-base-uncased'
        model_path = './models/Bert'
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=10, local_files_only=True)
        self.classifier = nn.Linear(self.model.config.hidden_size, config.num_classes)
        self.classifier_ul = nn.Linear(self.model.config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 获取最后一层的隐藏状态
        last_hidden_state = outputs[0]
        a = self.classifier(last_hidden_state[:, 0, :])  # 使用 [CLS] token 的输出
        b = self.classifier_ul(last_hidden_state[:, 0, :])  # 使用 [CLS] token 的输出

        # 将两个输出连接
        z = torch.cat((a, b), dim=1)

        return z