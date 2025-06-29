import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.Model_base import MyModel

class Model(MyModel):
    def __init__(self,config):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(300, 50)
        self.fc3 = nn.Linear(50,config.num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        mid_value = F.relu(self.fc2(x))
        x = F.relu(self.fc3(mid_value))
        return x, mid_value