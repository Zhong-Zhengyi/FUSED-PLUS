import torch
import torch.nn as nn
import torch.nn.init as init
# from models.layers.passportconv2d_private import PassportPrivateBlock

class AlexNet(nn.Module):

    def __init__(self, num_classes,in_channels): #in_channels, 
        super().__init__()
        self.num_classes=num_classes
        maxpoolidx = [1, 3, 7]
        layers = []
        inp = in_channels #in_channels
        oups = {
            0: 64,
            2: 192,
            4: 384,
            5: 256,
            6: 256
        }
        kp = {
            0: (5, 2),
            2: (5, 2),
            4: (3, 1),
            5: (3, 1),
            6: (3, 1)
        }
        for layeridx in range(8):
            if layeridx in maxpoolidx:
                layers.append(nn.MaxPool2d(2, 2))
            else:
                k = kp[layeridx][0]
                p = kp[layeridx][1]
                # if passport_kwargs[str(layeridx)]['flag']:
                #     layers.append(PassportPrivateBlock(inp, oups[layeridx], k, 1, p))
                # else:
                layers.append(ConvBlock(inp, oups[layeridx], k, 1, p))
                inp = oups[layeridx]

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(4*4*256, num_classes)

    def forward(self, x):
        for m in self.features:
            x = m(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        # a=torch.nn.functional.softmax(x[:,0:int(self.num_classes/2)]) #softmax
        # #print(a.size())
        # b=torch.nn.functional.softmax(x[:,int(self.num_classes/2):self.num_classes])
        #print(b.size())
        # z=torch.cat((a,b),dim=1)
        return x
    
class AlexNet_UL(nn.Module):

    def __init__(self, num_classes,in_channels): #, 
        super().__init__()
        self.num_classes=num_classes
        maxpoolidx = [1, 3, 7]
        layers = []
        inp = in_channels #in_channels
        oups = {
            0: 64,
            2: 192,
            4: 384,
            5: 256,
            6: 256
        }
        kp = {
            0: (5, 2),
            2: (5, 2),
            4: (3, 1),
            5: (3, 1),
            6: (3, 1)
        }
        for layeridx in range(8):
            if layeridx in maxpoolidx:
                layers.append(nn.MaxPool2d(2, 2))
            else:
                k = kp[layeridx][0]
                p = kp[layeridx][1]
                # if passport_kwargs[str(layeridx)]['flag']:
                #     layers.append(PassportPrivateBlock(inp, oups[layeridx], k, 1, p))
                # else:
                layers.append(ConvBlock(inp, oups[layeridx], k, 1, p))
                inp = oups[layeridx]

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(4*4*256, num_classes)
        self.classifier_ul = nn.Linear(4*4*256, num_classes)  #

    def forward(self, x):
        for m in self.features:
            x = m(x)
        print("x shape:", x.shape)
        print("classifier weight shape:", self.classifier.weight.shape)
        x = x.view(x.size(0), -1)
        a = self.classifier(x)
        b = self.classifier_ul(x)
        z = torch.cat((a,b),dim=1)
        

        # a=torch.nn.functional.softmax(x[:,0:int(self.num_classes/2)]) #softmax
        # #print(a.size())
        # b=torch.nn.functional.softmax(x[:,int(self.num_classes/2):self.num_classes])
        #print(b.size())
        # z=torch.cat((a,b),dim=1)
        return z
    
class ConvBlock(nn.Module):
    def __init__(self, i, o, ks=3, s=1, pd=1, relu=True):
        super().__init__()

        self.conv = nn.Conv2d(i, o, ks, s, pd, bias= False)

        # if bn == 'bn':
        #     self.bn = nn.BatchNorm2d(o)
        # else:
        #     self.bn = None

        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        # if self.bn is not None:
        #     x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
