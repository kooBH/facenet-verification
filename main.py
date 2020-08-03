import FacenetGapDataset as FGD
import FacenetGapTrainer as FGT

import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#        self.conv1 = nn.Conv1d(3, 6, 5)
#        self.pool = nn.MaxPool1d(2, 2)
#        self.conv2 = nn.Conv1d(6, 16, 5)
        self.fc1 = nn.Linear(128, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

datasets = FGD.FGD('data')

trainset,testset = datasets.get()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=12)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=12)

classes = ('same','diff')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

trainer = FGT.FGT(net,trainloader,testloader,criterion,optimizer,5,0)

trainer.train()



print('Finished Training')
