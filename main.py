import FacenetGapDataset as FGD
import FacenetGapTrainer as FGT
import FacenetGapNet as FGN

import torch
import torch.nn as nn
import torch.optim as optim

net = FGN.FGN()

datasets = FGD.FGD('data')

trainset,testset = datasets.get()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=8)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

trainer = FGT.FGT(net,trainloader,testloader,criterion,optimizer,num_epoch=100,device = 0)

trainer.load('./state/FG_acc_94.88971679140154.pt')

loss,acc = trainer.train()
path = trainer.save('state',acc)

print('Finished Training')
