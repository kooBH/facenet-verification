import FacenetGapDataset as FGD
import FacenetGapTrainer as FGT
import FacenetGapNet as FGN

import torch
import torch.nn as nn
import torch.optim as optim

net = FGN.FGN()

criterion = nn.CrossEntropyLoss()
#criterion = nn.BCEWithLogitsLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001,betas=(0.9,0.999), eps=1e-08,weight_decay=0)

#net.load_state_dict(torch.load("./state/state_acc_97.63114971740697.pt"))

epoch = 10

for i in range(10):
    print("iter : " + str(i) + " for epoch : "+str(epoch))
    datasets = FGD.FGD('data')

    trainset,testset = datasets.get()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=8)
    print('dataset load')
    trainer = FGT.FGT(net,trainloader,testloader,criterion,optimizer,num_epoch=epoch,device = 0)
    loss,acc = trainer.train()
    path = trainer.save('state',acc)


print('Finished Training')
