import torch

class FGT():
    def __init__(self,net,trainloader,testloader,criterion,optimizer,num_epoch,device=0):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.epochs = 0

    def train(self):
        for epochs in range(self.num_epoch):
            self.epochs = epochs
            self.epoch()
            self.inference()


    def epoch(self):
        self.net.train()
        running_loss = 0.0
        correct_cnt = 0
        total_cnt = 0
        for i, data in enumerate(self.trainloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels = data
            batch = inputs.shape[0]

            self.optimizer.zero_grad()

            outputs = self.net(inputs)
            self.loss = self.criterion(outputs, labels)
            self.loss.backward()
            self.optimizer.step()

            running_loss += self.loss.item()

            _, argmax = outputs.max(1)
            correct_cnt += (argmax == labels).sum()
            total_cnt += batch
            if i+1 == int(len(self.trainloader.dataset)/batch):    # print every 2000 mini-batches
                print('train [%d, %5d] loss: %.3f' %
                      (self.epochs + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        return running_loss, (correct_cnt.item()/total_cnt)*100

    def inference(self):
        self.net.eval()
        running_loss = 0.0
        correct_cnt = 0
        total_cnt = 0
        with torch.no_grad():
            for i, data in enumerate(self.testloader, 0):
                inputs, labels = data
                batch = inputs.shape[0]

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                self.loss = self.criterion(outputs, labels)

                running_loss += self.loss.item()

                _, argmax = outputs.max(1)
                correct_cnt += (argmax == labels).sum()
                total_cnt += batch

                if i+1 == int(len(self.testloader.dataset)/batch):    # print every 2000 mini-batches
                    print('test  [%d, %5d] loss: %.3f' %
                          (self.epochs + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        return running_loss, (correct_cnt.item()/total_cnt)*100


