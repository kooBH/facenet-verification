import torch

class FGT():
    def __init__(self,net,trainloader,testloader,criterion,optimizer,num_epoch,device=0):
        self.device = torch.device('cuda:{}'.format(device))
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.epochs = 0

        self.net = self.net.to(self.device)

    def train(self):
        train_acc = 0
        train_loss = 0
        test_acc =0
        test_loss =0
        for epochs in range(self.num_epoch):
            train_loss,train_acc = self.epoch()
            test_loss, test_acc  = self.inference()
            print("epoch {} | train [{} , {}] | test [{} , {}]".format(epochs,train_loss,train_acc,test_loss,test_acc))
        return test_loss, test_acc


    def epoch(self):
        self.net.train()
        running_loss = 0.0
        correct_cnt = 0
        total_cnt = 0
        for i, data in enumerate(self.trainloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            batch = inputs.shape[0]

            self.optimizer.zero_grad()

            outputs = self.net(inputs)
            self.loss = self.criterion(outputs, labels)
            self.loss.backward()
            self.optimizer.step()

            running_loss += self.loss.item()
            _, argmax = outputs.max(1)
#            print('-------------------')
#            print(argmax)
#            print(labels)
            correct_cnt += (argmax == labels).sum()
            total_cnt += batch
        return running_loss, (correct_cnt.item()/total_cnt)*100

    def inference(self):
        self.net.eval()
        running_loss = 0.0
        correct_cnt = 0
        total_cnt = 0
        with torch.no_grad():
            for i, data in enumerate(self.testloader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                batch = inputs.shape[0]

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                self.loss = self.criterion(outputs, labels)

                running_loss += self.loss.item()

                _, argmax = outputs.max(1)
                correct_cnt += (argmax == labels).sum()
                total_cnt += batch

        return running_loss, (correct_cnt.item()/total_cnt)*100

    def load(self,path):
        self.net.load_state_dict(torch.load(path))

    def save(self,path,acc):
        path = path+"/FG_acc_{}.pt".format(acc)
        torch.save(self.net.state_dict(),path)
        return path


