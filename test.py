import FacenetGapNet as FGN
import numpy as np
import torch 

def getDist(a,b):
    aa = np.fromfile(a,np.float32)
    bb = np.fromfile(b,np.float32)
    t = np.abs(aa-bb)
    return t

net = FGN.FGN()

net.load_state_dict(torch.load('./state/FG_acc_92.47634433417954.pt'))

a = '/home/nas/user/kbh/FaceVerification/original/kjh43.facebin'
#b = '/home/nas/user/kbh/FaceVerification/original/kbh1.facebin'
b = '/home/nas/user/kbh/FaceVerification/original/kjh1.facebin'

t = getDist(a,b)
t = torch.Tensor(t)

outputs = net(t)

print(outputs)
_, argmax = outputs.max(0)
print(argmax)


