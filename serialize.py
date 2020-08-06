import torch
import FacenetGapNet as FGN

if __name__ == '__main__':
    net = FGN.FGN()
    example_input = torch.rand(1,128)
    print('loading pre-trainded model..')
    net.load_state_dict(torch.load('./state/FG_acc_97.15142923081996.pt'))
    net.eval()

    print('tracing')
    traced_model = torch.jit.trace(net, example_input)
    print('saving')
    traced_model.save('traced-stat.pt')
