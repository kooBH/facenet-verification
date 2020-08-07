import torch
import FacenetGapNet as FGN

if __name__ == '__main__':
    net = FGN.FGN()
    example_input = torch.rand(2,128)
    print('loading pre-trainded model..')
    net.load_state_dict(torch.load('./state/state_acc_99.1408.pt'))
    net.eval()

    print('tracing')
    traced_model = torch.jit.trace(net, example_input)
    print('saving')
    traced_model.save('traced-state.pt')
