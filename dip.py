import torch
from util import *
from unet import *

def pad32(x):
    shape = x.shape
    pad1 = 32-shape[2]%32
    pad2 = 32-shape[3]%32
    m = nn.ZeroPad2d((pad2, 0, pad1, 0))
    return m(x), pad1, pad2

def dip_musical(s_musical, num_iter=1000, avg_num=1, plot_every=100, isrelu=True, device='cuda:0', learning_rate=0.01, seed=1234):
    S_musical = stft(s_musical, win=np.hamming, fftl=512, shift=128)
    As_musical = np.abs(S_musical)
    As_musical[As_musical==0] = np.spacing(1)
    target = torch.from_numpy(As_musical).clone()
    target = torch.unsqueeze(target, 0)
    target = torch.unsqueeze(target, 0)
    target, pad1, pad2 = pad32(target)
    target = torch.repeat_interleave(target, avg_num, dim=0)

    target = target.to(device)
    torch.manual_seed(seed)
    model = Unet(isrelu).to(device)
    L1 = torch.nn.L1Loss().to(device)

    params = []
    params += [x for x in model.parameters() ]
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    net_input = torch.rand(size=target.shape)*0.1
    net_input = net_input.to(device)

    for i in range(num_iter+1):
        print('\r%d回目' %(i), end='')
        optimizer.zero_grad()
        net_out = model(net_input)
        if plot_every>0 and i%plot_every==0:
            out_np = net_out.to('cpu').detach().numpy().copy()
            net_out = net_out.to(device)
            out_np = out_np[:,:,pad1:, pad2:]
            out_avg = np.mean(out_np[:,0], axis=0)
            proc = istft(out_avg*np.exp(1j*np.angle(S_musical)),x_len=len(s_musical))
            plt.clf()
            specshow(proc, fig_size=(10, 3), v_min=-100, v_max=40, title='result %d' %(i))
            plt.pause(0.05)
        loss = L1(net_out,target)
        loss.backward()
        optimizer.step()

    net_out = model(net_input)
    out_np = net_out.to('cpu').detach().numpy().copy()
    out_np = out_np[0,0,pad1:, pad2:]
    out_avg = np.mean(out_np[:,0], axis=0)
    proc = istft(out*np.exp(1j*np.angle(S_musical)),x_len=len(s_musical))
    return proc
