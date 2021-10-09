import torch
from util import *
from unet import *
from cnnblstm import *

def pad32(x):
    shape = x.shape
    pad1 = 32-shape[2]%32
    pad2 = 32-shape[3]%32
    m = nn.ZeroPad2d((pad2, 0, pad1, 0))
    return m(x), pad1, pad2

def dip_mscl_batch(s_hat, mode='unet', num_iter=1000, avg_num=1, avg_mode='mean', plot_every=200, seed=1234, device='cpu', isrelu=False):
    S_hat = stft(s_hat, win=np.hamming, fftl=512, shift=128)
    As_hat = np.abs(S_hat)
    As_hat[As_hat==0] = np.spacing(1)
    target = torch.from_numpy(As_hat).clone()
    target = torch.unsqueeze(target, 0)
    target = torch.unsqueeze(target, 0)
    if mode=='unet': target, pad1, pad2 = pad32(target)
    target = torch.repeat_interleave(target, avg_num, dim=0)

    target = target.to(device)
    torch.manual_seed(seed)
    if mode=='unet':
        model = Unet(isrelu).to(device)
    elif mode=='blstm':
        FFTL = 512
        IN_DIM = int(FFTL/2+1)
        HID_DIM = 100
        LAYER = 2
        DO_RATIO = 0.5
        model = CNN_BLSTM(IN_DIM, HID_DIM, LAYER, DO_RATIO).to(device)

    L1 = torch.nn.L1Loss().to(device)

    params = []
    params += [x for x in model.parameters() ]
    optimizer = torch.optim.Adam(params, lr=0.01)
    net_input = torch.rand(size=target.shape)*0.1
    net_input = net_input.to(device)
    for i in range(num_iter+1):
        print('\r%d回目' %(i), end='')
        optimizer.zero_grad()
        out = model(net_input)
        if plot_every>0 and i%plot_every==0:
            out_np = out.to('cpu').detach().numpy().copy()
            out = out.to('cuda')
            if mode=='unet': out_np = out_np[:,:,pad1:, pad2:]
            if avg_mode=='mean': out_avg = np.mean(out_np[:,0], axis=0)
            elif avg_mode=='median': out_avg = np.median(out_np[:,0], axis=0)
            else: out_avg = out_np[0,0]
            out_wav = istft(out_avg*np.exp(1j*np.angle(S_hat)),x_len=len(s_hat))
            plt.clf()
            specshow(out_wav, fig_size=(10, 3), v_min=-100, v_max=40)
            plt.pause(0.05)

        loss = L1(out, target)
        loss.backward()
        optimizer.step()
    out = model(net_input)
    out_np = out.to('cpu').detach().numpy().copy()
    if mode=='unet': out_np = out_np[:,:,pad1:, pad2:]
    if avg_mode=='mean': out_avg = np.mean(out_np[:,0], axis=0)
    elif avg_mode=='median': out_avg = np.median(out_np[:,0], axis=0)
    proc = istft(out_avg*np.exp(1j*np.angle(S_hat)), win=np.hamming, fftl=512, shift=128, x_len=len(s_hat))
    return proc
