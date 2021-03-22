import soundfile as sf
import numpy as np
from scipy import signal
import scipy.fftpack as scft
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def wavread(fn):
    data, sr = sf.read(fn)
    f = sf.info(fn)
    return data, sr, f.subtype

def wavwrite(fn, data, sr, subtype, resr=None):
    if resr is None:
        sf.write(fn, data, sr, subtype)
    else:
        data = signal.resample(data, int(resr*len(data)/sr))
        sf.write(fn, data, resr, subtype)

def stft(x, win=np.hamming, fftl=512, shift=128, winl=None, onesided=True):
    if winl is None:
        winl = fftl
    assert (winl <= fftl), "FFT length < window length."
    win = np.pad(np.hamming(winl),[int(np.ceil((fftl-winl)/2)),int(np.floor((fftl-winl)/2))], 'constant')
    l = len(x)
    new_l = 2*(fftl-shift)+int(np.ceil(l/shift))*shift
    new_x = np.zeros(new_l)
    new_x[fftl-shift:fftl-shift+l] = x
    M = int((new_l-fftl+shift)/shift)
    X = np.zeros([M,fftl],dtype = np.complex128)
    shift_matrix  = np.arange(0, shift*M, shift).reshape(M,1).repeat(fftl,1)
    index_matrix = np.arange(0, fftl)
    index_matrix = index_matrix + shift_matrix
    X = scft.fft(new_x[index_matrix]*win , fftl)
    if onesided: X = X[:,:fftl//2+1]
    return X

def istft(X, win=np.hamming, fftl=512, shift=128, winl=None, x_len=None, onesided=True):
    if winl is None:
        winl = fftl
    assert (winl <= fftl), "FFT length > window length."
    win = np.pad(np.hamming(winl),[int(np.ceil((fftl-winl)/2)),int(np.floor((fftl-winl)/2))], 'constant')
    if onesided:
        X = np.hstack((X, np.conjugate(np.fliplr(X[:,1:X.shape[1]-1]))))
    M, fftl = X.shape
    l = (M-1)*shift+fftl
    xx = scft.ifft(X).real * win
    xtmp = np.zeros(l, dtype = np.float64)
    wsum = np.zeros(l, dtype = np.float64)
    for m in range(M):
        start = shift*m                                                                     
        xtmp[start : start+fftl] += xx[m, :]
        wsum[start : start + fftl] += win ** 2
    pos = (wsum != 0)                                                  
    xtmp[pos] /= wsum[pos]
    x = xtmp[fftl-shift:-fftl+shift]
    if x_len is not None:
        x = x[:x_len]
    return x

def specshow(x, sr=16000, win=np.hamming, fftl=512, shift=128, winl=None, title='', mode='mesh', fig_size=(6,3), ax=None, c_map='magma', v_min=None, v_max=None, save_path=None):
    if winl is None:
        winl = fftl
    assert (winl <= fftl), "FFT length < window length."
    win = np.pad(np.hamming(winl),[int(np.ceil((fftl-winl)/2)),int(np.floor((fftl-winl)/2))], 'constant')
    if len(x.shape)==1:
        X = stft(x, win, fftl, shift)
        f = np.linspace(0, sr//2, X.shape[1])
        t = np.arange(X.shape[0])*shift/sr
    else:
        X = x[:,:fftl//2+1]
        f = np.linspace(0, sr//2, X.shape[1])
        t = np.arange(X.shape[0])*shift/sr
    if ax is None:
        plt.figure(figsize=fig_size)
        if(mode=='mesh'):im=plt.pcolormesh(t, f, 20*np.log10(np.abs(X.T)+1e-8), cmap=c_map, norm=Normalize(vmin=v_min, vmax=v_max))
        elif(mode=='imshow'):im=plt.imshow(20*np.log10(np.abs(X.T)+1e-8)[::-1,:], cmap=c_map, extent=[0,t[-1],0,f[-1]], aspect='auto', norm=Normalize(vmin=v_min, vmax=v_max))
        plt.title(title)
        plt.ylabel('Freq. [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(format='%+2.0f dB')
        if(save_path is not None):
            plt.savefig('%s' %(save_path))
            plt.show()
    else:
        if(mode=='mesh'):im=ax.pcolormesh(t, f, 20*np.log10(np.abs(X.T)+1e-8), cmap=c_map, norm=Normalize(vmin=v_min, vmax=v_max))
        elif(mode=='imshow'):im=ax.imshow(20*np.log10(np.abs(X.T)+1e-8)[::-1,:], cmap=c_map, extent=[0,t[-1],0,f[-1]], aspect='auto', norm=Normalize(vmin=v_min, vmax=v_max))