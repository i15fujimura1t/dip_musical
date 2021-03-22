import soundfile as sf
import numpy as np
from scipy import signal
import scipy.fftpack as scft

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