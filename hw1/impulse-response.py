import numpy as np
import scipy as sp
import math
import torch
import torchaudio


def get_spectr(signal):
    spectr = torch.fft.fft(signal)
    return spectr[1:len(spectr)//2+1]

def get_signal(spectr):
    spectr = torch.concat([torch.tensor([0]), spectr[:-1], spectr.flip(0).conj()])
    return torch.fft.ifft(spectr).real

def get_band_borders(T, n_bands=32, min_rate=20, max_rate=24000):
    return torch.exp(torch.linspace(math.log(min_rate), math.log(max_rate), n_bands+1))*T

def get_band_energy(spectr, borders):
    energy = []
    for s, e in zip(borders[:-1], borders[1:]):
        energy.append(spectr[int(s):int(e)].abs().mean())
    return torch.tensor(energy)

def apply_gain(signal, gain, borders):
    spectr = get_spectr(signal)
    for i in range(len(borders)-1):
        spectr[int(borders[i]):int(borders[i+1])] *= gain[i]
    return get_signal(spectr)

def deconvolve(x, y):
    y = y[:len(x)]
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    H = Y/X
    h = np.fft.irfft(H)
    return h


def main():
    rate = 48000
    T = 10
    min_freq = 20
    max_freq = 20000
    freq = torch.exp(torch.linspace(math.log(min_freq), math.log(max_freq), rate*T))
    delta = freq / rate * 2 * math.pi
    sweep = torch.sin(torch.cumsum(delta, 0))
    torchaudio.save('mysweep.wav', sweep.unsqueeze(0), rate)

    sweep_rec, rate = torchaudio.load('sweep_record5.mp3')
    sweep_rec = sweep_rec[0]
    sweep_rec = sweep_rec[:int(len(sweep_rec)//100)*100]

    shift = np.argmax(sp.signal.convolve(sweep, sweep_rec, mode='valid'))
    print(shift, T*rate)
    sweep_rec = sweep_rec[shift : shift+T*rate]

    spectr = get_spectr(sweep)
    spectr_rec = get_spectr(sweep_rec)

    borders = get_band_borders(T)
    orig_bands = get_band_energy(spectr, borders)
    rec_bands = get_band_energy(spectr_rec, borders)
    gain = rec_bands/orig_bands

    noise, rate = torchaudio.load('white-noise.wav')
    noise = noise[0]
    noise_gained = apply_gain(noise, gain**-1, borders)
    torchaudio.save('noise_gained.wav', noise_gained.unsqueeze(0), rate)

    noise_record, rate = torchaudio.load('noise_record4.mp3')
    noise_record = noise_record[0]
    noise_record = noise_record[1*rate:5*rate]

    conv = sp.signal.convolve(noise_gained.numpy(), noise_record.numpy(), mode='valid')
    shift = np.argmax(conv)

    deconv = deconvolve(noise[shift:shift+3*rate], noise_record)

    gt, rate = torchaudio.load('gt.wav')
    gt = gt[0]
    gt_conv = sp.signal.convolve(gt, deconv)
    torchaudio.save('gt_conv.wav', torch.tensor(gt_conv).float().unsqueeze(0), rate)


if __name__ == '__main__':
    main()