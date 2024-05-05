import math
import torch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from IPython.display import Audio


def _get_log_freq(sample_rate, max_sweep_rate):
    """Get freqs evenly spaced out in log-scale, between [0, max_sweep_rate // 2]

    offset is used to avoid negative infinity `log(offset + x)`.

    """
    DEFAULT_OFFSET = 201
    start, stop = math.log(DEFAULT_OFFSET), math.log(DEFAULT_OFFSET + max_sweep_rate // 2)
    return torch.exp(torch.linspace(start, stop, sample_rate, dtype=torch.double)) - DEFAULT_OFFSET

def get_sine_sweep(sample_rate):
    max_sweep_rate = sample_rate//2
    freq = _get_log_freq(sample_rate, max_sweep_rate)
    delta = 2 * math.pi * freq / sample_rate
    cummulative = torch.cumsum(delta, dim=0)
    signal = torch.sin(cummulative).unsqueeze(dim=0)
    return signal

def _get_inverse_log_freq(freq, sample_rate):
    """Find the time where the given frequency is given by _get_log_freq"""
    DEFAULT_OFFSET = 201
    half = sample_rate // 2
    return sample_rate * (math.log(1 + freq / DEFAULT_OFFSET) / math.log(1 + half / DEFAULT_OFFSET))


def _get_freq_ticks(sample_rate, f_max):
    # Given the original sample rate used for generating the sweep,
    # find the x-axis value where the log-scale major frequency values fall in
    times, freq = [], []
    for exp in range(2, 5):
        for v in range(1, 10):
            f = v * 10**exp
            if f < sample_rate // 2:
                t = _get_inverse_log_freq(f, sample_rate) / sample_rate
                times.append(t)
                freq.append(f)
    t_max = _get_inverse_log_freq(f_max, sample_rate) / sample_rate
    times.append(t_max)
    freq.append(f_max)
    return times, freq

def plot_sweep(
    waveform,
    sample_rate,
    title,
    max_sweep_rate=48000,
):
    x_ticks = [100, 500, 1000, 5000, 10000, 20000, max_sweep_rate // 2]
    y_ticks = [1000, 5000, 10000, 20000, sample_rate // 2]

    time, freq = _get_freq_ticks(max_sweep_rate, sample_rate // 2)
    freq_x = [f if f in x_ticks and f <= max_sweep_rate // 2 else None for f in freq]
    freq_y = [f for f in freq if f in y_ticks and 1000 <= f <= sample_rate // 2]

    figure, axis = plt.subplots(1, 1)
    _, _, _, cax = axis.specgram(waveform[0].numpy(), Fs=sample_rate)
    plt.xticks(time, freq_x)
    plt.yticks(freq_y, freq_y)
    axis.set_xlabel("Original Signal Frequency (Hz, log scale)")
    axis.set_ylabel("Waveform Frequency (Hz)")
    axis.xaxis.grid(True, alpha=0.67)
    axis.yaxis.grid(True, alpha=0.67)
    figure.suptitle(f"{title} (sample rate: {sample_rate} Hz)")
    plt.colorbar(cax)
