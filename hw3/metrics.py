import torchaudio
import torch
import pandas as pd
from pesq import pesq
from torchaudio.transforms import Resample
from torchmetrics.audio import PerceptualEvaluationSpeechQuality


def n2(tens):
    return (tens*tens).sum()

def norm(tens):
    return n2(tens)**.5

def sdr(ref, deg):
    # print('sdr:', n2(deg) / n2(deg-ref))
    return 10*torch.log10(n2(deg) / n2(deg-ref))

def si_sdr(ref, deg):
    alpha = norm(deg*ref)/n2(deg)
    return sdr(ref, deg*alpha)

def mix(a, b, ratio_db):
    # ratio_db = 2*10*log10(ratio)
    ratio = 10 ** (ratio_db/20) / norm(a) * norm(b)
    m = a + b/ratio
    return m


def main():
    voice, voice_rate = torchaudio.load('gt.wav')
    voice = voice[0]

    r = Resample(48000, 16000)
    fnames = []
    snr_vals = [-5, 0, 5, 10]
    pesq_vals = []
    sdr_vals = []
    sisdr_vals = []
    for snr in snr_vals:
        fnames.append(f'filtered/mix_{snr}_DeepFilterNet3.wav')
        m, rate = torchaudio.load(fnames[-1])
        assert rate == voice_rate
        m = m[0]
        pesq_vals.append(pesq(16000, r(voice).numpy(), r(m).numpy()))
        sdr_vals.append(float(sdr(voice, m)))
        sisdr_vals.append(float(si_sdr(voice, m)))
    df = pd.DataFrame({
        'Filename' : fnames,
        'SNR' : snr_vals,
        'SDR' : sdr_vals,
        'SI-SDR' : sisdr_vals,
        'PESQ' : pesq_vals,
        'NISQA' : [' ']*4,
        'DNSMOS' : [' ']*4,
        'MOS' : [' ']*4
    })

    print(df)
    print('Table was saved to output.md')

    with open('ouput.md', 'w') as table_output: 
        table_output.write(df.to_markdown())


if __name__ == '__main__':
    main()