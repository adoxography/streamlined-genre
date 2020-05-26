import math
import random
import sys
from pathlib import Path

import numpy as np
import librosa
from librosa.feature import melspectrogram
import soundfile as sf
import matplotlib.pyplot as plt
from nlpaug.augmenter.audio import AudioAugmenter
from nlpaug.augmenter.audio import MaskAug, VtlpAug, SpeedAug
from nlpaug.flow import Sometimes
from nlpaug.util import Action
from scipy.signal import butter, sosfilt

sample_rate = 22050


def freq_to_mel(freq):
    return 2595 * math.log10(1 + freq / 700)


def mel_to_freq(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def butter_bandstop(lowcut, highcut, sampling_rate, order=5):
    nyquist_freq = 0.5 * sampling_rate
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    try:
        sos = butter(order, [low, high], analog=False,
                     btype='bandstop', output='sos')
    except ValueError as e:
        print('sr:', sampling_rate)
        print('Nyquist:', nyquist_freq)
        print('Low:', lowcut, low)
        print('high:', highcut, high)
        raise e
    return sos


def butter_bandstop_filter(data, lowcut, highcut, sampling_rate, order=5):
    """
    See https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter/12233959#12233959  # noqa
    """
    sos = butter_bandstop(lowcut, highcut, sampling_rate, order=order)
    y = sosfilt(sos, data)
    return y


class BandpassAug(AudioAugmenter):
    def __init__(self, sampling_rate=None, span=200, name='BandpassAug',
                 verbose=0):
        self.sampling_rate = sampling_rate
        self.span = span
        super().__init__(name=name, action=Action.SUBSTITUTE, verbose=verbose)

    def substitute(self, data):
        max_filter = freq_to_mel(self.sampling_rate * 0.5)

        pass_start_mel = random.random() * (max_filter - self.span)
        pass_end_mel = pass_start_mel + self.span

        pass_start_freq = mel_to_freq(pass_start_mel)
        pass_end_freq = mel_to_freq(pass_end_mel)

        print('Filtering', pass_start_freq, pass_end_freq)

        return butter_bandstop_filter(
            data, pass_start_freq, pass_end_freq,
            self.sampling_rate, order=81)


def display_layered(original, augmented):
    """ Displays two wav files on top of each other """
    librosa.display.waveplot(augmented, sr=sample_rate, alpha=0.5)
    librosa.display.waveplot(original, sr=sample_rate, color='r', alpha=0.25)

    plt.tight_layout()
    plt.show()


def display_spectrogram(data):
    plt.figure(figsize=(10, 4))
    spectrogram = melspectrogram(y=data, sr=sample_rate, n_mels=128, fmax=8000)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel',
                             sr=sample_rate, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()
