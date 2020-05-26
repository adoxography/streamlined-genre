"""
genre.augment

Addon library to nlpaug; adds a frequency filter
"""
import math
import random

from nlpaug.augmenter.audio import AudioAugmenter
from nlpaug.util import Action
from scipy.signal import butter, sosfilt

# TODO: Extract to a constant; unify with the one in compile.py
SAMPLE_RATE = 22050


class BandpassAug(AudioAugmenter):
    """
    AudioAugmenter that masks frequency ranges
    """
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

        return butter_bandstop_filter(
            data, pass_start_freq, pass_end_freq,
            self.sampling_rate, order=81)


def freq_to_mel(freq):
    """ Converts a frequence in Hz to an approximation on the Mel spectrum """
    return 2595 * math.log10(1 + freq / 700)


def mel_to_freq(mel):
    """ Converts an approximation on the Mel spectrum to a frequency in Hz """
    return 700 * (10 ** (mel / 2595) - 1)


def butter_bandstop_filter(data, lowcut, highcut, sampling_rate, order=5):
    """
    See https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter/12233959#12233959  # noqa
    """
    sos = butter_bandstop(lowcut, highcut, sampling_rate, order=order)
    return sosfilt(sos, data)


def butter_bandstop(lowcut, highcut, sampling_rate, order=5):
    """
    Generates a Butterworth bandstop filter

    :param lowcut: The low end of the bandstop
    :param highcut: The high end of the bandstop
    :param sampling_rate: The sampling rate of the target WAV
    :param order: How sharp the filter is; higher values mean crisper corners.
                  Equivalent to Q in audio engineering.
    :return: An audio filter corresponding to the bandstop filter
    """
    nyquist_freq = 0.5 * sampling_rate
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    sos = butter(order, [low, high], analog=False,
                 btype='bandstop', output='sos')
    return sos
