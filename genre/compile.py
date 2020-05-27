"""
genre.compile

Contains logic for extracting features from files:

 - From WAVs to LLDs using openSMILE
 - From LLDs to BOWs using openXBOW
"""
import random
import shutil
import subprocess
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa
import soundfile as sf
from nlpaug.flow import Sometimes
from nlpaug.augmenter.audio import MaskAug, VtlpAug, SpeedAug

from genre.augment import BandpassAug
from genre.util import ensure_download_exists, get_project_root

OPENSMILE_EXE = 'SMILExtract'

OPENSMILE_CONFIG = get_project_root() / 'config' / 'openSMILE' / 'ComParE_2016.conf'  # noqa

OPENSMILE_OPTIONS = [
    '-configfile', OPENSMILE_CONFIG,
    '-appendcsvlld', '1',
    '-timestampcsvlld', '1',
    '-headercsvlld', '1'
]

OPEN_XBOW_JAR = get_project_root() / 'lib' / 'openXBOW.jar'
OPEN_XBOW_URL = 'https://github.com/openXBOW/openXBOW/blob/master/openXBOW.jar?raw=true'  # noqa

# openXBOW requires this sampling rate
SAMPLING_RATE = 22050

# Available augmentors, along with their string keys
AUGMENTORS = {
    'time_mask': MaskAug(sampling_rate=SAMPLING_RATE, mask_with_noise=False,
                         coverage=.1),
    'freq_mask': BandpassAug(sampling_rate=SAMPLING_RATE),
    'warp': SpeedAug(),
    'vtlp': VtlpAug(sampling_rate=SAMPLING_RATE)
}


def compile_to_llds(source, llds_train, llds_test, labels_train, labels_test,
                    num_augments, augments=None):
    """
    Compiles a directory of wav files into low level descriptors (LLDs) using
    openSMILE.

    :param source: A directory containing WAV files. WAV files should be named
                   {filename}__{label}.wav.
    :param llds_train: The path to the LLD train file
    :param llds_test: The path to the LLD test file
    :param labels_train: The path to the label train file
    :param labels_test: The path to the label test file
    :param tmp: A directory to store augmented WAV files before their
                conversion to LLDs
    :param num_augments: The number of augmented files to create per training
                         file
    :param augments: A list of augments to use. Should correspond to the keys
                     in `AUGMENTORS`. If None, will use all available augments.
    """
    sample_paths = list(source.iterdir())
    num_samples = len(sample_paths)
    index_iter = count(num_samples)
    augmentor = augmentor_factory(augments)

    random.shuffle(sample_paths)

    # TODO: Find a way to parallelize openSMILE
    with TemporaryDirectory() as tmp_dir_name:
        tmp = Path(tmp_dir_name)
        for i, path in enumerate(sample_paths):
            if i < num_samples * 0.75:
                # TODO: extract the ratio of input to test files
                compile_file_to_llds_and_labels(path, llds_train, labels_train)
                for _ in range(num_augments):
                    # TODO: Parallelize augmentation
                    file_data, _ = librosa.load(path)
                    augment_path = augment(augmentor, path, tmp, file_data,
                                           index_iter)
                    compile_file_to_llds_and_labels(augment_path, llds_train,
                                                    labels_train)
            else:
                compile_file_to_llds_and_labels(path, llds_test, labels_test)


def compile_to_bow(llds, labels, target, codebook, use_codebook=False,
                   memory='12G'):
    """
    Compiles a file of LLDS and their corresponding labels into a bag of words
    using openXBOW.

    :param llds: The path to the LLD file
    :param labels: The path to the labels file
    :param target: The location where the bag of words should be saved
    :param codebook: The location of the bag of words codebook
    :param use_codebook: If True, `codebook` should already exist and will be
                         used as a reference (i.e. will not be altered). If
                         False, `codebook` does not have to exist and will be
                         altered.
    :param memory: The amount of memory to request for the JVM
    """
    ensure_download_exists(OPEN_XBOW_JAR, OPEN_XBOW_URL)

    codebook_flag = '-b' if use_codebook else '-B'

    openxbow_call = [
        'java',
        f'-Xmx{memory}',
        '-jar', OPEN_XBOW_JAR,
        '-i', llds,
        '-o', target,
        '-l', labels,
        codebook_flag, codebook,
        '' if use_codebook else '-standardizeInput',
        '-log'
    ]

    subprocess.run(openxbow_call, check=True)


def augmentor_factory(keywords=None):
    """
    Generates an augmentor based on a list of keywords

    If the `keywords` is None, all available augmentors will be used.
    """
    if keywords is None:
        augmentors = list(AUGMENTORS.values())
    else:
        augmentors = [AUGMENTORS[key] for key in keywords]
    return Sometimes(augmentors)


def compile_file_to_llds_and_labels(path, lld_file, label_file):
    """
    :param path: The path to the original file
    :lld_file: The path to the output LLD file
    :label_file: The path to the label_file
    """
    name, label = path.stem.split('__')
    record_label(name, label, label_file)
    extract_llds(path, lld_file, name)


def record_label(name, label, label_path):
    """
    Records the label of the file in another file

    :param name: The name (identifier) of the file
    :param label: The classification label of the file
    :param label_path: The file to store the labels
    """
    with open(label_path, 'a') as label_file:
        label_file.write(f'{name};{label}\n')


def extract_llds(source, dest, name):
    """
    Extracts the low-level descriptors (LLDs) from a WAV file

    :param source: The source WAV file
    :param dest: The output LLDs
    :param name: The instance name of the file
    """
    if not shutil.which(OPENSMILE_EXE):
        raise RuntimeError(f'{OPENSMILE_EXE} is not on the PATH')

    opensmile_call = [
        OPENSMILE_EXE,
        *OPENSMILE_OPTIONS,
        '-inputfile', source,
        '-lldcsvoutput', dest,
        '-instname', name
    ]
    subprocess.run(opensmile_call, check=True)


def augment(augmentor, path, dest_path, data, index_generator):
    """
    Creates an augmented version of a WAV file

    :param augmentor: An object that can augment WAV file data
    :param path: The path to the original WAV file
    :param dest_path: The path where the augmented file will be stored
    :param data: The audio data to augment
    :param index_generator: A generator for the augment's file name
    :return: The name of the augmented file
    """
    _, label = path.stem.split('__')
    filename = dest_path / f'{next(index_generator)}__{label}.wav'
    augmented = augmentor.augment(data)
    sf.write(path, augmented, SAMPLING_RATE)
    return filename
