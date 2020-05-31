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
from genre.util import ensure_download_exists, split_list, get_project_root

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


def compile_to_llds(file_system, num_augments, augments=None,
                    train_percentage=0.75):
    """
    Compiles a directory of wav files into low level descriptors (LLDs) using
    openSMILE.

    :param file_system: An object containing the information about the host
                        file system
    :type file_system: genre.FileSystemConfig
    :param num_augments: The number of augmented files to create per training
                         file
    :type num_augments: int
    :param augments: A list of augments to use. Should correspond to the keys
                     in `AUGMENTORS`. If None, will use all available augments.
    :type augments: list
    :param train_percentage: The percentage of the wav files that should be
                             used as training data
    :type train_percentage: float
    """
    augmentor = augmentor_factory(augments)

    with TemporaryDirectory() as tmp_dir_name:
        tmp = Path(tmp_dir_name)
        train_paths, aug_paths, test_paths = prepare_lld_paths(
            file_system.wavs, tmp, num_augments, train_percentage, augmentor
        )

        for path in train_paths:
            compile_file_to_llds_and_labels(path, file_system.lld_train_file,
                                            file_system.labels_train_file)

        for path in aug_paths:
            compile_file_to_llds_and_labels(path, file_system.lld_train_file,
                                            file_system.labels_train_file)

        for path in test_paths:
            compile_file_to_llds_and_labels(path, file_system.lld_test_file,
                                            file_system.labels_test_file)


def prepare_lld_paths(source, tmp, num_augments, train_percentage, augmentor):
    """
    Prepares the arguments for LLD creation functions

    :param source: A directory containing WAV files
    :type source: Path
    :param tmp: A directory to store augmented WAV files before their
                conversion to LLDs
    :type tmp: Path
    :param num_augments: The number of augmented files to create per training
                         file
    :type num_augments: int
    :param train_percentage: The percentage of the wav files that should be
                             used as training data
    :type train_percentage: float
    :param augmentor: An augmentor object
    :type augmentor: Augmenter
    :return: The paths for the base training samples, the augmented samples,
             and the test samples
    :rtype: (list, list, list)
    """
    sample_paths = list(source.iterdir())
    train_samples, test_samples = split_list(sample_paths, train_percentage)
    num_samples = len(sample_paths)
    index_iter = count(num_samples)

    random.shuffle(sample_paths)

    train_augment_paths = []

    for path in train_samples:
        if num_augments > 0:
            augment_paths = store_augments(path, augmentor, num_augments,
                                           index_iter, tmp)
            train_augment_paths += augment_paths

    return train_samples, train_augment_paths, test_samples


def store_augments(origin_path, augmentor, num_augments,
                   ident_generator, output_dir):
    """
    Stores augmented versions of a WAV file in the filesystem

    :param origin_path: the path to the original WAV file
    :type origin_path: Path
    :param augmentor: An augmentor object
    :type augmentor: Augmenter
    :param num_augments: The number of augmented files to create
    :type num_augments: int
    :param ident_generator: A generator object that creates unique identifiers
                            for the file name
    :type ident_generator: Iterable
    :param output_dir: The directory to store the augmented files
    :type output_dir: Path
    :return: A list of file paths for the augments that were created
    :rtype: list
    """
    audio_data, _ = librosa.load(origin_path)
    augments = augmentor.augment(audio_data, n=num_augments)

    if num_augments == 1:
        # nlpaug.Augmenter doesn't return a list if n=1
        augments = [augments]

    _, label = origin_path.stem.split('__')
    output_paths = []

    for augment in augments:
        identifier = next(ident_generator)
        output_path = output_dir / f'{identifier}__{label}.wav'
        output_paths.append(output_path)
        sf.write(output_path, augment, SAMPLING_RATE)

    return output_paths


def compile_to_bow(llds, labels, target, codebook, use_codebook=False,
                   memory='12G'):
    """
    Compiles a file of LLDS and their corresponding labels into a bag of words
    using openXBOW.

    :param llds: The path to the LLD file
    :type llds: Path
    :param labels: The path to the labels file
    :type labels: Path
    :param target: The location where the bag of words should be saved
    :type target: Path
    :param codebook: The location of the bag of words codebook
    :type codebook: Path
    :param use_codebook: If True, `codebook` should already exist and will be
                         used as a reference (i.e. will not be altered). If
                         False, `codebook` does not have to exist and will be
                         altered.
    :type use_codebook: bool, optional
    :param memory: The amount of memory to request for the JVM
    :type memory: str, optional
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

    :param keywords: The keywords corresponding to the augmentors that should
                     be used
    :type keywords: list or None
    :return: An augmentor pipeline that randomly uses the specified augmentors
    :rtype: Augmenter
    """
    if keywords is None:
        augmentors = list(AUGMENTORS.values())
    else:
        augmentors = [AUGMENTORS[key] for key in keywords]
    return Sometimes(augmentors)


def compile_file_to_llds_and_labels(path, lld_file, label_file):
    """
    :param path: The path to the original file
    :type path: Path
    :param lld_file: The path to the output LLD file
    :type lld_file: Path
    :param label_file: The path to the label_file
    :type label_file: Path
    """
    name, label = path.stem.split('__')
    record_label(name, label, label_file)
    extract_llds(path, lld_file, name)


def record_label(name, label, label_path):
    """
    Records the label of the file in another file

    :param name: The name (identifier) of the file
    :type name: any
    :param label: The classification label of the file
    :type label: str
    :param label_path: The file to store the labels
    :type label_path: Path
    """
    with open(label_path, 'a') as label_file:
        label_file.write(f'{name};{label}\n')


def extract_llds(source, dest, name):
    """
    Extracts the low-level descriptors (LLDs) from a WAV file

    :param source: The source WAV file
    :type source: Path
    :param dest: The output LLDs
    :type dest: Path
    :param name: The instance name of the file
    :type name: str

    :raises RuntimeError: The openSMILE executable is not available on the PATH
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
