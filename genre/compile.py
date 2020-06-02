"""
genre.compile

Contains logic for extracting LLD features from files
"""
import logging
import random
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, List, Optional, Tuple

import librosa  # type: ignore
import soundfile as sf  # type: ignore
from nlpaug import Augmenter  # type: ignore
from nlpaug.flow import Sometimes  # type: ignore
from nlpaug.augmenter.audio import MaskAug, VtlpAug, SpeedAug  # type: ignore

from genre.augment import BandpassAug
from genre.config import FileSystemConfig
from genre.services import opensmile, openxbow
from genre.util import split_list, get_config_dir

OPENSMILE_CONFIG_DIR = get_config_dir() / 'openSMILE'
COMPARE_CONFIG = OPENSMILE_CONFIG_DIR / 'ComParE_2016.conf'
COMPARE_OPTIONS = {
    'appendcsvlld': 1,
    'timestampcsvlld': 1,
    'headercsvlld': 1
}

# openXBOW requires this sampling rate
SAMPLING_RATE = 22050

TRAIN_MODE = 'train'
TEST_MODE = 'test'

# Available augmenters, along with their string keys
AUGMENTORS = {
    'time_mask': MaskAug(sampling_rate=SAMPLING_RATE, mask_with_noise=False,
                         coverage=.1),
    'freq_mask': BandpassAug(sampling_rate=SAMPLING_RATE),
    'warp': SpeedAug(),
    'vtlp': VtlpAug(sampling_rate=SAMPLING_RATE)
}


def compile_to_llds(file_system: FileSystemConfig, num_augments: int,
                    augments: List[str] = None,
                    train_percentage: float = 0.75) -> None:
    """
    Compiles a directory of wav files into low level descriptors (LLDs) using
    openSMILE.

    :param file_system: An object containing the information about the host
                        file system
    :param num_augments: The number of augmented files to create per training
                         file
    :param augments: A list of augments to use. Should correspond to the keys
                     in `AUGMENTORS`. If None, will use all available augments.
    :param train_percentage: The percentage of the wav files that should be
                             used as training data
    """
    augmenter = augmenter_factory(augments)

    with TemporaryDirectory() as tmp_dir_name:
        tmp = Path(tmp_dir_name)
        *all_train_paths, test_paths = prepare_lld_paths(
            file_system.wav_dir, tmp, num_augments, train_percentage, augmenter
        )

        test_output_paths = (file_system.lld_test_file,
                             file_system.labels_test_file)

        for paths in all_train_paths:
            output_paths = file_system.new_lld_label_training_pair()
            for path in paths:
                compile_file_to_llds_and_labels(path, output_paths)

        for path in test_paths:
            compile_file_to_llds_and_labels(path, test_output_paths)


def prepare_lld_paths(source: Path, tmp: Path, num_augments: int,
                      train_percentage: float,
                      augmenter: Augmenter) -> List[List[Path]]:
    """
    Prepares the arguments for LLD creation functions

    :param source: A directory containing WAV files
    :param tmp: A directory to store augmented WAV files before their
                conversion to LLDs
    :param num_augments: The number of augmented files to create per training
                         file
    :param train_percentage: The percentage of the wav files that should be
                             used as training data
    :param augmenter: An augmenter object
    :return: A list of lists of paths to WAV files for LLD creation. The last
             member is the test group.
    """
    sample_paths = [path for path in source.iterdir() if path.suffix == '.wav']
    random.shuffle(sample_paths)

    train_samples, test_samples = split_list(sample_paths, train_percentage)
    num_samples = len(sample_paths)
    index_iter = count(num_samples)

    train_augment_paths: List[List[Path]] = [[] for _ in train_samples]

    for path in train_samples:
        if num_augments > 0:
            augment_paths = store_augments(path, augmenter, num_augments,
                                           index_iter, tmp)

            for i, augment_path in enumerate(augment_paths):
                train_augment_paths[i].append(augment_path)

    return [train_samples, *train_augment_paths, test_samples]


def store_augments(origin_path: Path, augmenter: Augmenter, num_augments: int,
                   ident_generator: Iterator, output_dir: Path) -> List[Path]:
    """
    Stores augmented versions of a WAV file in the filesystem

    :param origin_path: the path to the original WAV file
    :param augmenter: An augmenter object
    :param num_augments: The number of augmented files to create
    :param ident_generator: A generator object that creates unique identifiers
                            for the file name
    :param output_dir: The directory to store the augmented files
    :return: A list of file paths for the augments that were created
    """
    audio_data, _ = librosa.load(origin_path)
    augments = augmenter.augment(audio_data, n=num_augments)

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


def augmenter_factory(keywords: Optional[List[str]] = None) -> Augmenter:
    """
    Generates an augmenter based on a list of keywords

    If the `keywords` is None, all available augmenters will be used.

    :param keywords: The keywords corresponding to the augmenters that should
                     be used
    :type keywords: list or None
    :return: An augmenter pipeline that randomly uses the specified augmenters
    :rtype: Augmenter
    """
    if keywords is None:
        augmenters = list(AUGMENTORS.values())
    else:
        augmenters = [AUGMENTORS[key] for key in keywords]
    return Sometimes(augmenters)


def compile_file_to_llds_and_labels(input_path: Path,
                                    target_paths: Tuple[Path, Path]) -> None:
    """
    :param input_path: The path to the original file
    :param target_paths: A tuple of the path to the output LLD file and the
                         path to the label file
    """
    lld_path, label_path = target_paths
    name, label = input_path.stem.split('__')

    logging.info('Compiling %s to LLDs...', input_path)

    record_label(name, label, label_path)
    opensmile(input_path, lld_path, name,
              config=COMPARE_CONFIG, options=COMPARE_OPTIONS)

    logging.info('%s compiled to LLDs', input_path)


def record_label(name: Any, label: Any, label_path: Path) -> None:
    """
    Records the label of the file in another file

    :param name: The name (identifier) of the file
    :param label: The classification label of the file
    :param label_path: The file to store the labels
    """
    with open(label_path, 'a') as label_file:
        label_file.write(f'{name};{label}\n')


def compile_to_bow(data_pairs: List[Tuple[Path, Path]], dest: Path,
                   codebook: Path, use_codebook: bool = False,
                   memory: Optional[str] = None) -> None:
    """
    Compiles LLDs and labels to bags of words

    :param data_pairs: A list of LLD and label files to compile
    :param dest: The path to store the bag of words
    :param codebook: The path to the codebook
    :param use_codebook: If False, create the codebook on the first run instead
                         of using it
    :param memory: The amount of memory to reserve for the JVM
    """
    for pair in data_pairs:
        openxbow(pair, dest, codebook,
                 use_codebook=use_codebook, memory=memory)

        # After the first openxbow run, the codebook will exist
        use_codebook = True
