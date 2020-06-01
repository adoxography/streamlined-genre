"""
genre.compile

Contains logic for extracting LLD features from files
"""
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
from genre.util import split_list, get_project_root

OPENSMILE_CONFIG_DIR = get_project_root() / 'config' / 'openSMILE'
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

# Available augmentors, along with their string keys
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
    augmentor = augmentor_factory(augments)

    with TemporaryDirectory() as tmp_dir_name:
        tmp = Path(tmp_dir_name)
        train_paths, aug_paths, test_paths = prepare_lld_paths(
            file_system.wav_dir, tmp, num_augments, train_percentage, augmentor
        )

        train_output_paths = file_system.new_lld_label_training_pair()
        test_output_paths = (file_system.lld_test_file,
                             file_system.labels_test_file)

        for path in train_paths:
            compile_file_to_llds_and_labels(path, train_output_paths)

        for path in aug_paths:
            compile_file_to_llds_and_labels(path, train_output_paths)

        for path in test_paths:
            compile_file_to_llds_and_labels(path, test_output_paths)


def prepare_lld_paths(source: Path, tmp: Path, num_augments: int,
                      train_percentage: float,
                      augmentor: Augmenter) -> Tuple[List, List, List]:
    """
    Prepares the arguments for LLD creation functions

    :param source: A directory containing WAV files
    :param tmp: A directory to store augmented WAV files before their
                conversion to LLDs
    :param num_augments: The number of augmented files to create per training
                         file
    :param train_percentage: The percentage of the wav files that should be
                             used as training data
    :param augmentor: An augmentor object
    :return: The paths for the base training samples, the augmented samples,
             and the test samples
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


def store_augments(origin_path: Path, augmentor: Augmenter, num_augments: int,
                   ident_generator: Iterator, output_dir: Path) -> List[Path]:
    """
    Stores augmented versions of a WAV file in the filesystem

    :param origin_path: the path to the original WAV file
    :param augmentor: An augmentor object
    :param num_augments: The number of augmented files to create
    :param ident_generator: A generator object that creates unique identifiers
                            for the file name
    :param output_dir: The directory to store the augmented files
    :return: A list of file paths for the augments that were created
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


def augmentor_factory(keywords: Optional[List[str]] = None) -> Augmenter:
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


def compile_file_to_llds_and_labels(input_path: Path,
                                    target_paths: Tuple[Path, Path]) -> None:
    """
    :param input_path: The path to the original file
    :param target_paths: A tuple of the path to the output LLD file and the
                         path to the label file
    """
    lld_path, label_path = target_paths
    name, label = input_path.stem.split('__')

    record_label(name, label, label_path)
    opensmile(input_path, lld_path, name,
              config=COMPARE_CONFIG, options=COMPARE_OPTIONS)


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
