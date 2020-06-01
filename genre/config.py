"""
genre.config

Defines configuration objects used by the project
"""
from pathlib import Path
from typing import List, Tuple

from genre.util import first


class FileSystemConfig:
    """
    Configuration object for the file system
    """
    def __init__(self, sources: List[Path], wavs: Path, compiled: Path):
        """
        Initializes the object

        :param sources: Paths to the ELAR directories
        :param wavs: Path to where the WAV files should be stored
        :param compiled: Path to where all processed files should be stored
        """
        self.source_dirs = sources
        self.wav_dir = wavs
        self.compiled_dir = compiled

    def ensure_compiled_dir_exists(self) -> None:
        """
        Ensures that the `compiled_dir` directory exists by creating it if it
        does not already exist

        :raises ValueError: `compiled_dir` is None
        """
        if self.compiled_dir is None:
            raise ValueError

        self.compiled_dir.mkdir(exist_ok=True)

    def new_lld_label_training_pair(self) -> Tuple[Path, Path]:
        """
        :return: paths to new LLD and label training files
        """
        num_pairs = len(self.lld_label_training_pairs)
        llds = self.compiled_dir / f'audio_llds_train_{num_pairs}.csv'
        labels = self.compiled_dir / f'labels_train_{num_pairs}.csv'
        return llds, labels

    @property
    def lld_label_training_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Collects the LLD and label files into associated pairs

        :return: The list of LLD and label pairs
        """
        lld_files = self.lld_train_files
        label_files = self.labels_train_files
        pairs = []

        for lld_file in lld_files:
            fname = lld_file.stem
            ident = fname.split('_')[-1]

            label_file = first(
                label_files,
                lambda path: str(path).split('_')[-1] == ident  # noqa pylint: disable=cell-var-from-loop
            )
            pairs.append((lld_file, label_file))

        return pairs

    def ensure_valid_lld_label_training_pairs(self) -> None:
        """
        Ensures that the LLD and label training files are valid

        :raises ValueError: The files are not valid
        """
        lld_files = self.lld_train_files
        label_files = self.labels_train_files
        pairs = self.lld_label_training_pairs

        num_llds = len(lld_files)
        num_labels = len(label_files)
        num_pairs = len(pairs)
        num_unique_llds = len(set(lld_files))
        num_unique_labels = len(set(label_files))

        if not pairs:
            raise ValueError('There are no LLD and/or label files')

        if num_pairs != num_llds \
                or num_pairs != num_labels \
                or num_unique_llds != num_llds \
                or num_unique_labels != num_labels:
            raise ValueError('LLD and label file mismatch')

    def ensure_valid_lld_label_test_pairs(self) -> None:
        """
        Ensures that the LLD and label test files are valid

        :raises ValueError: The files are not valid
        """
        if not self.lld_test_file.exists() or self.labels_test_file.exists():
            raise ValueError('There are no LLD and/or label files')

    @property
    def lld_train_files(self) -> List[Path]:
        """ The locations of the training audio LLDs files """
        return [path for path in self.compiled_dir.iterdir() if
                'audio_llds_train' in str(path)]

    @property
    def labels_train_files(self) -> List[Path]:
        """ The locations of the train label files """
        return [path for path in self.compiled_dir.iterdir() if
                'labels_train' in str(path)]

    @property
    def lld_test_file(self) -> Path:
        """ The location of the test audio LLDs file """
        return self.compiled_dir / 'audio_llds_test.csv'

    @property
    def labels_test_file(self) -> Path:
        """ The location of the test label file """
        return self.compiled_dir / 'labels_train.csv'

    @property
    def xbow_train_file(self) -> Path:
        """ The location of the train BOW file """
        return self.compiled_dir / 'xbow_train.arff'

    @property
    def xbow_test_file(self) -> Path:
        """ The location of the test BOW file """
        return self.compiled_dir / 'xbow_test.arff'

    @property
    def codebook_file(self) -> Path:
        """ The location of the codebook """
        return self.compiled_dir / 'codebook'
