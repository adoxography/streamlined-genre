"""
genre.config

Defines configuration objects used by the project
"""
from pathlib import Path
from typing import List


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

    @property
    def lld_train_file(self) -> Path:
        """ The location of the training audio LLDs file """
        return self.compiled_dir / 'audio_llds_train.csv'

    @property
    def lld_test_file(self) -> Path:
        """ The location of the test audio LLDs file """
        return self.compiled_dir / 'audio_llds_test.csv'

    @property
    def labels_train_file(self) -> Path:
        """ The location of the train label file """
        return self.compiled_dir / 'labels_train.csv'

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
