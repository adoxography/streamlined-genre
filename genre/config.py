class FileSystemConfig:
    def __init__(self, sources, wavs, compiled):
        self.source_dirs = sources
        self.wav_dir = wavs
        self.compiled_dir = compiled

    def ensure_compiled_dir_exists(self):
        if self.compiled_dir is None:
            raise ValueError

        self.compiled_dir.mkdir(exist_ok=True)

    @property
    def lld_train_file(self):
        return self.compiled_dir / 'audio_llds_train.csv'

    @property
    def lld_test_file(self):
        return self.compiled_dir / 'audio_llds_test.csv'

    @property
    def labels_train_file(self):
        return self.compiled_dir / 'labels_train.csv'

    @property
    def labels_test_file(self):
        return self.compiled_dir / 'labels_train.csv'

    @property
    def xbow_train_file(self):
        return self.compiled_dir / 'xbow_train.arff'

    @property
    def xbow_test_file(self):
        return self.compiled_dir / 'xbow_test.arff'

    @property
    def codebook_file(self):
        return self.compiled_dir / 'codebook'
