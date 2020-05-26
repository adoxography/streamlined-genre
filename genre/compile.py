import random
import subprocess
from pathlib import Path
from itertools import count
from multiprocessing import Pool

import librosa
import soundfile as sf
from nlpaug.flow import Sometimes
from nlpaug.augmenter.audio import MaskAug, VtlpAug, SpeedAug

from genre.augment import BandpassAug

# TODO: make platform independent
AUGMENT_DIR = Path('/tmp/streamlined/augments')

# TODO: extract
EXE_OPENSMILE = Path('/opt/opensmile/bin/SMILExtract')

# TODO: extract
OPENSMILE_CONFIG = Path('config/openSMILE/ComParE_2016.conf')

OPENSMILE_OPTIONS = ' '.join([
    f'-configfile {OPENSMILE_CONFIG}',
    '-appendcsvlld 1',
    '-timestampcsvlld 1',
    '-headercsvlld 1'
])

# TODO: extract
OPEN_XBOW_JAR='/home2/gstill/genre/lib/openXBOW.jar'

SAMPLING_RATE = 22050

AUGMENTORS = {
    'time_mask': MaskAug(sampling_rate=SAMPLING_RATE, mask_with_noise=False,
                         coverage=.1),
    'freq_mask': BandpassAug(sampling_rate=SAMPLING_RATE),
    'warp': SpeedAug(),
    'vtlp': VtlpAug(sampling_rate=SAMPLING_RATE)
}


def compile_to_llds(source, llds_train, llds_test, labels_train, labels_test,
                    tmp, num_augments, augments=None):
    sample_paths = list(source.iterdir())
    num_samples = len(sample_paths)
    index_iter = count(num_samples)
    augmentor = augmentor_factory(augments)

    random.shuffle(sample_paths)
    file_q = []

    for i, path in enumerate(sample_paths):
        if i < num_samples * 0.75:
            file_q.append((path, llds_train, labels_train))
            for _ in range(num_augments):
                file_data, _ = librosa.load(path)
                augment_path = augment(augmentor, path, tmp, file_data,
                                       index_iter)
                file_q.append((augment_path, llds_train, labels_train))
        else:
            file_q.append((path, llds_test, labels_test))

    pool = Pool()
    pool.starmap(process_file, file_q)
    pool.close()

def compile_to_bow(llds, labels, target, codebook, use_codebook=False,
                   memory='12G'):
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

    process = subprocess.run(openxbow_call, capture_output=True)
    process.check_returncode()


def augmentor_factory(keywords=None):
    if keywords is None:
        augmentors = list(AUGMENTORS.values())
    else:
        augmentors = [AUGMENTORS[key] for key in keywords]
    return Sometimes(augmentors)


def process_file(path, lld_file, label_file):
    name, label = path.stem.split('__')
    record_label(name, label, label_file)
    extract_llds(path, lld_file, name)


def record_label(name, label, label_file):
    with open(label_file, 'a') as f:
        f.write(f'{name};{label}\n')


def extract_llds(source, dest, name):
    opensmile_call = [
        str(EXE_OPENSMILE),
        OPENSMILE_OPTIONS,
        '-inputfile', source,
        '-lldcsvoutput', dest,
        '-instname', name
    ]
    process = subprocess.run(opensmile_call, capture_output=True)
    process.check_returncode()


def augment(augmentor, path, dest_path, data, index_iter):
    name, label = path.stem.split('__')
    filename = dest_path / f'{next(index_iter)}__{label}.wav'
    augmented = augmentor.augment(data)
    sf.write(path, augmented, SAMPLING_RATE)
    return filename
