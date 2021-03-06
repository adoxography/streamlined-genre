"""
genre.extract

Handles extraction of data from ELAR directories
"""
import csv
import logging
import subprocess
import itertools
from typing import Any, Iterator, List, Tuple
from pathlib import Path

# Locactions of each piece of data within the ELAR manifest
TITLE_COL = 0
WAV_COL = 2
LABEL_COL = 4

logger = logging.getLogger('streamlined-genre')


def extract_from_elar_dirs(sources: List[Path], dest: Path) -> None:
    """
    Extracts all of the sound files from ELAR-format directories

    See `process_source` for a definition of ELAR-format.

    :param sources: A list of directories to extract from
    :param dest: A directory to store all of the extracted sound files
    """
    num_samples = len(list(dest.iterdir()))
    index_iter = itertools.count(num_samples)

    for source in sources:
        process_source(source, dest, index_iter)


def process_source(source: Path, dest: Path, index_iter: Iterator) -> None:
    """
    Extracts all of the sound files from an ELAR-format directory

    The directory should be named after the language in question, and should
    contain a file called `{Language name}_ELAR_Directory.csv`, and a directory
    called `Bundles`. The directory file should act as a reference to the files
    in `Bundles`.

    :param source: The ELAR-format directory
    :param dest: A directory to store all of the extracted sound files
    :param index_iter: A generator that produces a unique key
    """
    manifest_path = elar_manifest(source)

    logger.info('Processing %s', manifest_path)

    with open(manifest_path, encoding='utf-8-sig') as manifest:
        bundles_dir = source / 'Bundles'

        for row in csv.reader(manifest):
            title, wav, label = extract_row_data(row)
            dest_name = f'{generate_filename(index_iter, label)}.wav'

            bundle_dir = bundles_dir / title
            orig_wav_loc = bundle_dir / wav
            sph_loc = bundle_dir / f'{orig_wav_loc.stem}.sph'
            dest_wav_loc = dest / dest_name

            convert_to_wav(sph_loc, dest_wav_loc)

    logger.info('Finished processing %s', manifest_path)


def elar_manifest(path: Path) -> Path:
    """
    Retrieves the path to the ELAR manifest

    :param path: The path to the directory that should contain the manifest
    :return: The path to the ELAR manifest

    :raises RuntimeError: The manifest does not exist or is not correctly named
    """
    manifest_path = path / f'{path.stem}_ELAR_Directory.csv'

    if not manifest_path.exists():
        raise RuntimeError(f'{manifest_path} does not exist')

    return manifest_path


def extract_row_data(row: List[str]) -> Tuple[str, str, str]:
    """
    Extracts relevant data from an ELAR manifest

    :param row: The row to extract from
    :return: The title, sound file name, and label from the row
    """
    title = row[TITLE_COL]
    wav = row[WAV_COL]
    label = extract_label(row)
    return title, wav, label


def extract_label(row_data: List[str]) -> str:
    """
    Extracts the label from a row of an ELAR manifest

    :param row_data: The data from a single row of the manifest
    :return: The label for the row
    """
    labels = row_data[LABEL_COL].replace('\xa0', ' ').split(' - ')
    label = labels[0]
    return label.replace(' ', '-')


def generate_filename(generator: Iterator, label: Any) -> str:
    """
    Generates a unique filename that includes a unique string and the file's
    label

    :param generator: A generator that creates a unique name for the file
    :param label: The label for the file
    :return: A unique filename
    """
    return f'{next(generator)}__{label}'


def convert_to_wav(orig: Path, dest: Path) -> None:
    """
    Converts a sound file to a 16kbps WAV file

    :param orig: The path to the input to sound file
    :param dest: The path to where the WAV file should be saved
    """
    logger.info('Converting %s to %s', orig, dest)

    sox_call: List[str] = [
        'sox',
        str(orig),
        '-t', 'wav',
        '-b', '16',
        str(dest)
    ]
    subprocess.run(sox_call, check=False)
