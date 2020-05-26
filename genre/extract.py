"""
genre.extract

Handles extraction of data from ELAR directories
"""
import os
import csv
import itertools

# Locactions of each piece of data within the ELAR manifest
TITLE_COL = 0
WAV_COL = 2
LABEL_COL = 4


def extract_from_elar_dirs(sources, dest):
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


def process_source(source, dest, index_iter):
    """
    Extracts all of the sound files from an ELAR-format directory

    The directory should be named after the language in question, and should
    contain a file called `{Language name}_ELAR_Directory.csv`, and a directory
    called `Bundles`. The directory file should act as a reference to the files
    in `Bundles`.

    :param sources: The ELAR-format directory
    :param dest: A directory to store all of the extracted sound files
    :param index_iter: A generator that produces a unique key
    """
    with open(elar_manifest(source), encoding='utf-8-sig') as manifest:
        bundles_dir = source / 'Bundles'

        for row in csv.reader(manifest):
            title, wav, label = extract_row_data(row)
            dest_name = f'{generate_filename(index_iter, label)}.wav'

            bundle_dir = bundles_dir / title
            orig_wav_loc = bundle_dir / wav
            sph_loc = bundle_dir / f'{orig_wav_loc.stem}.sph'
            dest_wav_loc = dest / dest_name

            convert_to_wav(sph_loc, dest_wav_loc)


def elar_manifest(path):
    """ Retrieves the path to the ELAR manifest """
    # TODO: Throw an error if the manifest doesn't exist
    return path / f'{path.stem}_ELAR_Directory.csv'


def extract_row_data(row):
    """ Extracts relevant data from an ELAR manifest """
    title = row[TITLE_COL]
    wav = row[WAV_COL]
    label = extract_label(row)
    return title, wav, label


def extract_label(row_data):
    """ Extracts the label from a row of an ELAR manifest """
    labels = row_data[LABEL_COL].replace('\xa0', ' ').split(' - ')
    label = labels[0]
    return label.replace(' ', '-')


def generate_filename(generator, label):
    """
    Generates a unique filename that includes a unique string and the file's
    label
    """
    return f'{next(generator)}__{label}'


def convert_to_wav(orig, dest):
    """
    Converts a sound file to a 16kbps WAV file

    :param orig: The path to the input to sound file
    :param dest: The path to where the WAV file should be saved
    """
    # TODO: Use subprocess.run instead
    os.system(f'sox "{orig}" -t wav -b 16 "{dest}"')
