"""
genre.extract

Handles extraction of data from ELAR directories
"""
import csv
import subprocess
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
    :type sources: list of Path
    :param dest: A directory to store all of the extracted sound files
    :type dest: Path
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

    :param source: The ELAR-format directory
    :type source: Path
    :param dest: A directory to store all of the extracted sound files
    :type dest: Path
    :param index_iter: A generator that produces a unique key
    :type index_iter: Iterable
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
    """
    Retrieves the path to the ELAR manifest

    :param path: The path to the directory that should contain the manifest
    :type path: Path
    :return: The path to the ELAR manifest
    :rtype: Path

    :raises RuntimeError: The manifest does not exist or is not correctly named
    """
    manifest_path = path / f'{path.stem}_ELAR_Directory.csv'

    if not manifest_path.exists():
        raise RuntimeError(f'{manifest_path} does not exist')

    return manifest_path


def extract_row_data(row):
    """
    Extracts relevant data from an ELAR manifest

    :param row: The row to extract from
    :type row: list
    :return: The title, sound file name, and label from the row
    :rtype: (str, str, str)
    """
    title = row[TITLE_COL]
    wav = row[WAV_COL]
    label = extract_label(row)
    return title, wav, label


def extract_label(row_data):
    """
    Extracts the label from a row of an ELAR manifest

    :param row_data: The data from a single row of the manifest
    :type row_data: list
    :return: The label for the row
    :rtype: str
    """
    labels = row_data[LABEL_COL].replace('\xa0', ' ').split(' - ')
    label = labels[0]
    return label.replace(' ', '-')


def generate_filename(generator, label):
    """
    Generates a unique filename that includes a unique string and the file's
    label

    :param generator: A generator that creates a unique name for the file
    :type generator: Iterable
    :param label: The label for the file
    :type label: any
    :return: A unique filename
    :rtype: str
    """
    return f'{next(generator)}__{label}'


def convert_to_wav(orig, dest):
    """
    Converts a sound file to a 16kbps WAV file

    :param orig: The path to the input to sound file
    :type orig: Path
    :param dest: The path to where the WAV file should be saved
    :type dest: Path
    """
    sox_call = [
        'sox',
        orig,
        '-t', 'wav',
        '-b', '16',
        dest
    ]
    subprocess.run(sox_call, check=False)
