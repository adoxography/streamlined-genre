"""
genre.services

Adapters to 3rd party libraries.

Contains:
    openXBOW
"""
import subprocess
from pathlib import Path
from typing import Tuple

from genre.util import ensure_download_exists, get_project_root

OPEN_XBOW_JAR = get_project_root() / 'lib' / 'openXBOW.jar'
OPEN_XBOW_URL = 'https://github.com/openXBOW/openXBOW/blob/master/openXBOW.jar?raw=true'  # noqa


def openxbow(input_data: Tuple[Path, Path], dest: Path, codebook: Path,
             **options):
    """
    Executes a call to openXBOW

    :param input_data: A tuple of the input LLD and label files
    :param dest: The location where the bag of words should be saved
    :param codebook: The location of the bag of words codebook

    :param options:
        * *use_codebook*
            If True, the codebook will be referenced. If False, it will be
            created. Defaults to False.
        * *append*
            If True, the bag of words will be appended to. Defaults to False.
        * *memory*
            If provided, will be passed to the JVM as the memory request.
    """
    ensure_download_exists(OPEN_XBOW_JAR, OPEN_XBOW_URL)

    llds, labels = input_data

    use_codebook = options.get('use_codebook', False)
    append = options.get('append', False)
    memory = options.get('memory')

    memory_arg = f'-Xmx{memory}' if memory else ''
    append_arg = '-append' if append else ''
    codebook_flag = '-b' if use_codebook else '-B'
    standardize = '-standardizeInput' if append or not use_codebook else ''

    args = [
        'java',
        memory_arg,
        '-jar', str(OPEN_XBOW_JAR),
        '-i', str(llds),
        '-o', str(dest),
        '-l', str(labels),
        codebook_flag, str(codebook),
        append_arg,
        standardize,
        '-log'
    ]

    subprocess.run(args, check=True)
