"""
genre.services

Adapters to 3rd party libraries.

Contains:
 - openSMILE
 - openXBOW
"""
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from genre.util import ensure_download_exists, flatten, get_lib_dir

OPENSMILE_EXE = 'SMILExtract'
OPEN_XBOW_JAR = get_lib_dir() / 'openXBOW.jar'
OPEN_XBOW_URL = 'https://github.com/openXBOW/openXBOW/blob/master/openXBOW.jar?raw=true'  # noqa


def opensmile(source: Path, dest: Path, name: str,
              config: Optional[Path] = None,
              options: Optional[dict] = None) -> None:
    """
    Extracts the low-level descriptors (LLDs) from a WAV file

    :param source: The source WAV file
    :param dest: The output LLDs
    :param name: The instance name of the file
    :param config: The path to the openSMILE config to use.
    :param options: Extra flags to pass to SMILExtract. Should not have the
                    preceding hyphen.

    :raises RuntimeError: The openSMILE executable is not available on the PATH
    """
    if not shutil.which(OPENSMILE_EXE):
        raise RuntimeError(f'{OPENSMILE_EXE} is not on the PATH')

    config_args = [] if config is None else ['-configfile', str(config)]
    options = options or {}
    option_args = flatten([f'-{key}', str(value)]
                          for key, value in options.items())

    args = [
        str(OPENSMILE_EXE),
        *config_args,
        *option_args,
        '-appendLogfile',
        '-noconsoleoutput',
        '-inputfile', str(source),
        '-lldcsvoutput', str(dest),
        '-instname', name
    ]

    subprocess.run(args, check=True)


def openxbow(input_data: Tuple[Path, Path], dest: Path, codebook: Path,
             **options) -> None:
    """
    Executes a call to openXBOW

    :param input_data: A tuple of the input LLD and label files
    :param dest: The location where the bag of words should be saved
    :param codebook: The location of the bag of words codebook
    :param options:
        - *use_codebook*
            If True, the codebook will be referenced. If False, it will be
            created. Defaults to False.
        - *append*
            If True, the bag of words will be appended to. Defaults to False.
        - *memory*
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
        standardize
    ]

    subprocess.run(args, check=True)
