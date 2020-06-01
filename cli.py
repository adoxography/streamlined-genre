#!/usr/bin/env python3
"""
Command line interface into the streamlined-genre library
"""
import random
from pathlib import Path
from argparse import ArgumentParser

from genre.config import FileSystemConfig


parser = ArgumentParser(description='')

action_group = parser.add_argument_group(
    'actions', 'At least one of these must be present. Can be combined; e.g. '
               '-texc'
)
action_group.add_argument('-t',
                          action='store_true',
                          help='Transfer WAV files from ELAR directories. '
                               'Requires --wavs and at least one --source.')
action_group.add_argument('-e',
                          action='store_true',
                          help='Extracts LLDs from WAV files. Requires --wavs '
                               'and --compiled.')
action_group.add_argument('-x',
                          action='store_true',
                          help='Run openXBOW on the LLDS and labels in '
                               'OUTPUT_DIR. Requires --compiled.')
action_group.add_argument('-c',
                          action='store_true',
                          help='Classifies the BOWs. Requires --compiled.')

path_group = parser.add_argument_group(
    'path arguments',
    'Path locations for use by the various actions'
)
path_group.add_argument('-i', '--source',
                        metavar='SOURCE_DIR',
                        type=Path,
                        action='append',
                        help='An input ELAR directory (can specify as many '
                             'arguments as desired). Used by the -t action.')

path_group.add_argument('-w', '--wavs',
                        metavar='WAV_DIR',
                        type=Path,
                        help='The directory to store the intermediate wav '
                             'files. Used by the -t and -e actions.')
path_group.add_argument('-o', '--compiled',
                        metavar='OUTPUT_DIR',
                        type=Path,
                        help='The directory to store the compiled files. Used '
                        'by the -e, -x, and -c actions.')

extract_group = parser.add_argument_group('-e (extract) options')
extract_group.add_argument('-n', '--num-augments',
                           type=int,
                           default=5,
                           help='The number of extra augmented files to add '
                                'per training file. Defaults to 5.')
extract_group.add_argument('-a', '--augments',
                           action='append',
                           choices=['time_mask', 'freq_mask', 'warp', 'vtlp'],
                           help='The augments to randomly use. If this '
                                'argument is ommitted, all will be used.')
extract_group.add_argument('--train-percentage',
                           default=0.75,
                           type=float,
                           help='The percentage of the input files (expressed '
                                'as a number between 0 and 1) that should be '
                                'used as training data')

xbow_group = parser.add_argument_group('-x (xbow) options')
xbow_group.add_argument('-m', '--memory',
                        default='12G',
                        help='The amount of memory to reserve for the JVM. '
                             'Defaults to 12G.')


if __name__ == '__main__':
    random.seed(440)

    args = parser.parse_args()
    file_system = FileSystemConfig(sources=args.source, wavs=args.wavs,
                                   compiled=args.compiled)

    if not any([args.t, args.e, args.x, args.c]):
        parser.error('At least one action (-t, -e, -x, -c) must be specified. '
                     'Run this command with -h for more information on '
                     'available actions.')

    if args.t:
        if args.source is None:
            parser.error('-t requires at least one source')
        if args.wavs is None:
            parser.error('-t requires --wavs to be present')

        from genre.extract import extract_from_elar_dirs

        args.wavs.mkdir(exist_ok=True)
        extract_from_elar_dirs(args.source, args.wavs)

    if args.e:
        if args.wavs is None:
            parser.error('-e requires --wavs to be present')
        if args.compiled is None:
            parser.error('-e requires --compiled to be present')

        from genre.compile import compile_to_llds

        args.compiled.mkdir(exist_ok=True)

        compile_to_llds(file_system, args.num_augments, args.augments)

    if args.x:
        if args.compiled is None:
            parser.error('-x requires --compiled to be present')

        file_system.ensure_valid_lld_label_training_pairs()
        file_system.ensure_valid_lld_label_test_pairs()

        from genre.services import openxbow

        openxbow(file_system.lld_label_training_pairs[0],
                 file_system.xbow_train_file, file_system.codebook_file,
                 memory=args.memory)
        openxbow((file_system.lld_test_file,
                  file_system.labels_test_file),
                 file_system.xbow_test_file, file_system.codebook_file,
                 use_codebook=True, memory=args.memory)

    if args.c:
        if args.compiled is None:
            parser.error('-c requires --compiled to be present')
        if not file_system.xbow_train_file.exists():
            parser.error(f'-x requires {file_system.xbow_train_file} to exist')
        if not file_system.xbow_test_file.exists():
            parser.error(f'-x requires {file_system.xbow_test_file} to exist')

        from genre.classify import classify_bows

        train_acc, test_acc = classify_bows(file_system.xbow_train_file,
                                            file_system.xbow_test_file)

        print('Train accuracy:', train_acc)
        print('Test accuracy:', test_acc)
