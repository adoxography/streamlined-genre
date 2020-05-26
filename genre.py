#!/usr/bin/env python3
"""
Command line interface into the streamlined-genre library
"""
import random
import shutil
from pathlib import Path
from argparse import ArgumentParser

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

xbow_group = parser.add_argument_group('-x (xbow) options')
xbow_group.add_argument('-m', '--memory',
                        default='12G',
                        help='The amount of memory to reserve for the JVM. '
                             'Defaults to 12G.')


if __name__ == '__main__':
    random.seed(440)

    args = parser.parse_args()

    if not any([args.t, args.e, args.x, args.c]):
        parser.error('At least one action (-t, -e, -x, -c) must be specified. '
                     'Run this command with -h for more information on '
                     'available actions.')

    if args.compiled:
        # Establish the path names for files in the `compiled` directory
        llds_train = args.compiled / 'audio_llds_train.csv'
        llds_test = args.compiled / 'audio_llds_test.csv'
        labels_train = args.compiled / 'labels_train.csv'
        labels_test = args.compiled / 'labels_test.csv'
        xbow_train = args.compiled / 'xbow_train.arff'
        xbow_test = args.compiled / 'xbow_test.arff'
        codebook = args.compiled / 'codebook'

    if args.t:
        if len(args.source) == 0:
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

        compile_to_llds(args.wavs, llds_train, llds_test, labels_train,
                        labels_test, args.num_augments,
                        augments=args.augments)

    if args.x:
        if args.compiled is None:
            parser.error('-x requires --compiled to be present')
        if not llds_train.exists():
            parser.error(f'-x requires {llds_train} to exist')
        if not llds_test.exists():
            parser.error(f'-x requires {llds_test} to exist')
        if not labels_train.exists():
            parser.error(f'-x requires {labels_train} to exist')
        if not labels_test.exists():
            parser.error(f'-x requires {labels_test} to exist')

        from genre.compile import compile_to_bow

        compile_to_bow(llds_train, labels_train, xbow_train, codebook,
                       memory=args.memory)
        compile_to_bow(llds_test, labels_test, xbow_test, codebook,
                       use_codebook=True, memory=args.memory)

    if args.c:
        if args.compiled is None:
            parser.error('-c requires --compiled to be present')
        if not xbow_train.exists():
            parser.error(f'-x requires {xbow_train} to exist')
        if not xbow_test.exists():
            parser.error(f'-x requires {xbow_test} to exist')

        from genre.classify import classify_bows

        classify_bows(xbow_train, xbow_test)
