#!/usr/bin/env python3
import os
import csv
import itertools


def extract_from_elar_dirs(sources, dest):
    num_samples = len(list(dest.iterdir()))
    index_iter = itertools.count(num_samples)

    for source in sources:
        process_source(source, dest, index_iter)


def process_source(source, dest, index_iter):
    with open(lang_manifest(source), encoding='utf-8-sig') as manifest:
        bundles_dir = source / 'Bundles'

        for row in csv.reader(manifest):
            title, wav, label = extract_row_data(row)
            dest_name = f'{generate_filename(index_iter, label)}.wav'

            bundle_dir = bundles_dir / title
            orig_wav_loc = bundle_dir / wav
            sph_loc = bundle_dir / f'{orig_wav_loc.stem}.sph'
            dest_wav_loc = dest / dest_name

            convert_to_wav(sph_loc, dest_wav_loc)


def lang_manifest(path):
    return path / f'{path.stem}_ELAR_Directory.csv'


def extract_row_data(row):
    title = row[0]
    wav = row[2]
    label = extract_label(row)
    return title, wav, label


def extract_label(row_data):
    labels = row_data[4].replace('\xa0', ' ').split(' - ')
    label = labels[0]
    return label.replace(' ', '-')


def generate_filename(iterator, label):
    return f'{next(iterator)}__{label}'


def convert_to_wav(orig, dest):
    os.system(f'sox "{orig}" -t wav -b 16 "{dest}"')
