# streamlined-genre

## Installation

0. Ensure that [`python`](https://python.org) (>=3.6), Java, and [`openSMILE`](https://www.audeering.com/opensmile/) are installed. `python3`, `java`, and `SMILExtract` should be available on the `PATH`.

1. Ensure that `pip`, `wheel`, and `setuptools` are installed and up-to-date

```pip install --upgrade pip wheel setuptools```

2. Clone the repository and `cd` into it

```git clone https://github.com/adoxography/streamlined-genre && cd streamlined-genre```

3. Install the `streamlined-genre` package

```pip install .```

If you're developing `streamlined-genre`, you might want to install it with the optional dev dependencies instead:

```pip install -e ".[dev]"```


## Usage

### Data preparation

#### ELAR
`streamlined-genre` is designed to work with directories generated from [Endangered Languages Archive](https://elar.soas.ac.uk) (ELAR) content. The directories should be constructed as follows:

```
{LANGUAGE}
├── Bundles
│   └── {RECORDING_TITLE}
│       └── {AUDIO_FILE}
└── {LANGUAGE}_ELAR_Directory.csv
```

`{LANGUAGE}_ELAR_Directory.csv` should be structured as follows, where `{LABEL}` is the instance's genre label. Audio files may be in `wav` or `sph` format.

```csv
...
{RECORDING_TITLE},...,{AUDIO_FILE},...{LABEL}
...
```

#### Non-ELAR
`streamlined-genre` can also work with arbitrary `wav` files. The files should be in the same folder and be named `{IDENTIFIER}__{LABEL}.wav` (where `{IDENTIFIER}` is unique across all samples).

### Command line executable

Once installed, the executable `streamlined-genre` is available on the command line. It requires at least one action flag:

Flag | Action                                      | Requires
---- | ------------------------------------------- | ------------------------------------
`-t` | Transfers audio files from ELAR directories | `--wavs` and at least one `--source`
`-e` | Extracts LLDs and labels from wav files     | `--wavs` and `--compiled`
`-x` | Compiles bags of words from LLD files       | `--compiled`
`-c` | Classifies bags of words                    | `--compiled`

Action flags may be combined; e.g. `-texc`.

For a full description of command line arguments, run `streamlined-genre --help`.


## License

This project is licensed [GPL v.3](/LICENSE). It incorporates the following third party packages, which have their own licenses:

- [openXBOW](https://github.com/openXBOW/openXBOW): [LICENSE](https://github.com/openXBOW/openXBOW/blob/master/LICENSE.txt)
- [openSMILE](https://github.com/naxingyu/opensmile): [LICENSE](/config/openSMILE/LICENSE)
