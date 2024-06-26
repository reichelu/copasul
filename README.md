# CoPaSul - Contour-based, parametric, and superpositional intonation stylization

## Author

* Uwe Reichel, Research Institute for Linguistics, Hungarian Academy of Sciences, Budapest

## Purpose: Prosodic Feature extraction

* Intonation
  * Global f0 register
  * Local f0 shapes
  * Prosodic boundaries
* Energy
* Rhythm
* Voice quality

## Dependencies

* Python: `>= 3.8`
* Packages: see `requirements.txt`
* tested for Linux and Python 3.8, and 3.10

## Installation

### From PyPI

* set up a virtual environment `venv_copasul`, activate it, and install copasul. For Linux this works e.g. as follows:

```bash
$ virtualenv --python="/usr/bin/python3" venv_copasul
$ source venv_copasul/bin/activate
(venv_copasul) $ pip install copasul
```

### From GitHub

* project URL: [https://github.com/reichelu/copasul](https://github.com/reichelu/copasul)
* set up a virtual environment `venv_copasul`, activate it, and install requirements. For Linux this works e.g. as follows:

```bash
$ git clone git@github.com:reichelu/copasul.git
$ cd copasul/
$ virtualenv --python="/usr/bin/python3" venv_copasul
$ source venv_copasul/bin/activate
(venv_copasul) $ pip install -r requirements.txt
```

## Usage

### Required files
* audio (wav), f0, pulse, and annotation files (Textgrid), see on GitHub: `tests/minex/input`
* a config file (json), see on GitHub `tests/minex/config/minex.json`
    * see on Github `docs/copasul_commented_config.json.txt` and the [article](https://arxiv.org/abs/1612.04765) for further details

### Extract f0 and pulse by means of Praat

* see [example script](https://github.com/reichelu/copasul/scripts/run_praat.py)

```python
# add this line if you use the code from GitHub
# sys.path.append(PROJECT_ROOT)

from copasul import praat_utils

# Praat pitch extractor
piex = praat_utils.PraatPitchExtractor(
    hopsize=0.01, minfreq=75, maxfreq=600)

# process mono files
piex.process_mono_files(
    input_dir=INPUT_AUDIO_DIR,
    output_dir=OUTPUT_PITCH_DIR,
    input_ext="wav",
    output_ext="f0"
)

# ... in order to process stereo files, use:
# piex.process_stereo_files()

# Praat pulse extractor
puex = praat_utils.PraatPulseExtractor(
    hopsize=0.01, minfreq=75, maxfreq=400)

# process mono files
puex.process_mono_files(
    input_dir=INPUT_AUDIO_DIR,
    output_dir=OUTPUT_PULSE_DIR,
    input_ext="wav",
    output_ext="pulse"
)

# ... in order to process stereo files, use:
# puex.process_stereo_files()
```

### Call Copasul from terminal (if cloned from GitHub)

* see on [example script](https://github.com/reichelu/copasul/scripts/run_copasul.py)
* `python scripts/run_copasul.py [-c myConfigFile.json]`
* `PROJECT_ROOT`: directory where GitHub project has been cloned

```bash
(venv_copasul) $ cd PROJECT_ROOT/scripts/
(venv_copasul) $ python run_copasul.py -c ../tests/minex/config/minex.json
```

* if you use `../tests/minex/config/minex.json` as config file, CoPaSul
    * processes the files in `../tests/minex/input/`, and
    * outputs feature tables to `../tests/minex/output/test.FEATURESET.{csv, R}`
    * outputs the CoPaSul output dict to `../tests/minex/output/test.pickle`
    * this output dict can be used for further (warm start) processing
* if you use your own config file, make sure, that all directories in `fsys:*:dir` are either absolute paths or relative paths relative to your config file

### Example integration into python code

```python
import json
import pickle
import sys

# add this line if you use the code from GitHub
# sys.path.append(PROJECT_ROOT)

from copasul import copasul

# feature extractor
fex = copasul.Copasul()

# processing based on config file
config_file = MYCONFIGFILE.json
copa = fex.process(config=config_file)

# processing based on config dict
with open(config_file, 'r') as h:
    config_dict = json.load(h)
copa = fex.process(config=config_dict)

# warm start: continue processing
#   - implicit loading from config
config_dict["navigate"]["from_scratch"] = False
config_dict["navigate"]["overwrite_config"] = True
#     ... change further navigation values to your needs
copa_further_processed = fex.process(config=config_dict)

#   - explicit loading from file
copa_file = MYCOPASULOUTPUT.pickle
config_dict["navigate"]["overwrite_config"] = True
#     ... change further navigation values to your needs
with open(copa_file, "rb") as h:
   copa = pickle.load(h)
copa_further_processed = fex.process(config=config_dict, copa=copa)
```

## Material

* [Manual](https://github.com/reichelu/copasul/blob/master/docs/copasul_manual_latest.pdf)
* [Example configuration](https://github.com/reichelu/copasul/blob/master/tests/minex/config/minex.json)
* [Commented configuration](https://github.com/reichelu/copasul/blob/master/docs/copasul_commented_config.json.txt)

## Reference

Reichel, U.D. (2017). [CoPaSul Manual: Contour-based, parametric, and superpositional intonation stylization](https://arxiv.org/abs/1612.04765), arXiv:1612.04765.
