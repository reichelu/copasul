# CoPaSul - Contour-based, parametric, and superpositional intonation stylization

## version 0.8.27

## author

* Uwe Reichel, Research Institute for Linguistics, Hungarian Academy of Sciences, Budapest

## language, dependencies

* python 3.*
* matplotlib >= 1.3.1
* numpy >= 1.8.2
* pandas >= 0.13.1
* scipy >= 0.13.3
* scikit learn >= 0.17.1
* tested only for Linux

## installation

* set up a virtual environment `venv_copasul`, activate it, and install requirements. For Linux this works e.g. as follows:

    ```
    $ cd /my/Path/to/copasul/
    $ virtualenv --python="/usr/bin/python3" --no-site-packages venv_copasul
    $ source venv_copasul/bin/activate
    (venv_copasul) $ pip install -r requirements.txt
    ```

## example call

* call of main script `src/copasul.py` with a configuration file

    ```
    $ cd /my/Path/to/copasul/
    $ source venv_copasul/bin/activate
    (venv_copasul) $ cd src/
    (venv_copasul) $ python copasul.py -c ../config/minex.json
    ```

* processes input files in `minex/input/`
* outputs feature tables to `minex/output/`

## further information, license, history

* manual: `doc/copasul_manual_latest.pdf`
* example configuration: `config/minex.json`
* commented configurations: `doc/copasul_commented_config.json.txt`
* license, disclaimer: `doc/LEGAL`
* history: `doc/history.txt`

## please cite

Reichel, U.D. (2017). CoPaSul Manual: Contour-based, parametric, and superpositional intonation stylization, arXiv:1612.04765.

