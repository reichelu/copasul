# CoPaSul - Contour-based, parametric, and superpositional intonation stylization

* code for prosody analyses (python 3)

## author

* Uwe Reichel, Budapest, 2017
* Research Institute for Linguistics, Hungarian Academy of Sciences
* uwe.reichel@nytud.mta.hu

## language, dependencies

* python 3
* matplotlib >= 1.3.1
* numpy >= 1.8.2
* pandas >= 0.13.1
* scipy >= 0.13.3
* scikit learn >= 0.17.1
* so far the software is tested only for Linux!

## installation

* set up a virtual environment "venv_copasul", activate it, and install requirements. For Linux this works as follows:

    ```
    $ virtualenv --python="/usr/bin/python3" --no-site-packages venv_copasul
    $ source venv_copasul/bin/activate
    (venv_copasul) $ pip install -r requirements.txt
    ```

## example call

    ```
    (venv_copasul) $ cd src
    (venv_copasul) $ python copasul.py -c ../config/minex.json
    ```

* processes input files in minex/input/
* outputs feature tables in minex/output/


## further information

* manual: doc/copasul_manual_latest.pdf
* example configuration: config/example.json

## License, Disclaimer

* doc/LEGAL

## history

* doc/history.txt

## please cite

Reichel, U.D. (2017). CoPaSul Manual: Contour-based, parametric, and superpositional intonation stylization, arXiv:1612.04765.

