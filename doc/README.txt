############################################################################
## CoPaSul -                                                             ###
## Contour-based, parametric, and superpositional intonation stylization ###
############################ä###############################################

Author: Uwe Reichel, Budapest, 2017
	Research Institute for Linguistics
	Hungarian Academy of Sciences
        uwe.reichel@nytud.mta.hu

# Documentation:
doc/copasul_manual.pdf
https://arxiv.org/abs/1612.04765

# Language:
Python 3

# Package Dependencies:
matplotlib 1.3.1
numpy 1.8.2
pandas 0.13.1
scipy 0.13.3
scikit learn 0.17.1

So far the software is tested only for Linux!

# Code:
src/*py    for prosody analyses
    *praat for f0 extraction

# Shell call:
> copasul.py -c myConfigFile.json

# Python environment call:
>>> import copasul as copa
>>> myCopa = copa.copasul({'config':myConfig})

# Example configuration file
> config/example.json

# License, Disclaimer:
doc/LEGAL

# History
doc/history.txt

# please cite:
if you use the toolkit for publications

@Manual{copasulManual,
  title = 	 {CoPaSul manual},
  OPTkey = 	 {},
  author = 	 {Reichel, U.D.},
  organization = {RIL, HAS},
  address = 	 {Budapest, Hungary},
  OPTedition = 	 {},
  OPTmonth = 	 {},
  year = 	 {2016},
  note = 	 {arXiv:1612.04765},
  OPTannote = 	 {}
}

