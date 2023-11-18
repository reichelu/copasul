import argparse
import json
import os
import sys

# Example script how to call copasul with a config json file.
# If omitted a default config file is read.
# Call:
# $ python call_copasul.py [-c MYCONFIGFILE.json]

# customize for own usage:
# case 1) copasul cloned from github project
#         calling script is in PROJECT_DIR/scripts/
#         (PROJECT_DIR: absolute path of cloned github project)
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

# case 2) copasul cloned from github project
#         calling script is in some other directory:
# PROJECT_DIR = ...
# sys.path.append(PROJECT_DIR)

# case 3) copasul package has been pip-installed
#     remove the sys.path.append() line

from copasul import copasul


def call_copasul(config: str):
    
    # init Copasul() object
    fex = copasul.Copasul()

    # process data specified in f_config
    copa = fex.process(config=config)

    # access output, e.g. local contour feature set
    print("local contour features:")
    print(copa["export"]["loc"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="call copasul from command line")
    cwd = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.normpath(f"{cwd}/../tests/minex/config/minex.json")
    parser.add_argument('-c', '--config', help='name of copasul config file',
                        type=str, required=False, default=default_config)
    kwargs = vars(parser.parse_args())
    call_copasul(**kwargs)

