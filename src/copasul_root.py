#!/usr/bin/env python3

# author: Uwe Reichel, Budapest, 2016

import os

# root directory for copasul scripts and config files
# to be adjusted by user.
# Platform-dependent directory separator (/ for Unix,
# \ for Windows) is set automatically.
#
#  example: copasul_*py files are stored in:
#    /homes/myHome/tools/copasul/
#
#  adjust input for os.path.join() as follows:
#    os.path.join('homes','myHome','tools','copasul')
#
def copa_root():
    r = os.path.abspath(os.sep)
    x = r + os.path.join('homes','reichelu','repos','repo_pl','src','copasul_py')
    return x

