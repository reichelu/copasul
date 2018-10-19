#!/usr/bin/env python3

# author: Uwe Reichel, Budapest, 2016

import argparse
import os
import sys
import mylib as myl
import numpy as np
import copasul_root as coro
import copasul_init as coin
import copasul_augment as coag
import copasul_preproc as copp
import copasul_styl as cost
import copasul_clst as cocl
import copasul_plot as copl
import copasul_export as coex
import os.path as op
import re

##### wrapper ###############################################
# wrapper aroung all copasul analysis steps
# IN:
#   args dict
#       ['config']: string myConfigFile.json
#                   or dict of config file content
#       ['opt']: config dict for embedded calls
#       ['copa']: copa dict, facultatively. Will be read from file
#                 if not provided
# OUT:
#   copa dict; see documentation
def copasul(args={}):

    ##### config input, copa init ###########################
    if 'config' not in args:
        # predefined config file
        #myCwd = os.getcwd()
        myCwd = coro.copa_root()
        args['config'] = os.path.join(myCwd,'config','config.json')
        if not os.path.isfile(args['config']):
            sys.exit("no predefined config file {}. Specify your own.".format(f_config))
    opt = coin.copa_opt_init(args['config'])

    # generate new copa dict?
    # or load it/take it from input args
    if opt['navigate']['from_scratch']:
        copa = coin.copa_init(opt)
    else:
        if ('copa' in args and 
            (type(copa) is dict) and
            len(copa.keys())>0):
            copa = args['copa']
        else:
            copa = copa_load(opt)
        # replace copa-contained config?
        if opt['navigate']['overwrite_config']:
            copa['config'] = opt
        else:
            opt = copa['config']

    ## log file #############################################
    f_log = copa_log('open',copa)

    ##### augmenting input ##################################
    if opt['navigate']['do_augment']:
        print("augment ...")
        coag.aug_main(copa,f_log)

    ##### preprocessing #####################################
    if opt['navigate']['do_preproc']:
        print("preproc ...")
        copa = copp.pp_main(copa,f_log)
        copa_save(copa)

    ##### diagnosis #########################################
    if opt['navigate']['do_diagnosis']:
        print("diagnosis ...")
        diagnosis_err = copp.diagnosis(copa,f_log)

    ##### stylization #######################################
    # global segments
    if opt['navigate']['do_styl_glob']:
       # or opt['navigate']['do_styl_loc'] or
       # opt['navigate']['do_styl_loc_ext']):
        print("styl_glob ...")
        copa = cost.styl_glob(copa,f_log)
        copa_save(copa)
    
    # local segments
    if opt['navigate']['do_styl_loc']:
        print("styl_loc ...")
        copa = cost.styl_loc(copa,f_log)
        copa_save(copa)

    # extended local feature set
    if opt['navigate']['do_styl_loc_ext']:
        print("styl_loc_ext ...")
        copa = cost.styl_loc_ext(copa,f_log)
        copa_save(copa)

    # boundary signals
    if (opt['navigate']['do_styl_bnd'] or
        opt['navigate']['do_styl_bnd_win'] or
        opt['navigate']['do_styl_bnd_trend']):
        print("styl_bnd ...")
        copa = cost.styl_bnd(copa,f_log)
        copa_save(copa)

    # general features: f0, energy
    if opt['navigate']['do_styl_gnl_f0']:
        print("styl_gnl_f0 ...")
        copa = cost.styl_gnl(copa,'f0',f_log)
        copa_save(copa)
    if opt['navigate']['do_styl_gnl_en']:
        print("styl_gnl_en ...")
        copa = cost.styl_gnl(copa,'en',f_log)
        copa_save(copa)

    # speech rhythm: f0, energy
    if opt['navigate']['do_styl_rhy_f0']:
        print("styl_rhy_f0 ...")
        copa = cost.styl_rhy(copa,'f0',f_log)
        copa_save(copa)
    if opt['navigate']['do_styl_rhy_en']:
        print("styl_rhy_en ...")
        copa = cost.styl_rhy(copa,'en',f_log)
        copa_save(copa)

    # voice quality
    if opt['navigate']['do_styl_voice']:
        print("styl_voice ...")
        copa = cost.styl_voice(copa,f_log)
        copa_save(copa)

    ##### clustering ########################################
    # global segments
    if opt['navigate']['do_clst_glob']:
        print("clst_glob ...")
        copa = cocl.clst_main(copa,'glob',f_log)
        copa_save(copa)
        
    # local segments
    if opt['navigate']['do_clst_loc']:
        print("clst_loc ...")
        copa = cocl.clst_main(copa,'loc',f_log)
        copa_save(copa)

    ##### export ############################################
    if opt['navigate']['do_export']:
        print("export ...")
        #copp.pp_grp_wrapper(copa) #!
        #copa_save(copa)
        coex.export_main(copa)

    ##### plot ##############################################
    if opt['navigate']['do_plot']:
        copa_plots(copa,opt)

    ##### log end ###########################################
    did_log = copa_log('val',copa,f_log)

    return copa


#### plots ##################################################
def copa_plots(copa,opt):
    ## browse through copa
    copl.plot_browse(copa)

    ## groupings
    # types: 'glob'|'loc'...
    for x in opt['plot']['grp']['type'].keys():
        # sets: 'acc'|'decl'...
        for y in opt['plot']['grp']['type'][x].keys():
            if opt['plot']['grp']['type'][x][y]:
                copl.plot_main({'call':'grp','state':'final','type':x,
                                'set':y,'fit':copa},opt)

##### log file #############################################

# add copa['config']['fsys']['log']['f']: log file name
# appends init row to log file
# IN:
#   task 'open'|'val' - init logfile, write validation metrics into logfile
#   copa (can be dummy container with 'config' key only {'config':opt})
#   f_log (for task 'val'
# OUT:
#   if task=='val': True
#   if task=='open': f_log, logFileHandle
def copa_log(task,copa,f_log=''):
    if task=='open':
        ff = os.path.join(copa['config']['fsys']['export']['dir'],
                          "{}.log.txt".format(copa['config']['fsys']['export']['stm']))
        copa['config']['fsys']['log'] = {'f':ff}
        f_log = open(copa['config']['fsys']['log']['f'],'a')
        f_log.write("\n## {}\n".format(myl.isotime()))
        return f_log
    elif task=='val':
        vv = copa['val']
        f_log.write("# validation\n")
        for x in sorted(vv.keys()):
            for y in sorted(vv[x].keys()):
                for z in sorted(vv[x][y].keys()):
                    f_log.write("{}.{}.{}: {}\n".format(x,y,z,vv[x][y][z]))
        f_log.close()
        return True



##### save/load #########################################
# saves copa into pickle file according to copa['config']['fsys']['export']
# IN:
#   copa
#   infx <''> to further specify file name
def copa_save(copa,infx=''):
    if len(infx)>0:
        f = "{}.{}.pickle".format(os.path.join(copa['config']['fsys']['export']['dir'],copa['config']['fsys']['export']['stm']),infx)
    else:
        f = "{}.pickle".format(os.path.join(copa['config']['fsys']['export']['dir'],copa['config']['fsys']['export']['stm']))

    myl.output_wrapper(copa,f,'pickle')

# loads copa from pickle file
# IN:
#   opt (copa['config'])
#   infx <''> to further specify file name
# OUT:
#   copa
def copa_load(opt,infx=''):
    if len(infx)>0:
        f = "{}.{}.pickle".format(os.path.join(opt['fsys']['export']['dir'],opt['fsys']['export']['stm']),infx)
    else:
        f = "{}.pickle".format(os.path.join(opt['fsys']['export']['dir'],opt['fsys']['export']['stm']))
    return myl.input_wrapper(f,'pickle')

##### Call ##############################################

# for command line calls
# > copasul.py -c myConfigs.json
if __name__ == "__main__":
    #myCwd = os.getcwd()
    myCwd = coro.copa_root()
    ##### command line input ####################################
    parser = argparse.ArgumentParser(description="copasul.py -- Intonation analysis tool version 0.8.12")
    parser.add_argument('-c','--config', help='myConfigFile.json', required=True)
    args = vars(parser.parse_args())
    copa = copasul(args)
