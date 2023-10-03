# author: Uwe Reichel, Budapest, 2016

import argparse
import numpy as np
import os
import re
import sys

import copasul_augment as coag
import copasul_clst as cocl
import copasul_export as coex
import copasul_init as coin
import copasul_plot as copl
import copasul_preproc as copp
import copasul_styl as cost
import copasul_utils as utils


class Copasul(object):

    r"""prosodic analyses"""

    def __init__(self):

        super().__init__()

    def process(self, config: dict = None, copa: dict = None):

        ''' process files specified in config
        
        Args:
        config: (dict) of process specifications
        copa: (dict) previously generated copasul output dict for warm starts

        Returns:
        copa: (dict) copasul output
        '''
        
        if dict is None:
            sys.exit("process() requires config dictionary argument")

        if type(copa) is dict:
            config["navigate"]["from_scratch"] = False

        return copasul(config=config, copa=copa)


def copasul(config, copa=None):

    '''
    wrapper aroung all copasul analysis steps
    calls from terminal or from Copasul.process()
    
    Args:
    config: (str) name of config json file
    copa: (dict) copasul output for war start

    Returns:
      copa: (dict) copasul output
    '''

    # config #########################################
    opt = coin.copa_opt_init(config)
    
    # generate new copa dict?
    # or load it/take it from input args
    if opt['navigate']['from_scratch']:
        copa = coin.copa_init(opt)
    else:
        if copa is None or len(copa.keys()) == 0:
            copa = copa_load(opt)
            
        # replace copa-contained config?
        if opt['navigate']['overwrite_config']:
            copa['config'] = opt
        else:
            opt = copa['config']
            
    # log file #############################################
    f_log = copa_log('open', copa)

    # augmenting input #####################################
    if opt['navigate']['do_augment']:
        print("augment ...")
        coag.aug_main(copa, f_log)

    # preprocessing ########################################
    if opt['navigate']['do_preproc']:
        print("preproc ...")
        copa = copp.pp_main(copa, f_log)
        copa_save(copa)

    # diagnosis ############################################
    if opt['navigate']['do_diagnosis']:
        print("diagnosis ...")
        diagnosis_err = copp.diagnosis(copa, f_log)

    # stylization ##########################################
    print("stylization ...")
    # global segments
    if opt['navigate']['do_styl_glob']:
        copa = cost.styl_glob(copa, f_log)
        copa_save(copa)

    # local segments
    if opt['navigate']['do_styl_loc']:
        copa = cost.styl_loc(copa, f_log)
        copa_save(copa)

    # extended local feature set
    if opt['navigate']['do_styl_loc_ext']:
        copa = cost.styl_loc_ext(copa, f_log)
        copa_save(copa)

    # boundary signals
    if (opt['navigate']['do_styl_bnd'] or
        opt['navigate']['do_styl_bnd_win'] or
            opt['navigate']['do_styl_bnd_trend']):
        copa = cost.styl_bnd(copa, f_log)
        copa_save(copa)

    # general features: f0, energy
    if opt['navigate']['do_styl_gnl_f0']:
        copa = cost.styl_gnl(copa, 'f0', f_log)
        copa_save(copa)
    if opt['navigate']['do_styl_gnl_en']:
        copa = cost.styl_gnl(copa, 'en', f_log)
        copa_save(copa)

    # speech rhythm: f0, energy
    if opt['navigate']['do_styl_rhy_f0']:
        copa = cost.styl_rhy(copa, 'f0', f_log)
        copa_save(copa)
    if opt['navigate']['do_styl_rhy_en']:
        copa = cost.styl_rhy(copa, 'en', f_log)
        copa_save(copa)

    # voice quality
    if opt['navigate']['do_styl_voice']:
        copa = cost.styl_voice(copa, f_log)
        copa_save(copa)

    # clustering ##############################################
    print("clustering ...")
    # global segments
    if opt['navigate']['do_clst_glob']:
        copa = cocl.clst_main(copa, 'glob', f_log)
        copa_save(copa)

    # local segments
    if opt['navigate']['do_clst_loc']:
        copa = cocl.clst_main(copa, 'loc', f_log)
        copa_save(copa)

    # export ##################################################
    if opt['navigate']['do_export']:
        print("export ...")
        copa = coex.export_main(copa)

    # plot ####################################################
    if opt['navigate']['do_plot']:
        copa_plots(copa, opt)

    # log end #################################################
    did_log = copa_log('val', copa, f_log)

    return copa


def copa_plots(copa, opt):

    '''
    plotting
    '''

    # browse through copa
    copl.plot_browse(copa)

    # groupings
    # types: 'glob'|'loc'...
    for x in opt['plot']['grp']['type'].keys():
        # sets: 'acc'|'decl'...
        for y in opt['plot']['grp']['type'][x].keys():
            if opt['plot']['grp']['type'][x][y]:
                copl.plot_main({'call': 'grp', 'state': 'final', 'type': x,
                                'set': y, 'fit': copa}, opt)


def copa_log(task, copa, f_log=''):

    '''
    log file
    add copa['config']['fsys']['log']['f']: log file name
    appends init row to log file
    
    Args:
      task 'open'|'val' - init logfile, write validation metrics into logfile
      copa (can be dummy container with 'config' key only {'config':opt})
      f_log (for task 'val'
    
    Returns:
      if task=='val': True
      if task=='open': f_log, logFileHandle
    '''

    if task == 'open':
        ff = os.path.join(copa['config']['fsys']['export']['dir'],
                          f"{copa['config']['fsys']['export']['stm']}.log.txt")
        copa['config']['fsys']['log'] = {'f': ff}
        f_log = open(copa['config']['fsys']['log']['f'], 'a')
        f_log.write(f"\n## {utils.isotime()}\n")
        return f_log
    elif task == 'val':
        vv = copa['val']
        f_log.write("# validation\n")
        for x in sorted(vv.keys()):
            for y in sorted(vv[x].keys()):
                for z in sorted(vv[x][y].keys()):
                    f_log.write(f"{x}.{y}.{z}: {vv[x][y][z]}\n")
        f_log.close()
        return True


def copa_save(copa, infx=None):

    '''
    save/load
    saves copa into pickle file according to copa['config']['fsys']['export']
    
    Args:
      copa
      infx <''> to further specify file name
    '''
    
    z = os.path.join(copa['config']['fsys']['export']['dir'],
                     copa['config']['fsys']['export']['stm'])
    
    if infx is not None:
        f = f"{z}.{infx}.pickle"
    else:
        f = f"{z}.pickle"

    utils.output_wrapper(copa, f, 'pickle')


def copa_load(opt, infx=''):

    '''
    loads copa from pickle file
    
    Args:
      opt (dict) copa['config']
      infx (str) to further specify file name
    
    Returns:
      copa (dict)
    '''

    z = os.path.join(opt['fsys']['export']['dir'],
                     opt['fsys']['export']['stm'])
    
    if infx is not None:
        f = f"{z}.{infx}.pickle"
    else:
        
        f = f"{z}.pickle"
        
    return utils.input_wrapper(f, 'pickle')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="copasul.py -- Intonation analysis tool version 1.0.6")
    parser.add_argument(
        '-c', '--config', help='myConfigFile.json', type=str, required=True)
    kwargs = vars(parser.parse_args())
    copa = copasul(**kwargs)

