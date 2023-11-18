# author: Uwe Reichel, Budapest, 2016

import argparse
import audeer
import numpy as np
import os
import re
import sys
from typing import Union, IO

import copasul.copasul_augment as coag
import copasul.copasul_clst as cocl
import copasul.copasul_export as coex
import copasul.copasul_init as coin
import copasul.copasul_plot as copl
import copasul.copasul_preproc as copp
import copasul.copasul_styl as cost
import copasul.copasul_utils as utils


class Copasul(object):

    r"""prosodic analyses"""

    def __init__(self):

        super().__init__()
        
    def process(self, config: Union[dict, str], copa: dict = None) -> dict:

        ''' process files specified in config
        
        Args:
        config: (dict or str) of process specifications. If type is str, a name
             of a config json file is expected
        copa: (dict) previously generated copasul output dict for warm starts

        Returns:
        copa: (dict) copasul output
        '''
        
        if copa is not None:
            config["navigate"]["from_scratch"] = False

        # config #########################################
        opt = coin.copa_opt_init(config)

        # output dir
        _ = audeer.mkdir(opt["fsys"]["export"]["dir"])
    
        # generate new copa dict?
        # or load it/take it from input args
        if opt['navigate']['from_scratch']:
            copa = coin.copa_init(opt)
        else:
            if copa is None or len(copa.keys()) == 0:
                copa = self.load(opt)
            
            # replace copa-contained config?
            if opt['navigate']['overwrite_config']:
                copa['config'] = opt
            else:
                opt = copa['config']
            
        # log file #############################################
        f_log = self.log('open', copa)

        # augmenting input #####################################
        if opt['navigate']['do_augment']:
            coag.aug_main(copa, f_log)

        # preprocessing ########################################
        if opt['navigate']['do_preproc']:
            copa = copp.pp_main(copa, f_log)
            self.save(copa)

        # diagnosis ############################################
        if opt['navigate']['do_diagnosis']:
            print("diagnosis ...")
            diagnosis_err = copp.diagnosis(copa, f_log)

        # stylization ##########################################
        # global segments
        if opt['navigate']['do_styl_glob']:
            copa = cost.styl_glob(copa, f_log)
            self.save(copa)

        # local segments
        if opt['navigate']['do_styl_loc']:
            copa = cost.styl_loc(copa, f_log)
            self.save(copa)

        # extended local feature set
        if opt['navigate']['do_styl_loc_ext']:
            copa = cost.styl_loc_ext(copa, f_log)
            self.save(copa)

        # boundary signals
        if (opt['navigate']['do_styl_bnd'] or
            opt['navigate']['do_styl_bnd_win'] or
            opt['navigate']['do_styl_bnd_trend']):
            copa = cost.styl_bnd(copa, f_log)
            self.save(copa)

        # general features: f0, energy
        if opt['navigate']['do_styl_gnl_f0']:
            copa = cost.styl_gnl(copa, 'f0', f_log)
            self.save(copa)
        if opt['navigate']['do_styl_gnl_en']:
            copa = cost.styl_gnl(copa, 'en', f_log)
            self.save(copa)

        # speech rhythm: f0, energy
        if opt['navigate']['do_styl_rhy_f0']:
            copa = cost.styl_rhy(copa, 'f0', f_log)
            self.save(copa)
        if opt['navigate']['do_styl_rhy_en']:
            copa = cost.styl_rhy(copa, 'en', f_log)
            self.save(copa)

        # voice quality
        if opt['navigate']['do_styl_voice']:
            copa = cost.styl_voice(copa, f_log)
            self.save(copa)

        # clustering ##############################################
        # global segments
        if opt['navigate']['do_clst_glob']:
            print("clustering glob ...")
            copa = cocl.clst_main(copa, 'glob', f_log)
            self.save(copa)

        # local segments
        if opt['navigate']['do_clst_loc']:
            print("clustering loc ...")
            copa = cocl.clst_main(copa, 'loc', f_log)
            self.save(copa)

        # export ##################################################
        if opt['navigate']['do_export']:
            print("export ...")
            copa = coex.export_main(copa)

        # plot ####################################################
        if opt['navigate']['do_plot']:
            self.plotting(copa, opt)

        # log end #################################################
        did_log = self.log('val', copa, f_log)

        print("done.")
    
        return copa


    def plotting(self, copa: dict, opt: dict):

        '''
        plotting. Wraps around copasul.copasul_plot
        
        Args:
        copa: (dict)
        opt: (dict)
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

                    
    def log(self, task: str, copa: dict, f_log: IO = None) -> Union[bool, IO]:

        '''
        log file
        add copa['config']['fsys']['log']['f']: log file name
        appends init row to log file
    
        Args:
        task (str) 'open'|'val' - init logfile, write validation metrics into logfile
        copa (dict)
        f_log (file handle) (for task 'val')
    
        Returns:
        file handle for task=='open', else bool True
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

        
    def save(self, copa: dict, infx: str = None):

        '''
        saves copa into pickle file according to copa['config']['fsys']['export']
    
        Args:
        copa (dict)
        infx (str) for file name infix
        '''
    
        z = os.path.join(copa['config']['fsys']['export']['dir'],
                         copa['config']['fsys']['export']['stm'])
    
        if infx is not None:
            f = f"{z}.{infx}.pickle"
        else:
            f = f"{z}.pickle"

        utils.output_wrapper(copa, f, 'pickle')

        
    def load(self, opt: dict, infx: str = None) -> dict:

        '''
        loads copa from pickle file
    
        Args:
        opt (dict) copa['config']
        infx (str) for file name infix
    
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

