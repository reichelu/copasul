#!/usr/bin/env python3


# author: Uwe Reichel, Budapest, 2016

import argparse
import os
import sys
import mylib as myl
import copasul_root as coro
import numpy as np
import os.path as op
import re

##### initialize copa #######################################
def copa_init(opt):
    copa = {'config':opt, 'data':{}, 'clst':{}, 'val':{}}
    copa['clst']={'loc':{}, 'glob':{}}
    copa['clst']['loc']={'c':[], 'ij':[]}
    copa['clst']['glob']={'c':[], 'ij':[]}
    copa['val']['styl'] = {'glob':{}, 'loc':{}}
    copa['val']['clst'] = {'glob':{}, 'loc':{}}
    return copa

##### initialize config #####################################
# IN:
#   conf: config dict or string myConfigFile.json
# OUT:
#   opt: initialized config (defaults added etc.)
def copa_opt_init(conf):

    ## user defined
    # input is dictionary or json file name string
    if type(conf) is dict:
        opt=conf
    else:
        opt = myl.input_wrapper(conf,'json')
    ## defaults
    myCwd = coro.copa_root()
    f_dfl = os.path.join(myCwd,'config','copasul_default_config.json')
    dfl =  myl.input_wrapper(f_dfl,'json')
    # merged
    opt = myl.opt_default(opt,dfl)

    # adjust variable types
    # filter frequencies as list
    for x in ['chunk','syl']:
        opt['augment'][x]['flt']['f'] = np.asarray(opt['augment'][x]['flt']['f'])
    # channels -> array indices, i.e. -1
    # store highest idx+1 (! for range()) in  opt['fsys']['nc']
    opt['fsys']['nc'] = 1
    for x in opt['fsys']['channel']:
        if opt['fsys']['channel'][x] > opt['fsys']['nc']:
             opt['fsys']['nc'] = opt['fsys']['channel'][x]
        opt['fsys']['channel'][x] -= 1

    # add defaults and channel idx for AUGMENT tier_out_stm . '_myChannelIdx'
    if 'augment' in opt['fsys']:
        # over 'chunk', 'syl', 'glob', 'loc' (order important)
        for x in ['chunk','syl','glob','loc']:
            if x not in opt['fsys']['augment']:
                opt['fsys']['augment'][x] = {}
            if (('tier_out_stm' not in opt['fsys']['augment'][x]) or
                len(opt['fsys']['augment'][x]['tier_out_stm'])==0):
                opt['fsys']['augment'][x]['tier_out_stm'] = "copa_{}".format(x)
            # over channelIdx
            for i in range(opt['fsys']['nc']):
                ##!!ci opt['fsys']['channel']["{}_{}".format(opt['fsys']['augment'][x]['tier_out_stm'],i)]=i
                opt['fsys']['channel']["{}_{}".format(opt['fsys']['augment'][x]['tier_out_stm'],int(i+1))]=i
            # special add of _bnd for syllable boundaries
            if x=='syl':
                for i in range(opt['fsys']['nc']):
                    ##!!ci opt['fsys']['channel']["{}_bnd_{}".format(opt['fsys']['augment'][x]['tier_out_stm'],i)]=i
                    opt['fsys']['channel']["{}_bnd_{}".format(opt['fsys']['augment'][x]['tier_out_stm'],int(i+1))]=i
            # add further defaults
            for y in ['glob','loc','syl']:
                if x != y: continue
                if re.search('(glob|syl)',y):
                    parent = 'chunk'
                else:
                    parent = 'glob'
                if 'tier_parent' not in opt['fsys']['augment'][x]:
                    opt['fsys']['augment'][x]['tier_parent'] = opt['fsys']['augment'][parent]['tier_out_stm']
                if y=='syl':
                    continue
                if 'tier' not in opt['fsys']['augment'][x]:
                    opt['fsys']['augment'][x]['tier'] = opt['fsys']['augment']['syl']['tier_out_stm']
                    
    # unify bnd, gnl_* and rhy_* tiers as lists
    for x in ['bnd','gnl_en','gnl_f0','rhy_en','rhy_f0','voice']:
        for y in ['tier', 'tier_rate']:
            if x in opt['fsys'] and y in opt['fsys'][x]:
                if type(opt['fsys'][x][y]) is not list:
                    opt['fsys'][x][y] = [opt['fsys'][x][y]]
                    
    # dependencies
    if re.search('^seed',opt['augment']['glob']['cntr_mtd']):
        if not opt['augment']['glob']['use_gaps']:
            print('Init: IP augmentation by means of {} centroid method requires value 1 for use_gaps. Changed to 1.'.format(opt['augment']['glob']['cntr_mtd']))
            opt['augment']['glob']['use_gaps']=1

    # force at least 1 ncl and accent to be in file. Otherwise creation summary statistics table fails
    # due to unequal number of rows
    if myl.ext_true(opt['fsys']['export'],'summary'):
        opt['augment']['loc']['force']=True
    if myl.ext_true(opt['augment']['loc'],'force'):
        opt['augment']['syl']['force']=True
    

        
    # distribute label and channel specs to subdicts
    # take over dir|typ|ext from 'annot'
    # needed e.g. to extract tiers by pp_tiernames()
    fs = myl.lists('featsets')
    fs.append('augment')
    fs.append('chunk')
    for x in fs:
        if x not in opt['fsys']:
            continue
        opt['fsys'][x]['lab_pau'] = opt['fsys']['label']['pau']
        opt['fsys'][x]['channel'] = opt['fsys']['channel']
        opt['fsys'][x]['nc'] = opt['fsys']['nc']
        for y in ['dir','typ','ext']:
            opt['fsys'][x][y] = opt['fsys']['annot'][y]

    # add labels to 'annot' and 'augment' subdicts
    for x in ['syl','chunk','pau']:
        opt['fsys']['augment']['lab_{}'.format(x)] = opt['fsys']['label'][x]
        opt['fsys']['annot']['lab_{}'.format(x)] = opt['fsys']['label'][x]
        
    # distribute sample rate, and set to 100
    opt['fs']=100
    for x in myl.lists('featsets'):
        if (x not in opt['styl']):
            opt['styl'][x] = {}
        opt['styl'][x]['fs'] = opt['fs']
        if ((x in opt['preproc']) and ('point_win' in opt['preproc'][x])):
            opt['styl'][x]['point_win'] = opt['preproc'][x]['point_win']
        else:
            opt['styl'][x]['point_win'] = opt['preproc']['point_win']

    # add missing navigations
    for x in ["from_scratch", "overwrite_config", "do_diagnosis",
              "do_augment_chunk", "do_augment_syl", "do_augment_glob",
              "do_augment_loc", "do_preproc", "do_styl_glob",
              "do_styl_loc", "do_styl_loc_ext",
              "do_styl_gnl_en", "do_styl_bnd", "do_styl_bnd_win",
              "do_styl_bnd_trend", "do_styl_rhy_f0", "do_styl_rhy_en",
              "do_clst_loc", "do_clst_glob", "do_styl_voice", "do_export",
              "do_plot"]:
        if x not in opt['navigate']:
            opt['navigate'][x] = False

    # set navigate macros
    opt['navigate']['do_augment'] = False
    for x in ['chunk','syl','glob','loc']:
        if opt['navigate']["do_augment_{}".format(x)]:
            opt['navigate']['do_augment'] = True
            break

    # full path export (default False; transform [0,1] to boolean)
    if 'fullpath' not in opt['fsys']['export']:
        opt['fsys']['export']['fullpath']=False
    else:
        opt['fsys']['export']['fullpath']=bool(opt['fsys']['export']['fullpath'])

    # set fromScratch to True if no copa.pickle file available
    f = "{}.pickle".format(os.path.join(opt['fsys']['export']['dir'],
                                        opt['fsys']['export']['stm']))
    if not op.isfile(f):
        opt['navigate']['from_scratch']=True

    # default empty grouping
    if ('grp' not in opt['fsys']):
        opt['fsys']['grp'] = {'lab':[],'src':'','sep':''}

    # plot options
    if 'color' not in opt['plot']:
        opt['plot']['color'] = 1
    for x in ['browse', 'grp']:
        if 'color' not in opt['plot'][x]:
            opt['plot'][x]['color'] = opt['plot']['color']

    # check [preproc][base_prct_grp] <-> [fsys][grp][lab] compliance
    if 'base_prct_grp' in opt['preproc']:
        # empty or not needed subdict
        if ((len(list(opt['preproc']['base_prct_grp'].keys()))==0) or
            opt['preproc']['base_prct']==0):
            del opt['preproc']['base_prct_grp']
        else:
            # over channel idx
            for ci in opt['preproc']['base_prct_grp']:
                g = opt['preproc']['base_prct_grp'][ci]
                # label cannot be inferred from file name
                if g not in opt['fsys']['grp']['lab']:
                    print('WARNING: labels in opt.preproc.base_prct_grp need to be extractable from filename by opt.fsys.grp.lab. Removed. Base percentile will be calculated separately for each file.')
                    del opt['preproc']['base_prct_grp']
                    break

    # transform string key to channel idx in copa['data']['fileIdx']
    if 'base_prct_grp' in opt['preproc']:
        bpg={}
        lcis = myl.sorted_keys(opt['preproc']['base_prct_grp'])
        for cis in lcis:
            ci = int(cis)-1
            opt['preproc']['base_prct_grp'][ci] = opt['preproc']['base_prct_grp'][cis]
            del opt['preproc']['base_prct_grp'][cis]

    return opt

