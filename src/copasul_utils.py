import os
import shutil as sh
import sys
import numpy as np
import pandas as pd
import statistics as stat
import scipy.stats as st
import scipy.cluster.vq as sc
import sklearn.preprocessing as sp
import json
import pickle
import re
import datetime
import math
import xml.etree.ElementTree as et
import xml.sax.saxutils as xs
import subprocess
import copy as cp
import csv
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from pathlib import Path


###########################################################
### collection of general purpose functions ###############
###########################################################


# I/O functions
# input_wrapper()
# i_tg()       -> tg dict
# i_par()      -> interchange dict
# i_copa_xml() -> interchange dict
# i_lol()      -> 1- or 2dim list of strings
# i_seg_lab()  -> d.t [[on off]...]
#                  .lab [label ...]
# i_keyVal() -> dict z: a;b -> z["a"]="b"
# i_numpy: calls np.loadtxt() returns np.array list of floats
#          (1 col -> 1-dim array; else 1 sublist per row)
#       'pandas_csv': csv file into dict colName -> colContent (using pandas)
# output_wrapper()
# o_tg()
# o_par()
# o_copa_xml()
# annotation processing
# par_mau2word()
#   creates word segment tier from .par file MAU tier
# tg_mau2chunk()
#   creates chunk tier from pauses in MAU
# tg_add()
#   adds/replaces tier to TextGrid
# tg_tier()
#   returns tier from TextGrid
# tg_tn()
#   returns list TextGrid tier names
# tg_tierType()
#   returns 'points' or 'intervals' (= key to access items in tier)
# tg_mrg()
#   select tiers of >=1 TextGrid and create new TextGrid from these tiers
# format transformations
# tg_tab2tier():
#   numpy array to TextGrid tier
# tg_tier2tab():
#   TextGrid tier to numpy and label array
# tg2inter()
#   TextGrid -> interchange format
# inter2tg()
#   interchange format -> TextGrid
# tg2par()
#   as tg2inter() + added header element with sample rate
# par2tg()
#   as inter2tg(), but omitting the header item
# tab2copa_xml_tier()
#   numpy array to copa_xml tier
# pipeline: add tiers to existing TextGrid (w/o replacement)
# myTg = myl.input_wrapper(myInputFile,'TextGrid')
# myAdd = myl.input_wrapper(myAddFile,'TextGrid')
# with tier replacement:
# opt = {'repl':True}
# without replacement
# opt = {'repl':False}
# for x in tg_tn(myAdd):
#    myTg = tg_add(myTg,tg_tier(myAdd,x),opt)
# processing pandas_csv dict, e.g. in copasul context
# split_by_grp() - if data to be split e.g. by analysis tier
#                  -> dd[myGrpLevel] = subDict
# profile_wrapper() - transforming it into profile incl plotting
#  calls the following functions that can also be applied in isolation
#    pw_preproc(): wrapper around the following functions
#    pw_str2float(): panda is biased towards strings -> corrections, incl. NaNs
#    pw_outl2nan(): replacing outliers by NaN
#    pw_nan2mean(): replacing NaNs by column means/medians (can be further grouped by 1 variable)
#    pw_abs(): abs-transform of selected columns
#    pw_centerScale(): column-wise robust center/scaling

# basic matrix op functions
# cmat(): 2-dim matrix concat, converts if needed all input to 2-dim arrays
# lol(): ensure always 2-dim array (needed e.g. for np.concatenate of 2-dim and
#        1-sim array)
# push(): appending without dimensioning flattening (vs append)
# find(): simulation of matlab find functionality
# find_interval(): indices of values within range, pretty slow
# first_interval(): first index of row containint specific element
#              IMPORTANT: ask whether idx is >=0 since -1 indicates not found
def push(x, y, a=0):

    '''
    pushes 1 additional element y to array x (default: row-wise)
    !! if x is not empty, i.e. not []: yDim must be xDim-1, e.g.
          if x 1-dim: y must be scalar
          if x 2-dim: y must 1-dim
    if x is empty, i.e. [], the dimension of the output is yDim+1
    Differences to np.append:
      append flattens arrays if dimension of x,y differ, push does not
    REMARK: cmat() might be more appropriate if 2-dim is to be returned
    
    Args:
      x array (can be empty)
      y array (if x not empty, then one dimension less than x)
      a axis (0: push row, 1: push column)
    
    Returns:
      [x y]
    '''

    if (listType(y) and len(y) == 0):
        return x
    if len(x) == 0:
        return np.array([y])
    return np.concatenate((x, [y]), axis=a)



def colconcat(x, y):

    '''
    concatenate 2 1-dim arrays to 2-column 2d array
    '''

    xt = np.reshape(np.asarray(x), (-1, 1))
    yt = np.reshape(np.asarray(y), (-1, 1))
    return np.concatenate((xt, yt), axis=1)


def listType(y):

    '''
    returns True if input is numpy array or list; else False
    '''

    if (type(y) == np.ndarray or type(y) == list):
        return True
    return False



def isotime():

    '''
    returns isotime format, e.g. 2016-12-05T18:25:54Z
    '''

    return datetime.datetime.now().isoformat()



def sec2idx(i, fs, ons=0):

    '''
    transforms seconds to numpy array indices (=samplesIdx-1)
    '''

    #ok

    return np.round(i*fs+ons-1).astype(int)



def sec2smp(i, fs, ons=0):

    '''
    transforms seconds to sample indices (arrayIdx+1)
    '''

    #ok
    
    return np.round(i*fs+ons).astype(int)



def idx2sec(i, fs, ons=0):

    '''
    transforms numpy array indices (=samplesIdx-1) to seconds
    '''

    return (i+1+ons)/fs



def smp2sec(i, fs, ons=0):

    '''
    transforms sample indices (arrayIdx+1) to seconds
    '''

    return (i+ons)/fs



def fileExists(n):

    '''
    
    Args:
      s file name
    
    Returns:
      TRUE if file exists, else FALSE
    '''

    if os.path.isfile(n):
        return True
    return False



def seg_list(x):

    '''
    segment list into same element segments
    
    Args:
      x: 1-dim list
    
    Returns:
      y: 2-dim list
    example:
      [a,a,b,b,a,c,c] -> [[a,a],[b,b],[a],[c,c]]
    '''

    if len(x) == 0:
        return [[]]
    y = [[x[0]]]
    for i in range(1, len(x)):
        if x[i] == x[i-1]:
            y[-1].append(x[i])
        else:
            y.append([x[i]])
    return y



def reversi(x, opt={}):

    '''
    reversi operation: replace captured element by surround ones
    
    Args:
      x: 1-dim list
      opt:
        .infx: <1> infix max length
        .ngb_l: <2> left context min length
        .ngb_r: <2> right context max length
    
    Returns:
      y: x or all elements replaced by first one
    Examples (default options):
      [a,a,b,a,a] -> [a,a,a,a,a]
      [a,a,b,a]   -> [a,a,b,a] (right context too short)
      [a,a,b,c,c] -> [a,a,b,c,c] (left and right context not the same)
      [a,a,b,c,c] -> [a,a,b,a,c] (4 instead of 3 uniform-element sublists)
    '''

    opt = opt_default(opt, {"infx": 1, "ngb_l": 2, "ngb_r": 2})
    if len(x) < opt["infx"]+opt["ngb_l"]+opt["ngb_r"]:
        return x
    y = seg_list(x)
    if (len(y) == 3 and y[0][0] == y[-1][0] and
        len(y[0]) >= opt["ngb_l"] and
        len(y[-1]) >= opt["ngb_r"] and
            len(y[1]) <= opt["infx"]):
        return [y[0][0]] * len(x)
    return x



def lists(typ='register', ret='list'):

    '''
    returns predefined lists or sets (e.g. to address keys in copa dict)
    ret: return 'set' or 'list'
    '''

    ll = {'register': ['bl', 'ml', 'tl', 'rng'],
          'bndtyp': ['std', 'win', 'trend'],
          'bndfeat': ['r', 'rms', 'rms_pre', 'rms_post',
                      'sd_prepost', 'sd_pre', 'sd_post',
                      'corrD', 'corrD_pre', 'corrD_post',
                      'rmsR', 'rmsR_pre', 'rmsR_post',
                      'aicI', 'aicI_pre', 'aicI_post',
                      'd_o', 'd_m'],
          'bgd': ['bnd', 'gnl_f0', 'gnl_en', 'rhy_f0', 'rhy_en', 'voice'],
          'featsets': ['glob', 'loc', 'bnd', 'gnl_f0', 'gnl_en',
                       'rhy_f0', 'rhy_en', 'voice'],
          'afa': ['aud', 'f0', 'annot', 'pulse'],
          'fac': ['glob', 'loc', 'bnd', 'gnl_f0', 'gnl_en',
                  'rhy_f0', 'rhy_en', 'augment', 'chunk', 'voice'],
          'facafa': ['glob', 'loc', 'bnd', 'gnl_f0', 'gnl_en',
                     'rhy_f0', 'rhy_en', 'augment', 'chunk',
                     'aud', 'f0', 'annot', 'pulse'],
          'factors': ['class', 'ci', 'fi', 'si', 'gi', 'stm', 'tier', 'spk',
                      'is_init', 'is_fin', 'is_init_chunk', 'is_fin_chunk']}
    if typ in ll:
        if ret == 'list':
            return ll[typ]
        else:
            return set(ll[typ])
    if ret == 'list':
        return []
    return set()



def split_by_grp(dd, gc):

    '''
    splits dict d by levels of column gv
    
    Args:
      dd: dict derived from input_wrapper(...,'pandas_csv')
      gc: grouping column name
    
    Returns:
      ds: dict
         myGrpLevel -> dSubset with gv = myGrpLevel (same keys as d)
      if gv is not in d, then myGrpLevel = 'x'
    '''

    d = cp.deepcopy(dd)
    ds = {}
    if gc not in d:
        ds['x'] = d
        return ds
    # colnames
    cn = d.keys()
    # make lists to arrays for indexing
    for n in cn:
        d[n] = np.array(d[n])
    # grouping column
    g = d[gc]
    # over grouping levels
    for lev in np.unique(g):
        ds[lev] = {}
        # assign subset
        for n in cn:
            ds[lev][n] = d[n][(g == lev)]
    return ds


def profile_wrapper(dd, opt):

    '''
    wrapper around profile plotting of opt.mean values for selected features
    
    Args:
      d: dict derived from input_wrapper(...,'pandas_csv')
      opt:
         'navigate': processing steps
               str2float: ensure that all values in d are floats and not strings
                          (relevant to correctly interprete np.nan) <True>
               nan2mean: replace NaN by column opt.mean <True>
               nrm: column-wise centering+scaling <True>
               dict2df: create pandas data frame (internally set to True)
               plot: output plot files <False> (requires subdict 'plot')
         'feat': [list of feature column names
                  (in wanted output order from top to bottom)] !
         'absfeat': [list of features for which absolute values to be taken] <[]>
         'abs_add': add (=True) or replace (=False) absfeat column names <False>
         'grp': [list of grouping variable column names] !
         'stat': <'median'>|'mean'
         'plot': dict
              'stm': dir/stem ! of output file
              'figsize': <()> tuple width x height
              'bw': <False>|True
              'fs_ytick': <20> fontsize ytick
              'fs_legend': <30> fontsize legend
              'title': <''> figure title
              'lw': <5> line width
              'concat_grp': <True> concatenate grouping variable name to plot['title']
        'sort': dict if values/featnames to be sorted by one groupingLevel of one
                groupVar (usually only makes sense if opt['grp'] contains only
                1 element). just working for single grpvar (since feature names
                  are just stored once)
                  'grp':  grpvar name <''>
                  'level': groupingLevel <''>
                  'reverse': True|<False> (default from low to high)
        (! = obligatory)
    
    Returns:
      p: profile dict
         lab -> [featureNames]
         grp[myGrpVar][myGrpLevel] -> [values] same length and order as lab
    REMARKS:
      it is not controlled whether or not input D is already split by analysis
      tiers. If such a splitting is needed (e.g. for gnl or bnd feature sets)
      use ds = split_by_grp(d,'tier') first and
      apply profile_wrapper() separately for each ds[myTierName]
    '''


    # opt init
    opt = opt_default(opt, {'stat': 'median', 'navigate': {},
                            'absfeat': [], 'plot': {}, 'sort': {}})

    opt['navigate'] = opt_default(opt['navigate'],
                                  {'str2float': True,
                                   'nan2mean': True,
                                   'abs_add': False,
                                   'nrm': True,
                                   'plot': False})
    opt['plot'] = opt_default(opt['plot'], {'bw': False, 'fs_ytick': 20,
                                            'fs_title': 30,
                                            'fs_legend': 30, 'title': '',
                                            'lw': 5, 'figsize': (),
                                            'loc_legend': 'upper left',
                                            'lab_legend': [],
                                            'concat_grp': False})
    opt['sort'] = opt_default(opt['sort'], {'grp': [],
                                            'level': [],
                                            'reverse': False})

    opt['navigate']['dict2df'] = True
    if 'stm' not in opt['plot']:
        opt['navigate']['plot'] = False

    # preprocessing
    # opt['feat'] might be updated by *_abs columns
    # d is now pd.dataframe
    d, opt = pw_preproc(dd, opt)

    # profile
    p = pw_prof(d, opt)

    # sorting
    if len(opt['sort']['grp']) > 0:
        p = pw_sort(p, opt)

    # plotting
    if opt['navigate']['plot']:
        pw_plot(p, opt['plot'])

    return p



def pw_sort(p, opt):

    '''
    Args:
    p (dict)
    opt (dict)

    Returns:
    p (dict)
    lab -> [featureNames]
    grp[myGrpVar][myGrpLevel] -> [values] same length and order as lab
    
    Remarks
    'sort': dict if values/featnames to be sorted by one groupingLevel of one
            groupVar (usually only makes sense if opt['grp'] contains only
            1 element)
            .myGroupingVar:
                'level': myGroupingLevel
                'reverse': True|False (default from low to high)
    '''

    os = opt['sort']
    gv = os['grp']
    gl = os['level']
    # values to be sorted
    y = p['grp'][gv][gl]
    i = np.argsort(y)
    if os['reverse']:
        i = np.flip(i, 0)
    # sort featnames
    p['lab'] = np.asarray(p['lab'])[i]
    # sort mean values of all grp levels accordingly
    for lev in p['grp'][gv]:
        p['grp'][gv][lev] = np.asarray(p['grp'][gv][lev])[i]
    return p



def pw_plot(p, opt):

    '''
    called by profile_wrapper() for plotting
    
    Args:
      p: profile dict
         lab -> [featureNames]
         grp[myGrpVar][myGrpLevel] -> [values] same length and order as lab
      opt: opt['plot'] dict from profile_wrapper()
    '''


    # plotting
    if opt['bw']:
        cols = ['k-', 'k--', 'k-.', 'k:']
    else:
        cols = ['-g', '-r', '-b', '-k']

    # y-axis ticks
    yi = np.asarray(range(1, len(p['lab'])+1))

    # over grouping variables
    for g in p['grp']:

        colI = 0
        y = p['grp'][g]

        # file name
        if len(p['grp'].keys()) > 1 or opt["concat_grp"]:
            fo = "{}_{}.pdf".format(opt['stm'], g)
        else:
            fo = "{}.pdf".format(opt['stm'])

        fig = newfig(opt['figsize'])
        # over levels
        for lev in sorted(y.keys()):
            plt.plot(y[lev], yi, "{}".format(
                cols[colI]), label=lev, linewidth=5)
            colI += 1

        plt.yticks(yi, p['lab'], fontsize=opt['fs_ytick'])
        if opt['concat_grp']:
            plt.title("{}: {}".format(opt['title'], g),
                      fontsize=opt['fs_title'])
        else:
            plt.title(opt['title'], fontsize=opt['fs_title'])
        if len(opt['lab_legend']) > 0:
            plt.legend(opt['lab_legend'],
                       fontsize=opt['fs_legend'],
                       loc=opt['loc_legend'])
        else:
            plt.legend(fontsize=opt['fs_legend'],
                       loc=opt['loc_legend'])
        plt.show()
        print(fo)
        fig.savefig(fo)
        # stopgo()
    return


def pw_prof(d, opt):

    '''
    called by profile_wrapper()
    IN, OUT, cf there
    '''

    p = {'lab': opt['feat'], 'grp': {}}

    # over grouping columns
    for g in opt['grp']:
        # m[myFeat][myGrpLevel] = meanValue
        if opt['stat'] == 'median':
            m = d.groupby([g]).median()
        else:
            m = d.groupby([g]).mean()

        p['grp'][g] = {}
        xf = opt['feat'][0]
        # over grpLevels
        for lev in m[xf].keys():
            p['grp'][g][lev] = []
            # over features
            for f in opt['feat']:
                p['grp'][g][lev].append(m[f][lev])

    return p


def pw_preproc(d, opt):

    '''
    wrapper around preprocessing for profile generation.
    called by profile_wrapper() or standalone
    
    Args:
      d: dict by input_wrapper(...,'pandas_csv')
      opt:
         'navigate': processing steps
               str2float: ensure that all values in d are floats and not strings
                          (relevant to correctly interprete np.nan) <True>
               nan2mean: replace NaN by column opt.mean <True>
               nrm: column-wise centering+scaling <True>
               dict2df: create pandas data frame (always <True> if called by
                        profile_wrapper(); if not set: <False>
         'feat': [list of feature column names
                  (in wanted output order from top to bottom)] !
         'absfeat': [list of features for which absolute values to be taken] <[]>
         'abs_add': <False> add (=True) or replace (=False) absfeat column names in d
         'stat': <'median'>|'mean'
         'grp_n2m': grouping column name for nan2mean separately for each factor level <''>
    
    Returns:
      d preprocessed
      opt: evtl. with updated 'feat' list (in case of abs_add
    '''


    # opt init
    opt = opt_default(opt, {'stat': 'median', 'navigate': {},
                            'absfeat': [], 'grp_n2m': ''})
    opt['navigate'] = opt_default(opt['navigate'],
                                  {'str2float': True,
                                   'outl2nan': False,
                                   'nan2mean': True,
                                   'abs_add': False,
                                   'nrm': True,
                                   'dict2df': False})

    d = cp.deepcopy(d)
    if opt['navigate']['str2float']:
        d = pw_str2float(d, opt['feat'])
    if opt['navigate']['outl2nan']:
        d = pw_outl2nan(d, opt['feat'])
    if opt['navigate']['nan2mean']:
        d = pw_nan2mean(d, opt['feat'], opt['stat'], opt['grp_n2m'])
    if len(opt['absfeat']) > 0:
        d, cn = pw_abs(d, opt['absfeat'], opt['navigate']['abs_add'])
        if opt['navigate']['abs_add']:
            # element-wise to keep pre-specified order
            for x in cn:
                if x in opt['feat']:
                    continue
                opt['feat'].append(x)
    if opt['navigate']['nrm']:
        d = pw_centerScale(d, opt['feat'])
    if opt['navigate']['dict2df']:
        d = pd.DataFrame(d)

    return d, opt


def dict2mat(d, cn=[]):

    '''
    transforms dictionary/dataFrame to 2-dim array
    (e.g. for machine learning input)
    
    Args:
      d: dict
      cn: <[]> selected column names; if empty all taken
    
    Returns:
      x: matrix; columns in order of cn (if given), else sorted alphanumerically
    '''

    if len(cn) == 0:
        cn = sorted(d.keys())
    x = ea()
    for z in cn:
        x = push(x, d[z])
    # transpose
    return x.T



def dict2array(d):

    '''
    DEPREC! use dict2mat()
    transforms dictionary/dataFrame to 2-dim array
    (e.g. for machine learning input)
    row in dict = row in array
    
    Args:
      dict
    
    Returns:
      2-dim array
    '''

    x = ea()
    for n in d:
        x = push(x, d[n])
    return x.T



def pw_str2float(d, cn):

    '''
    column-wise string to float
    
    Args:
      d: dict from input_wrapper(...,'pandas_csv')
      cn: list of names of columns to be processed
    
    Returns:
      d: with processed columns
    '''

    for x in cn:
        v = ea()
        for i in range(len(d[x])):
            v = np.append(v, float(d[x][i]))
        d[x] = v
    return d



def pw_outl2nan(d, cn):

    '''
    replaces outliers by np.nan
    
    Args:
      d: dict from input_wrapper(...,'pandas_csv')
      cn: list of names of columns to be processed (!needs to be numeric!)
    '''

    # ignore zeros
    opt = {'zi': False,
           'm': 'mean',
           'f': 4}
    for x in cn:
        io = outl_idx(d[x], opt)
        if np.size(io) > 0:
            d[x][io] = np.nan
    return d



def pw_nan2mean(d, cn, mv='median', grp=''):

    '''
    replaces np.nan by column medians
    
    Args:
      d: dict from input_wrapper(...,'pandas_csv')
      cn: list of names of columns to be processed
      mv: type of mean value '<median>'|'mean'
      grp: grouping variable for nan2mean by factor-level
    
    Returns:
      d: with processed columns
    '''

    # factor-level nan2mean
    if len(grp) > 0 and grp in d:
        # grpLevel -> row indices in d
        gri = {}
        for g in uniq(d[grp]):
            gri[g] = find(d[grp], '==', g)

        # over features
        for x in cn:
            xina = find(d[x], 'is', 'nan')
            if len(xina) == 0:
                continue
            xifi = find(d[x], 'is', 'finite')
            # over grouping levels
            for g in gri:
                ina = intersect(xina, gri[g])
                ifi = intersect(xifi, gri[g])
                if len(ina) == 0 or len(ifi) == 0:
                    continue
                if mv == 'median':
                    m = np.median(d[x][ifi])
                else:
                    m = np.mean(d[x][ifi])
                d[x][ina] = m

    # global nan2mean
    # redo also for factor level nan2mean to catch
    # non-replacement cases
    for x in cn:
        ina = find(d[x], 'is', 'nan')
        if len(ina) == 0:
            continue
        ifi = find(d[x], 'is', 'finite')
        if mv == 'median':
            m = np.median(d[x][ifi])
        else:
            m = np.mean(d[x][ifi])
        d[x][ina] = m
    return d



def pw_abs(d, cn, do_add=False):

    '''
    replace column content by abs values or add column with name *_abs
    
    Args:
      d: dict from input_wrapper(...,'pandas_csv')
      cn: list of names of columns to be processed
      do_add: <False>; if True add abs val columns, if False, replace
          column content
    
    Returns:
      d: with processed columns
      cna: list with updated column names (only different for do_add=True)
    '''

    cna = cp.deepcopy(cn)
    for c in cn:
        if do_add:
            ca = "{}_abs".format(c)
            cna.append(ca)
            d[ca] = np.abs(d[c])
        else:
            d[c] = np.abs(d[c])
    return d, cna



def inf2nan(x):

    '''
    replace +/-Inf values by np.nan (preprocessing for imputing)
    
    Args:
      x: vector of floats
    
    Returns:
      x: infs replaced (now np.array)
    '''

    x = np.asarray(x)
    i = find(x, 'is', 'inf')
    if len(i) > 0:
        x[i] = np.nan
    return x



def pw_centerScale(d, cn):

    '''
    center/scale feature values
    
    Args:
     dict from input_wrapper(...,'pandas_csv')
     cn - relevant column names
    
    Returns:
     d - with relevant columns centered+scaled
    '''

    for x in cn:
        d[x] = d[x].reshape(-1, 1)
        cs = sp.RobustScaler().fit(d[x])
        d[x] = cs.transform(d[x])
        d[x] = d[x][:, 0]
    return d


def find(x_in, op, v):

    '''
    simulation of matlab find for 1-dim data
    
    Args:
     1-dim array
     op: >,<,==,is,isinfimum,issupremum
     value (incl. 'max'|'min'|'nan' etc. in combination with op 'is')
    
    Returns:
     1-dim index array
       exceptions: is infimum/ is supremum: floats (nan if none)
    '''

    x = np.asarray(x_in)
    if op == '==':
        xi = np.asarray((x == v).nonzero())[0, :]
    elif op == '!=':
        xi = np.asarray((x != v).nonzero())[0, :]
    elif op == '>':
        xi = np.asarray((x > v).nonzero())[0, :]
    elif op == '>=':
        xi = np.asarray((x >= v).nonzero())[0, :]
    elif op == '<':
        xi = np.asarray((x < v).nonzero())[0, :]
    elif op == '<=':
        xi = np.asarray((x <= v).nonzero())[0, :]
    elif (op == 'is' and v == 'nan'):
        xi = np.asarray(np.isnan(x).nonzero())[0, :]
    elif (op == 'is' and v == 'finite'):
        xi = np.asarray(np.isfinite(x).nonzero())[0, :]
    elif (op == 'is' and v == 'inf'):
        xi = np.asarray(np.isinf(x).nonzero())[0, :]
    elif (op == 'is' and v == 'max'):
        return find(x, '==', np.max(x))
    elif (op == 'is' and v == 'min'):
        return find(x, '==', np.min(x))
    elif (op == 'isinfimum'):
        xi = np.asarray((x < v).nonzero())[0, :]
        if len(xi) == 0:
            return np.nan
        return int(xi[-1])
    elif (op == 'issupremum'):
        xi = np.asarray((x > v).nonzero())[0, :]
        if len(xi) == 0:
            return np.nan
        return int(xi[0])
    return xi.astype(int)



def mae(x, y=[]):

    '''
    returns mean absolute error of vectors
    or of one vector and zeros (=mean abs dev)
    
    Args:
      x
      y <zeros>
    
    Returns:
      meanAbsError(x,y)
    '''

    if len(y) == 0:
        y = np.zeros(len(x))
    x = np.asarray(x)
    return np.mean(abs(x-y))



def rss(x, y=[]):

    '''
    residual squared deviation
    
    Args:
      x: data vector
      y: prediction vector (e.g. fitted line) or []
    
    Returns:
      r: residual squared deviation
    '''

    if len(y) == 0:
        y = np.zeros(len(x))
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sum((x-y)**2)


def aic_ls(x, y, k=3):

    '''
    aic information criterion for least squares fit
    for model comparison, i.e. without constant terms
    
    Args:
      x: underlying data
      y: predictions (same length as x!)
      k: number of parameters (<3> for single linear fit)
    '''

    n = len(x)
    r = rss(x, y)
    if r == 0:
        return 2*k
    aic = 2*k + n*np.log(r)
    return aic



def robust_log(y):

    '''
    robust natural log
    log(<=0) is np.nan
    '''

    if y <= 0:
        return np.nan
    return np.log(y)



def robust_div(x, y):

    '''
    robust division
    returns np.nan for 0-divisions
    '''

    #ok
    
    if y == 0 or np.isnan(x) or np.isnan(y):
        return np.nan
    
    return x / y



def mse(x, y=[]):

    '''
    mean squared error of vectors
    or of one vector and zeros (=mean squared dev)
    '''

    #ok
    
    if len(y) == 0:
        y = np.zeros(len(x))
        
    return np.mean((x - y) ** 2)



def rmsd(x, y=[]):

    '''
    returns RMSD of two vectors or of one vector and zeros
    
    Args:
     x: (np.array)
     y: (np.array) <zeros(len(x))>
    
    Returns:
     root mean squared dev between x and y
    '''

    #ok

    if len(y) == 0:
        y = np.zeros(len(x))
        
    return np.sqrt(np.mean((x - y) ** 2))



def find_interval(x, iv):

    '''
    ~opposite of which_interval()
    returns indices of values in 1-dim x that are >= iv[0] and <= iv[1]
    
    Args:
      x: 1-dim array
      iv: 2-element interval array
    
    Returns:
      1-dim index array of elements in x within (inclusive) range of iv
    '''

    xi = np.where((x > iv[0]) & (x < iv[1]))
    return xi[0]


def find_interval_deprec(x, iv, fs=-1):
    xi = sorted(intersect(find(x, '>=', iv[0]),
                          find(x, '<=', iv[1])))
    return np.asarray(xi).astype(int)



def first_interval(x, iv):

    '''
    ~opposite of find_interval()
    returns row index of seg containing t (only first in case of multiple)
    
    Args:
     x number
     iv 2-dim array, interval [on offset]
    
    Returns:
     rowIdx <-1>
    IMPORTANT: ask whether idx is >=0 since -1 indicates not found
    '''

    ri = -1
    xi = sorted(intersect(find(iv[:, 0], '<=', x),
                          find(iv[:, 1], '>=', x)))
    if len(xi) > 0:
        ri = xi[0]

    return int(ri)



def seq_windowing(s):

    '''
    vectorized version of windowing
    
    Args:
      s
        .win: window length
        .rng: [on, off] range of indices to be windowed
        .align: <center>|left|right
    
    Returns: [[on off] ...]
    '''

    s = opt_default(s, {"align": "center"})
    if s["align"] == "center":
        vecwin = np.vectorize(windowing)
    elif s["align"] == "right":
        vecwin = np.vectorize(windowing_rightAligned)
    elif s["align"] == "left":
        vecwin = np.vectorize(windowing_leftAligned)
    r = s['rng']
    ww = np.asarray(vecwin(range(r[0], r[1]), s))
    return ww.T


def windowing(i, s):

    '''
    window of length wl on and offset around single index in range [on off]
    vectorized version: seq_windowing
    
    Args:
      i current index
      s
       .win window length
       .rng [on, off] range of indices to be windowed
    
    Returns:
     on, off of window around i
    '''

    # half window
    wl = max([1, math.floor(s['win']/2)])
    r = s['rng']
    on = max([r[0], i-wl])
    off = min([i+wl, r[1]])
    # extend window
    d = (2*wl-1) - (off-on)
    if d > 0:
        if on > r[0]:
            on = max([r[0], on-d])
        elif off < r[1]:
            off = min([off+d, r[1]])
    return on, off



def windowing_rightAligned(i, s):

    '''
    window around each sample so that it is at the right end (no look-ahead)
    '''

    wl, r = int(s['win']), s['rng']
    on = max([r[0], i-wl])
    off = min([i, r[1]])
    # extend window (left only)
    d = wl - (off-on)
    if d > 0:
        if on > r[0]:
            on = max([r[0], on-d])
    # relax 0,0 case (zero length win)
    if off == on:
        off += 1
    return on, off



def windowing_leftAligned(i, s):

    '''
    window around each sample so that it is at the left end (no looking back)
    '''

    wl, r = int(s['win']), s['rng']
    on = max([r[0], i])
    off = min([i+wl, r[1]])
    # extend window (right only)
    d = wl - (off-on)
    if d > 0:
        if off < r[1]:
            off = min([r[1], off+d])
    # relax -1, -1 case (zero length win)
    if on == off:
        on -= 1
    return on, off



def windowing_idx(i, s):

    '''
    as windowing(), but returning all indices from onset to offset
      i current index
      s
       .win window length
       .rng [on, off] range of indices to be windowed
    
    Returns:
     [on:1:off] in window around i
    REMARK for short windows use windowing_idx1() !!
    '''

    on, off = windowing(i, s)
    return np.arange(on, off, 1)



def windowing_idx1(i, s):

    '''
    as windowing(), but returning all indices from onset to offset
      i current index
      s
       .win window length
       .rng [on, off] range of indices to be windowed
    
    Returns:
     [on:1:off] in window around i; off+1 !!
    '''

    on, off = windowing(i, s)
    return np.arange(on, off+1, 1)


def intersect(a, b):

    '''
    returns intersection list of two 1-dim lists
    '''

    #ok
    
    return sorted(set(a) & set(b))



def args2opt(args, reqKey=''):

    '''
    for flexible command line vs embedded call of some function
    - args contains a 'config' key?
    --- config value is dict: opt := args['config']
    --- config value is string (file name): opt is read from config file
    - args contains a <reqKey> key?
    - config := args
    
    Args:
      args dict
      reqKey required key <''>
    
    Returns:
      opt dict read from args.config or equal args
    '''

    if 'config' in args:
        if type(args['config']) is dict:
            opt = args['config']
        else:
            opt = input_wrapper(args['config'], 'json')
    elif reqKey in args:
        opt = args
    else:
        sys.exit("args cannot be transformed to opt")
    return opt


def pr(*args):

    '''
    printing arbitrary number of variables
    '''

    for v in args:
        print(v)



def numkeys(x):

    '''
    returns sorted list of numeric (more general: same data-type) keys
    
    Args:
      x dict with numeric keys
    
    Returns:
      sortedKeyList
    '''

    return sorted(list(x.keys()))



def sorted_keys(x):

    '''
    same as numkeys(), only for name clarity purpose
    '''

    return sorted(list(x.keys()))



def add_subdict(d, s):

    '''
    add key to empty subdict if not yet part of dict
    
    Args:
      d dict
      s key
    
    Returns:
      d incl key spointing to empty subdict
    '''

    if not (s in d):
        d[s] = {}
    return d



def stopgo(x=''):

    '''
    for debugging
    wait until <return>
    
    Args:
      x - optional message
    '''

    z = input(x)
    return



def file_collector(d, e=''):

    '''
    returns files incl full path as list (recursive dir walk)
    
    Args:
      d - string, directory; or dict containing fields 'dir' and 'ext'
      e - string, extension; or <''> if d dict
    
    Returns:
      ff - list of fullPath-filenames
    '''

    if type(d) is dict:
        pth = d['dir']
        ext = d['ext']
    else:
        pth = d
        ext = e

    ff = []
    for root, dirs, files in os.walk(pth):
        files.sort()
        for f in files:
            if f.endswith(ext):
                ff.append(os.path.join(root, f))
    return sorted(ff)



def optmap(opt, maps):

    '''
    renames key names for intercompatibility of opt dictionaries
    
    Args: dict opt
        dict maps oldName -> newName
    
    Returns: dict mopt with replaced keynames
    '''

    mopt = {}
    for key in maps:
        mopt[maps[key]] = opt[key]
    return mopt


def outl_rm(y, opt):

    '''
    removes outliers in 1-dim array
    
    Args:
      y - 1dim array
      opt: 'f' -> factor of min deviation
           'm' -> from 'mean' or 'median'
    
    Returns:
      z - y without outliers
    '''

    opt['zi'] = False
    oi = outl_idx(y, opt)
    mask = np.ones(len(y), np.bool)
    mask[oi] = False
    return y[mask]



def outl_idx(y, opt):

    '''
    marks outliers in arrayreturns idx of outliers
    
    Args:
    y - numeric array
    opt - 'f' -> factor of min deviation
          'm' -> from 'mean', 'median' or 'fence'
              (mean: m +/- f*sd,
               median: med +/- f*iqr,
               fence: q1-f*iqr, q3+f*iqr)
          'zi' -> true|false - ignore zeros in m and outlier calculation
    
    Returns:
    io - indices of outliers
    '''

    if opt['zi'] == True:
        # i = (y!=0).nonzero()
        i = (y != 0 & np.isfinite(y)).nonzero()
    else:
        # i = range(np.size(y))
        i = (np.isfinite(y)).nonzero()

    f = opt['f']

    if np.size(i) == 0:
        return ea()

    # getting lower and upper boundary lb, ub
    if opt['m'] == 'mean':
        # mean +/- f*sd
        m = np.mean(y.take(i))
        r = np.std(y.take(i))
        lb, ub = m-f*r, m+f*r
    else:
        m = np.median(y.take(i))
        q1, q3 = np.percentile(y.take(i), [25, 75])
        r = q3 - q1
        if opt['m'] == 'median':
            # median +/- f*iqr
            lb, ub = m-f*r, m+f*r
        else:
            # Tukey's fences: q1-f*iqr , q3+f*iqr
            lb, ub = q1-f*r, q3+f*r

    if opt['zi'] == False:
        # io = ((y>ub) | (y<lb)).nonzero()
        io = (np.isfinite(y) & ((y > ub) | (y < lb))).nonzero()
    else:
        # io = ((y>0) & ((y>ub) | (y<lb))).nonzero()
        io = (np.isfinite(y) & ((y > 0) & ((y > ub) | (y < lb)))).nonzero()


    # stopgo("m: {}, lb: {}, ub: {}, n: {}".format(trunc2(m),trunc2(lb),trunc2(ub),len(io[0])))
    return io



def cont2class(x, opt):

    '''
    splits continuous values into classes
    
    Args:
      x: array of float
      opt:
        "splitAt": <val>|prct
                  split at values in x or percentiles
        "bin":
           myCategory: [myLowerThreshold, myUpperTreshold]
           ! will lead to different feature/class vector lengths
             if featmat is not adjusted accordingly
    
    Returns:
      y: array of strings (keys in opt.bin)
      ii: array of indices of x-values transformed to class labels
    REMARK:
    splitting: greedy from below
    values outside all bins are not returned, thus output array
    maybe shorter than input. Sync related feature matrices by ii
    '''

    opt = opt_default(opt, {"splitAt": "val", "bin": {}, "out": ""})

    # prct -> val
    if opt["splitAt"] == "val":
        bins = opt["bin"]
    else:
        bins = {}
        for c in opt["bin"]:
            bins[c] = list(np.percentile(x, opt["bin"][c]))

    # sort classes by lower bin bound
    cc = list(opt["bin"].keys())
    for i in range(len(cc)-1):
        for j in range(1, len(cc)):
            if bins[cc[j]][0] < bins[cc[i]][0]:
                b = cc[j]
                cc[j] = cc[i]
                cc[i] = b

    # float -> class
    y = []
    ii = eai()
    for i in idx(x):
        for c in cc:
            if (x[i] >= bins[c][0] and x[i] <= bins[c][1]):
                y.append(c)
                ii = np.append(ii, i)
                break
    return y, ii



def iqr(x):

    '''
    returns interquertile range
    
    Args:
      x array
    
    Returns:
      iqr scalar
    '''

    return np.diff(np.percentile(x, [25, 75]))


def output_wrapper(v, f, typ, opt={'sep': ',', 'header': True}):

    '''
    output wrapper, extendable, so far only working for typ 'pickle'
    'TextGrid' and 'list' (1-dim)
    
    Args:
      anyVariable (for pickle; dict type for json, csv, csv_quote; TextGrid/list/string
      fileName
      typ: 'pickle'|'TextGrid'|'json'|'list'|'list1line'|'string'|'csv'|'csv_quote'
      opt dict used by some output types, e.g. 'sep', 'header' for csv
    
    Returns:
      <fileOutput>
    '''

    if typ == 'pickle':
        m = 'wb'
    else:
        m = 'w'
    if typ == 'pickle':
        with open(f, m) as h:
            pickle.dump(v, h)
            h.close()
    elif typ == 'TextGrid':
        o_tg(v, f)
    elif re.search('xml', typ):
        o_copa_xml(v, f)
    elif re.search('(string|list|list1line)', typ):
        if typ == 'list':
            x = "\n".join(v)
            x += "\n"
        elif typ == 'list1line':
            x = " ".join(v)
            x += "\n"
        else:
            x = v
        h = open(f, mode='w', encoding='utf-8')
        h.write(x)
        h.close()
    elif typ == 'json':
        with open(f, m) as h:
            json.dump(v, h, indent="\t", sort_keys=True)
            h.close()
    elif typ == 'csv':
        if re.search('\.csv$', f):
            pd.DataFrame(v).to_csv("{}".format(f), na_rep='NA', index_label=False,
                                   index=False, sep=opt['sep'], header=opt['header'])
        else:
            pd.DataFrame(v).to_csv("{}.csv".format(f), na_rep='NA', index_label=False,
                                   index=False, sep=opt['sep'], header=opt['header'])
    elif typ == 'csv_quote':
        if re.search('\.csv$', f):
            pd.DataFrame(d).to_csv("{}".format(f), na_rep='NA', index_label=False, index=False,
                                   quoting=csv.QUOTE_NONNUMERIC, sep=opt['sep'],
                                   header=opt['header'])
        else:
            pd.DataFrame(d).to_csv("{}.csv".format(f), na_rep='NA', index_label=False, index=False,
                                   quoting=csv.QUOTE_NONNUMERIC, sep=opt['sep'],
                                   header=opt['header'])



def o_tg(tg, fil):

    '''
    TextGrid output of dict read in by i_tg()
    (appended if file exists, else from scratch)
    
    Args:
      tg dict
      f fileName
    
    Returns:
      intoFile
    '''

    h = open(fil, mode='w', encoding='utf-8')
    idt = '    '
    fld = tg_fields()
    # head
    if tg['format'] == 'long':
        h.write("File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n")
        h.write("xmin = {}\n".format(tgv(tg['head']['xmin'], 'xmin')))
        h.write("xmax = {}\n".format(tgv(tg['head']['xmax'], 'xmax')))
        h.write("tiers? <exists>\n")
        h.write("size = {}\n".format(tgv(tg['head']['size'], 'size')))
    else:
        h.write("File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n")
        h.write("{}\n".format(tgv(tg['head']['xmin'], 'xmin')))
        h.write("{}\n".format(tgv(tg['head']['xmax'], 'xmax')))
        h.write("<exists>\n")
        h.write("{}\n".format(tgv(tg['head']['size'], 'size')))

    # item
    if (tg['format'] == 'long'):
        h.write("item []:\n")

    for i in numkeys(tg['item']):
        # subkey := intervals or points?
        if re.search(tg['item'][i]['class'], 'texttier', re.I):
            subkey = 'points'
        else:
            subkey = 'intervals'
        if tg['format'] == 'long':
            h.write("{}item [{}]:\n".format(idt, i))
        for f in fld['item']:
            if tg['format'] == 'long':
                if f == 'size':
                    h.write("{}{}{}: size = {}\n".format(
                        idt, idt, subkey, tgv(tg['item'][i]['size'], 'size')))
                else:
                    h.write("{}{}{} = {}\n".format(
                        idt, idt, f, tgv(tg['item'][i][f], f)))
            else:
                h.write("{}\n".format(tgv(tg['item'][i][f], f)))

        # empty tier
        if subkey not in tg['item'][i]:
            continue
        for j in numkeys(tg['item'][i][subkey]):
            if tg['format'] == 'long':
                h.write("{}{}{} [{}]:\n".format(idt, idt, subkey, j))
            for f in fld[subkey]:
                if (tg['format'] == 'long'):
                    h.write("{}{}{}{} = {}\n".format(idt, idt, idt,
                            f, tgv(tg['item'][i][subkey][j][f], f)))
                else:
                    h.write("{}\n".format(tgv(tg['item'][i][subkey][j][f], f)))
    h.close()


def tg_fields():

    '''
    returns field names of TextGrid head and items
    
    Returns:
      hol fieldNames
    '''

    return {'head': ['xmin', 'xmax', 'size'],
            'item': ['class', 'name', 'xmin', 'xmax', 'size'],
            'points': ['time', 'mark'],
            'intervals': ['xmin', 'xmax', 'text']}


def tgv(v, a):

    '''
    rendering of TextGrid values
    
    Args:
      s value
      s attributeName
    
    Returns:
      s renderedValue
    '''

    if re.search('(xmin|xmax|time|size)', a):
        return v
    else:
        return "\"{}\"".format(v)



def tg_tier(tg, tn):

    '''
    returns tier subdict from TextGrid
    
    Args:
      tg: dict by i_tg()
      tn: name of tier
    
    Returns:
      t: dict tier (deepcopy)
    '''

    if tn not in tg['item_name']:
        return {}
    return cp.deepcopy(tg['item'][tg['item_name'][tn]])



def tg_tn(tg):

    '''
    returns list of TextGrid tier names
    
    Args:
      tg: textgrid dict
    
    Returns:
      tn: sorted list of tiernames
    '''

    return sorted(list(tg['item_name'].keys()))



def tg_tierType(t):

    '''
    returns tier type
    
    Args:
      t: tg tier (by tg_tier())
    
    Returns:
      typ: 'points'|'intervals'|''
    '''

    for x in ['points', 'intervals']:
        if x in t:
            return x
    return ''



def tg_txtField(typ):

    '''
    returns text field name according to tier type
    
    Args:
      typ: tier type returned by tg_tierType(myTier)
    
    Returns:
      'points'|<'text'>
    '''

    if typ == 'points':
        return 'mark'
    return 'text'



def tg_mau2chunk(tg, tn='MAU', cn='CHUNK', cl='c'):

    '''
    creates chunk tier (interpausal units) from MAU tier in TextGrid
    
    Args:
      tg: textgrid dict
      tn: MAUS tier name <'MAU'>
      cn: CHUNK tier name <'CHUNK'>
      cl: chunk label <'c'>
    
    Returns:
      c: chunk tier dict
    REMARK:
      c can be added to tg by myl.tg_add(tg,c)
    '''

    pau = '<p:>'
    # MAU tier
    t = tg_tier(tg, tn)
    # t = tg['item'][tg['item_name'][tn]]
    # CHUNK tier
    k = 'intervals'
    c = {'size': 0, 'name': cn, k: {}}
    for x in ['xmin', 'xmax', 'class']:
        c[x] = t[x]
    # t onset, idx in chunk tier
    to, j = 0, 1
    kk = numkeys(t[k])
    for i in kk:
        t1 = trunc4(t[k][i]['xmin'])
        t2 = trunc4(t[k][i]['xmax'])
        if t[k][i]['text'] == pau:
            if to < t1:
                c[k][j] = {'xmin': to, 'xmax': t1, 'text': cl}
                j += 1
            c[k][j] = {'xmin': t1, 'xmax': t2, 'text': pau}
            j += 1
            to = t2
    # final non-pause segment
    if t[k][kk[-1]]['text'] != pau:
        c[k][j] = {'xmin': to, 'xmax': t[k][kk[-1]]['xmax'], 'text': cl}
        j += 1
    # size
    c['size'] = j-1
    return c



def tg_mrg(tgf, tiers, opt={}):

    '''
    merge tiers from >=1 TextGrid in tgf to new TextGrid
    
    Args:
      tgf: string or list of TextGrid files
      tiers: string or list of tier names (contained in one of these files)
      opt: output TextGrid specs
          'format': <'long'>|'short'
          'name': <'mrg'>
    
    Returns:
      tg_mrg: mergedTextGrid
    REMARKS:
      - tier order in tg_mrg is defined by order in tiers
      - header is taken over from first TextGrid in tgf and from opt
      - if a tier occurs in more than one TextGrid, only its occurrence
        in the first one is considered
    '''

    opt = opt_default(opt, {'format': 'long', 'name': 'mrg'})
    if type(tgf) == str:
        tgf = [tgf]
    if type(tiers) == str:
        tiers = [tiers]

    # d: myIdx -> 'tg': myTg, 'tn': [tierNames]
    d = {}
    for i in idx(tgf):
        d[i] = {'tg': i_tg(tgf[i])}
        d[i]['tn'] = tg_tn(d[i]['tg'])

    # output TextGrid
    tg_mrg = {'type': 'TextGrid',
              'format': opt['format'],
              'name': opt['name'],
              'head': cp.deepcopy(d[0]['tg']['head']),
              'item_name': {},
              'item': {}}
    tg_mrg['head']['size'] = 0
    # collect tiers
    for x in tiers:
        for i in numkeys(d):
            if x not in d[i]['tn']:
                continue
            tier_x = tg_tier(d[i]['tg'], x)
            tg_mrg = tg_add(tg_mrg, tier_x, {'repl': True})
            break

    return tg_mrg


def nan_repl(x, mp={}):

    '''
    replaces NA... values in list according to mapping in map
    
    Args:
      x: 1-dim list
      mp: mapping dict (default NA, NaN, nan -> np.nan)
    
    Returns:
      x: NANs, INFs etc replaced
    '''

    if len(mp.keys()) == 0:
        mp = {'NA': np.nan, 'NaN': np.nan, 'nan': np.nan}
    x = np.array(x)
    for z in mp:
        x[x == z] = mp[z]
    return x



def nan2mean(x, v="median"):

    '''
    replaces NaN and INF by median values
    
    Args:
      x: vector
      v: value by which to replace, float or string <"median">,
    
    Returns:
      y: vector with replaced NaN
    '''

    inan = find(x, 'is', 'nan')
    iinf = find(x, 'is', 'inf')
    if max(len(inan), len(iinf)) == 0:
        return x
    ifi = find(x, 'is', 'finite')
    if type(v) is not str:
        m = v
    elif v == "mean":
        m = np.mean(x[ifi])
    else:
        m = np.median(x[ifi])
    if len(inan) > 0:
        x[inan] = m
    if len(iinf) > 0:
        x[iinf] = m

    return x



def input_wrapper(f, typ, opt={}):

    '''
    input wrapper, extendable, so far working for
    'json', 'pickle', 'TextGrid', 'tab'/'lol', 'par', 'xml', 'l_txt', 'string'
    'lol_txt', 'seg_lab', 'list','csv', 'copa_csv', 'pandas_csv', 'key_val',
    'glove'
      xml is restricted to copa_xml format
      *csv: recommended: 'pandas_csv', since 'csv' treats all values as strings
      and 'copa_csv' is more explicit than 'pandas_csv' in type def but
      might get deprecated
      Current diffs between copa_csv and pandas_csv: idx columns as ci, fi etc
      and  numeric speakerIds will be treated as integer in pandas_csv but as
      strings in cop_csv
    
    Args:
       fileName
       typ
       opt <{}> additional options
         'sep': <None>, resp ',' separator
         'to_str': <False> (transform all entries in pandas dataframe (returned as dict) to strings)
    
    Returns:
       containedVariable
    REMARKS:
       diff between lol and lol_txt:
       lol: numeric 2-dim np.array (text will be ignored)
       lol_txt: any 2-dim array
       l_txt: 1-dim array splitting input text at blanks
       seg_lab: on off label
       i_numpy: output of np.loadtxt (1- or 2-dim array)
       csv: returns dict with keys defined by column titles
       csv: NA, NaN, Inf ... are kept as strings. In some
           context need to be replaced by np.nan... (e.g. machine learning)
           use myl.nan_repl() for this purpose
       key_val: split each 2-element row into key-value pairs in dict z
           e.g. a;b -> z["a"]="b"
       glove: word2vec[myWord] = myVec
    '''

    # opt: col separator, string conversion
    if 'sep' in opt:
        sep = opt['sep']
    else:
        if typ == 'i_numpy':
            sep = None
        else:
            sep = ','
    opt = opt_default(opt, {'to_str': False})

    # string
    if typ == 'string':
        with open(f, 'r') as h:
            return h.read()

    # 1-dim list of rows
    if typ == 'list':
        return i_list(f)
    # 1-dim nparray of floats
    if typ == 'i_numpy':
        return np.loadtxt(f, delimiter=sep)
    # TextGrid
    if typ == 'TextGrid':
        return i_tg(f)
    # xml
    if typ == 'xml':
        return i_copa_xml(f)
    # csv into dict (BEWARE: everything is treaten as strings!)
    if typ == 'csv':
        o = {}
        for row in csv.DictReader(open(f, 'r'), delimiter=sep):
            for a in row:
                if a not in o:
                    o[a] = []
                o[a].append(row[a])
        return o
    # copa csv into dict (incl. type conversion)
    if typ == 'copa_csv':
        o = {}
        for row in csv.DictReader(open(f, 'r'), delimiter=sep):
            for a in row:
                if a not in o:
                    o[a] = []
                if copa_categ_var(a):
                    v = row[a]
                else:
                    try:
                        v = float(row[a])
                    except ValueError:
                        if row[a] == 'NA':
                            v = np.nan
                        else:
                            v = row[a]
                o[a].append(v)
        return o
    # csv with pandas: automatic type guessing
    if typ == 'pandas_csv':
        o = pd.read_csv(f, sep=sep, engine='python')
        if opt['to_str']:
            o = o.astype(str)
        return o.to_dict('list')
    # glove input format: word vec
    if typ == "glove":
        o = {}
        with open(f, encoding='utf-8') as h:
            for z in h:
                u = z.split()
                o[u[0]] = np.asarray(u[1:len(u)]).astype(float)
        return o
    # par
    if typ == 'par_as_tab':
        return i_par(f)
    if typ == 'pickle':
        m = 'rb'
    else:
        m = 'r'
    if (typ == 'lol' or typ == 'tab'):
        return lol(f, opt)
    # 1 dim list of blanksep items
    if typ == 'l_txt':
        return i_lol(f, sep='', frm='1d')
    # 2 dim list
    if typ == 'lol_txt':
        return i_lol(f, sep=sep)
    # d.t = [t] or [[on off]], d.lab = [label ...]
    if typ == 'seg_lab':
        return i_seg_lab(f)
    if typ == 'key_val':
        return i_keyVal(f, opt["sep"])
    # json, pickle
    with open(f, m) as h:
        if typ == 'json':
            return json.load(h)
        elif typ == 'pickle':
            return pickle.load(h)
    return False



def is_pau(s, lab=''):

    '''
    returns True if string s is empty or equal pause label lab
    
    Args:
     s someString
     lab pauseLabel <''>
    
    Returns:
     boolean
    '''

    if re.search('^\s*$', s) or s == lab:
        return True
    return False



def copa_categ_var(x):

    '''
    decides whether name belongs to categorical variable
    '''

    if ((x in lists('factors', 'set')) or
        re.search('^(grp|lab|class|spk|tier)', x) or
        re.search('_(grp|lab|class|tier)$', x) or
        re.search('_(grp|lab|class|tier)_', x) or
            re.search('is_(init|fin)', x)):
        return True
    return False



def copa_reference_var(x):

    '''
    decides whether name belongs to reference variable (time, bv etc)
    '''

    if re.search("(t_on|t_off|bv|dur)", x):
        return True
    return False


def copa_drop_var(x):
    if copa_categ_var(x) or copa_reference_var(x):
        return True
    if re.search("_qb", x):
        return True
    return False


def copa_unifChannel(d):

    '''
    replaces multiple channel rhy columns to single one that
    contains values of respective channel idx+1
    example row i:
      ci=0
      SYL_1_rate = 4
      SYL_2_rate = 0
     -> SYL_rate = 4
    !! BEWARE: only works if channel-tier relations are expressed by index=ci+1 !!
    
    Args:
     d: featdict
    
    Returns:
     y: featdict with unified columns
    '''

    y, z = {}, {}
    for x in d:
        if re.search('[12]_', x):
            # key and subkey in z
            k = re.sub('_*[12]_', '_', x)
            if k not in z:
                z[k] = {}
            if re.search('1_', x):
                sk = 0
            else:
                sk = 1
            y[k] = []
            z[k][sk] = d[x]
        else:
            y[x] = d[x]
    # over unified columns
    for k in z:
        # over rows in y
        for i in idx(y['fi']):
            # channel idx -> key in z
            ci = int(y['ci'][i])
            y[k].append(z[k][ci][i])

    return y



def copa_opt_dynad(task, opt):

    '''
    dynamically adjusts 'fsys' part on copa options based on input
    requires uniform config format as in wrapper_py/config/ids|hgc.json
    
    Args:
      task: subfield in opt[fsys][config] pointing to copa opt json file
      opt: in which to find this subfield
    
    Returns:
      copa_opt: copasul options with adjusted fsys specs
    '''


    # read copa config file
    copa_opt = input_wrapper(opt['fsys']['config'][task], 'json')

    # adjust fsys specs according to opt
    for d in ['aud', 'f0', 'annot', 'export', 'pic']:
        if d in opt['fsys']['data']:
            copa_opt['fsys'][d]['dir'] = opt['fsys']['data'][d]
    copa_opt['fsys']['export']['stm'] = opt['fsys']['data']['stm']
    copa_opt['fsys']['pic']['stm'] = opt['fsys']['data']['stm']

    return copa_opt



def copa_read_export(opt):

    '''
    reads copasul output dataframe into dict, given copa config
    
    Args:
       opt: copa options
    
    Returns:
       df: copa feature dict
    '''

    f = os.path.join(opt['fsys']['export']['dir'],
                     "{}.summary.csv".format(opt['fsys']['export']['stm']))
    return input_wrapper(f, "copa_csv")



def rlim(x, r):

    '''
    gets two range vectors, and limits first range to second
    
    Args:
      x: range to be limited [on off]
      r: limiting range [on off]
    
    Returns:
      x: limited range [on off]
    '''

    if x[0] < r[0]:
        x[0] = r[0]
    if x[1] > r[1]:
        x[1] = r[1]
    return x



def i_keyVal(f, sep=''):

    '''
    returns dict z: z["a"]="b" from rows as a;b
    rows with length <2 are skipped
    '''

    z = dict()
    with open(f, encoding='utf-8') as h:
        for u in h:
            if len(sep) == 0:
                x = u.split()
            else:
                u = str_standard(u, True)
                x = u.split(sep)
            if len(x) < 2:
                continue
            z[x[0]] = x[1]
        h.close()
    return z


def i_seg_lab(f, sep=''):

    '''
    [on off label] rows converted to 2-dim np.array and label list
    
    Args:
      f - file name
      sep - <''> separator,
          regex need to be marked by 'r:sep'
          e.g.: 'r:\\t+' - split at tabs (double backslash needed!)
    
    Returns:
      d.t [[on off]...]
       .lab [label ...]
    '''

    if re.search('^r:', sep):
        is_re = True
        sep = re.sub('^r:', '', sep)
    else:
        is_re = False
    d = {'t': ea(), 'lab': []}
    js = ' '
    with open(f, encoding='utf-8') as h:
        for z in h:
            # default or customized split
            if len(sep) == 0:
                x = z.split()
            else:
                # whitspace standardizing
                z = str_standard(z, True)
                # regex or string split
                if is_re:
                    x = re.split(sep, z)
                else:
                    x = z.split(sep)
            x[0] = float(x[0])
            x[1] = float(x[1])
            d['t'] = push(d['t'], x[0:2])
            if len(x) >= 3:
                d['lab'].append(js.join(x[2:len(x)]))
            else:
                d['lab'].append('')
        h.close()
    return d



def trunc2(n):

    '''
    truncates float n to precision 2
    '''

    return float('%.2f' % (n))



def trunc4(n):

    '''
    truncates float n to precision 4
    '''

    return float('%.4f' % (n))



def trunc6(n):

    '''
    truncates float n to precision 6
    '''

    return float('%.6f' % (n))



def counter(x):

    '''
    counting events from input list
    
    Args:
      x - list of strings
    
    Returns:
      c - dict
       myType -> myCount
      n - number of tokens in x
    '''

    c = {}
    for y in x:
        if y not in c:
            c[y] = 1
        else:
            c[y] += 1
    return c, len(x)



def count2prob(c, n):

    '''
    counts to MLE probs
    
    Args:
       c - counter dict from counter()
       n - number of tokens from counter()
    
    Returns:
       p - dict
        myType -> myProb
    '''

    p = {}
    for x in c:
        p[x] = c[x]/n
    return p



def bg_counter(x):

    '''
    bigram counter
    
    Args:
      x - list of strings
    
    Returns:
      c - dict
        c[x_i][x_i-1]=count
    '''

    c = {}
    for i in range(1, len(x)):
        if x[i] not in c:
            c[x[i]] = {}
        if x[i-1] not in c[x[i]]:
            c[x[i]][x[i-1]] = 0
        c[x[i]][x[i-1]] += 1
    return c


def prob2entrop(p):

    '''
    probs to entropy from prob DICT
    
    Args:
       p - dict: myType -> myProb from count2prob()
    
    Returns:
       h - unigramEntropy
    '''

    h = 0
    for x in p:
        if p[x] == 0:
            continue
        h -= (p[x] * math.log(p[x], 2))
    return h



def problist2entrop(pp, b=2):

    '''
    probs to entropy from prob LIST
    
    Args:
       p - list: [myProb ...]
       b - <2> log base
    
    Returns:
       h - entropy
    '''

    h = 0
    for p in pp:
        if p == 0:
            continue
        h -= (p * math.log(p, b))
    return h



def prob2irad(p_in, q_in):

    '''
    information radius of two probability distributions
    
    Args:
       p - dict: myType -> myProb from count2prob()
       q - same
    
    Returns:
       r - information radius between p and q
    '''

    p = cp.deepcopy(p_in)
    q = cp.deepcopy(q_in)

    # mutual filling of probmodels
    for e in set(p.keys()):
        if e not in q:
            q[e] = 0
    for e in set(q.keys()):
        if e not in p:
            p[e] = 0

    r = 0
    for e in set(p.keys()):
        m = (p[e]+q[e])/2
        if m == 0:
            continue
        if p[e] == 0:
            a = q[e]*binlog(q[e]/m)
        elif q[e] == 0:
            a = p[e]*binlog(p[e]/m)
        else:
            a = p[e]*binlog(p[e]/m)+q[e]*binlog(q[e]/m)
        r += a
    return r


def binlog(x):

    '''
    shortcut to log2
    '''

    return math.log(x, 2)


def list2prob(x, pm=None):

    '''
    wrapper around counter() and count2prob()
    alternatively to be called with unigram language model
        -> assigns MLE without smoothing
    
    Args:
      x - list or set of strings
      pm <None> unigram language model
    
    Returns:
      p - dict
        myType -> myProb
    '''

    if type(pm) is None:
        c, n = counter(x)
        return count2prob(c, n)
    p = {}
    for z in x:
        p[z] = prob(z, pm)
    return p



def prob(x, pm):

    '''
    simply assigns mle prob or 0
    
    Args:
      x: string
      pm: unigram probmod
    
    Returns:
      prob
    '''

    if x in pm:
        return pm[x]
    return 0



def list2entrop(x):

    '''
    wrapper around counter() and count2prob(), prob2entrop()
    
    Args:
      x - list of strings
    
    Returns:
      myUnigramEntropy
    '''

    c, n = counter(x)
    p = count2prob(c, n)
    return prob2entrop(p)



def uniq(x):

    '''
    returns sorted + unique element list
    
    Args:
      x - list
    
    Returns:
      x - list unique
    '''

    return sorted(list(set(x)))



def precRec(c):

    '''
    returns precision, recall, and F1
    from input dict C
    
    Args:
      c: dict with count for 'hit', 'ans', 'ref'
    
    Returns:
      val: dict
         'precision', 'recall', 'f1'
    '''

    p = c['hit']/c['ans']
    r = c['hit']/c['ref']
    f = 2*(p*r)/(p+r)
    return {'precision': p, 'recall': r, 'f1': f}


def cellwise(f, x):

    '''
    apply scalar input functions to 1d or 2d arrays
    
    Args:
      f - function
      x - array variable
    '''

    if len(x) == 0:
        return x
    vf = np.vectorize(f)
    # 1d input
    if np.asarray(x).ndim == 1:
        return vf(x)
    # 2d input
    return np.apply_along_axis(vf, 1, x)


def lol(f, opt={}):

    '''
    outputs also one-line inputs as 2 dim array
    
    Args: fileName or list
        opt:
          'colvec' <FALSE> +/- enforce to return column vector
                (for input files with 1 row or 1 column
                 FALSE returns [[...]], and TRUE returns [[.][.]...])
    
    Returns: numpy 2-dim array
    BEWARE
    '''

    opt = opt_default(opt, {'colvec': False})

    if (type(f) is str and (not is_file(f))):
        sys.exit(f, ": file does not exist.")

    try:
        x = np.loadtxt(f)
    except:
        x = f

    if x.ndim == 1:
        x = x.reshape((-1, x.size))
    x = np.asarray(x)

    if opt['colvec'] and len(x) == 1:
        x = np.transpose(x)

    return x



def robust_median(x, opt):

    '''
    returns robust median from 1- or 2-dim array
    
    Args:
      x array
      opt: 'f' -> factor of min deviation
           'm' -> from 'mean' or 'median'
    
    Returns:
      m median scalar or vector
    '''

    nc = ncol(x)
    # 1-dim array
    if nc == 1:
        return np.median(outl_rm(x, opt))
    # 2-dim
    m = ea()
    for i in range(nc):
        m = np.append(m, np.median(outl_rm(x[:, i], opt)))
    return m



def ncol(x):

    '''
    returns number of columns
    1 if array is one dimensional
    
    Args:
      x: 2-dim array
    
    Returns:
      numfOfColumns
    '''

    if np.ndim(x) == 1:
        return 1
    return len(x[0, :])



def nrow(x):

    '''
    returns number of rows
    1 for 1-dim array
    
    Args:
      x: array
    
    Returns:
      numfOfRows
    '''

    if np.ndim(x) == 1:
        return 1
    return len(x)



def df(f, col):

    '''
    outputs a data frame assigning each column a title
    
    Args:
       s F   - fileName
       l COL - colNames
    
    Returns:
       dataFrame DF with content of x
    '''

    x = lol(f)
    col_names = {}
    for i in range(0, len(x[0, :])):
        col_names[col[i]] = x[:, i]
    df = pd.DataFrame(col_names)
    return df



def dfe(x):

    '''
    returns dir, stm, ext from input
    
    Args:
      x: fullPathString
    
    Returns:
      dir
      stm
      ext
    '''

    dd = os.path.split(x)
    d = dd[0]
    s = os.path.splitext(os.path.basename(dd[1]))
    e = s[1]
    e = re.sub('\.', '', e)
    return d, s[0], e


def stm(f):

    '''
    returns file name stem
    
    Args:
      f fullPath/fileName
    
    Returns:
      s stem
    '''

    s = os.path.splitext(os.path.basename(f))[0]
    return s



def repl_dir(f, d, e=''):

    '''
    replaces path (and extension) and keeps stem (and extension)
    
    Args:
      f: 'my/dir/to/file.ext'
      d: 'my/new/dir'
      e: myNewExt <''>
    
    Returns:
      n: 'my/new/dir/file.ext' / 'my/new/dir/file.myNewExt'
    '''

    dd, stm, ext = dfe(f)
    if len(e) > 0:
        ext = e
    return os.path.join(d, "{}.{}".format(stm, ext))


def repl_ext(f, ext=""):

    '''
    replaces extension
    
    Args:
      f: 'my/dir/to/file.ext'
      ext: newext, if empty only dir/stm returned
    
    Returns:
      f: 'my/dir/to/file.newext'
    '''

    d, stm, e = dfe(f)
    if len(ext) == 0:
        return os.path.join(d, stm)
    return os.path.join(d, "{}.{}".format(stm, ext))



def nrm_vec(x, opt):

    '''
    normalizes vector x according to
      opt.mtd|(rng|max|min)
    mtd: 'minmax'|'zscore'|'std'
      'minmax' - normalize to opt.rng
      'zscore' - z-transform
      'std' - divided by std (whitening)
    
    Args:
      x: (np.array)
      opt: (dict) keys: 'mtd', 'rng', 'max', 'min'
    
    Returns:
      x normalized
    '''

    #ok
    
    if opt['mtd'] == 'minmax':
        r = opt['rng']
        if 'max' in opt:
            ma = opt['max']
        else:
            ma = max(x)
        if 'min' in opt:
            mi = opt['min']
        else:
            mi = min(x)
        if ma > mi:
            x = (x - mi) / (ma - mi)
            x = r[0] + x * (r[1] - r[0])
    elif opt['mtd'] == 'zscore':
        x = st.zscore(x)
    elif opt['mtd'] == 'std':
        x = sc.whiten(x)
    return x



def nrm(x, opt):

    '''
    normalizes scalar to range opt.min|max set to opt.rng
    supports minmax only
    '''

    if opt['mtd'] == 'minmax':
        mi = opt['min']
        ma = opt['max']
        r = opt['rng']
        if ma > mi:
            x = (x-mi)/(ma-mi)
            x = r[0] + x*(r[1]-r[0])
    return x



def wav_int2float(s):

    '''
    maps integers from -32768 to 32767 to interval [-1 1]
    '''

    # return nrm_vec(s,{'mtd':'minmax','min':-32768,'max':32767,'rng':[-1,1]})
    s = s/32768
    s[find(s, '<', -1)] = -1
    s[find(s, '>', 1)] = 1
    return s



def nrm_zero_set(t, opt):

    '''
    normalisation of T to range specified in vector RANGE
    opt
        .t0  zero is placed to value t0
        .rng [min max] val for t nrmd, must span interval
    RNG must span interval including 0
    '''

    if len(t) == 0:
        return t
    # t halves
    t1 = t[find(t, '<=', opt['t0'])]
    t2 = t[find(t, '>', opt['t0'])]

    if len(t1) == 0 or len(t2) == 0:
        return nrm_vec(t, {'mtd': 'minmax', 'rng': opt['rng']})

    # corresponding ranges
    r1 = [opt['rng'][0], 0]
    r2 = [opt['rng'][1]/len(t2), opt['rng'][1]]

    # separate normalisations for t-halves
    o = {}
    o['mtd'] = 'minmax'
    o['rng'] = r1
    t1n = nrm_vec(t1, o)

    o['rng'] = r2
    t2n = nrm_vec(t2, o)

    return np.concatenate((t1n, t2n))



def idx_a(l, sts=1):

    '''
    returns index array for vector of length len() l
    thus highest idx is l-1
    '''

    #ok
    
    return np.arange(0, l, sts)


def idx_seg(on, off, sts=1):

    '''
    returns index array between on and off (both included)
    '''

    #ok

    return np.arange(on, off+1, sts)



def idx(l):

    '''
    returns index iterable of list L
    '''

    #ok
    
    return range(len(l))


def cp_dict(a, b):

    '''
    copy key-value pairs from dict A to dict B
    '''

    for x in list(a.keys()):
        b[x] = a[x]
    return b



def ndim(x):

    '''
    returns dimension of numpy array
    
    Args:
      array
    
    Returns:
      int for number of dims
    '''

    return len(x.shape)



def tg_tier2tab(t, opt={}):

    '''
    transforms TextGrid tier to 2 arrays
    point -> 1 dim + lab
    interval -> 2 dim (one row per segment) + lab
    
    Args:
      t: tg tier (by tg_tier())
      opt dict
          .skip <""> regular expression for labels of items to be skipped
                if empty, only empty items will be skipped
    
    Returns:
      x: 1- or 2-dim array of time stamps
      lab: corresponding labels
    REMARK:
      empty intervals are skipped
    '''

    opt = opt_default(opt, {"skip": ""})
    if len(opt["skip"]) > 0:
        do_skip = True
    else:
        do_skip = False
    x = ea()
    lab = []
    if 'intervals' in t:
        for i in numkeys(t['intervals']):
            z = t['intervals'][i]
            if len(z['text']) == 0:
                continue
            if do_skip and re.search(opt["skip"], z["text"]):
                continue

            x = push(x, [z['xmin'], z['xmax']])
            lab.append(z['text'])
    else:
        for i in numkeys(t['points']):
            z = t['points'][i]
            if do_skip and re.search(opt["skip"], z["mark"]):
                continue
            x = push(x, z['time'])
            lab.append(z['mark'])
    return x, lab


def tg_tab2tier(t, lab, specs):

    '''
    transforms table to TextGrid tier
    
    Args:
       t - numpy 1- or 2-dim array with time info
       lab - list of labels <[]>
       specs['class'] <'IntervalTier' for 2-dim, 'TextTier' for 1-dim>
            ['name']
            ['xmin'] <0>
            ['xmax'] <max tab>
            ['size'] - will be determined automatically
            ['lab_pau'] - <''>
    
    Returns:
       dict tg tier (see i_tg() subdict below myItemIdx)
    for 'interval' tiers gaps between subsequent intervals will be bridged
    by lab_pau
    '''

    tt = {'name': specs['name']}
    nd = ndim(t)
    # 2dim array with 1 col
    if nd == 2:
        nd = ncol(t)
    # tier class
    if nd == 1:
        tt['class'] = 'TextTier'
        tt['points'] = {}
    else:
        tt['class'] = 'IntervalTier'
        tt['intervals'] = {}
        # pause label for gaps between intervals
        if 'lab_pau' in specs:
            lp = specs['lab_pau']
        else:
            lp = ''
    # xmin, xmax
    if 'xmin' not in specs:
        tt['xmin'] = 0
    else:
        tt['xmin'] = specs['xmin']
    if 'xmax' not in specs:
        if nd == 1:
            tt['xmax'] = t[-1]
        else:
            tt['xmax'] = t[-1, 1]
    else:
        tt['xmax'] = specs['xmax']
    # point tier content
    if nd == 1:
        for i in idx_a(len(t)):
            # point tier content might be read as [[x],[x],[x],...] or [x,x,x,...]
            if of_list_type(t[i]):
                z = t[i, 0]
            else:
                z = t[i]
            tt['points'][i+1] = {'time': z, 'mark': lab[i]}
        tt['size'] = len(t)
    # interval tier content
    else:
        j = 1
        # initial pause
        if t[0, 0] > tt['xmin']:
            tt['intervals'][j] = {'xmin': tt['xmin'],
                                  'xmax': t[0, 0], 'text': lp}
            j += 1
        for i in idx_a(len(t)):
            # pause insertions
            if ((j-1 in tt['intervals']) and
                    t[i, 0] > tt['intervals'][j-1]['xmax']):
                tt['intervals'][j] = {'xmin': tt['intervals'][j-1]['xmax'],
                                      'xmax': t[i, 0], 'text': lp}
                j += 1
            tt['intervals'][j] = {'xmin': t[i, 0],
                                  'xmax': t[i, 1], 'text': lab[i]}
            j += 1
        # final pause
        if tt['intervals'][j-1]['xmax'] < tt['xmax']:
            tt['intervals'][j] = {'xmin': tt['intervals'][j-1]['xmax'],
                                  'xmax': tt['xmax'], 'text': lp}
            j += 1  # so that uniform 1 subtraction for size
        # size
        tt['size'] = j-1
    return tt



def tg_add(tg, tier, opt={'repl': True}):

    '''
    add tier to TextGrid
    
    Args:
      tg dict from i_tg(); can be empty dict
      tier subdict to be added:
          same dict form as in i_tg() output, below 'myItemIdx'
      opt
         ['repl'] <True> - replace tier of same name
    
    Returns:
      tg updated
    REMARK:
      !if generated from scratch head xmin and xmax are taken over from the tier
       which might need to be corrected afterwards!
    '''


    # from scratch
    if 'item_name' not in tg:
        fromScratch = True
        tg = {'name': '', 'format': 'long', 'item_name': {}, 'item': {},
              'head': {'size': 0, 'xmin': 0, 'xmax': 0, 'type': 'ooTextFile'}}
    else:
        fromScratch = False

    # tier already contained?
    if (opt['repl'] == True and (tier['name'] in tg['item_name'])):
        i = tg['item_name'][tier['name']]
        tg['item'][i] = tier
    else:
        # item index
        ii = numkeys(tg['item'])
        if len(ii) == 0:
            i = 1
        else:
            i = ii[-1]+1
        tg['item_name'][tier['name']] = i
        tg['item'][i] = tier
        tg['head']['size'] += 1

    if fromScratch and 'xmin' in tier:
        for x in ['xmin', 'xmax']:
            tg['head'][x] = tier[x]

    return tg



def inter2tg(an):

    '''
    transform interchange format to TextGrid
    transforms table to TextGrid tier
    
    Args:
       an: annot dict e.g. by i_par() or i_copa_xml()
    
    Returns:
       tg: TextGrid dict
    for 'interval' tiers gaps between subsequent intervals are bridged
    only tiers with time information are taken over!
    '''

    typeMap = {'segment': 'IntervalTier', 'event': 'TextTier'}
    itemMap = {'segment': 'intervals', 'event': 'points'}
    tg = {'type': 'TextGrid', 'format': 'long',
          'head': {'xmin': 0, 'xmax': -1, 'size': 0},
          'item_name': {}, 'item': {}}
    # item idx
    ii = 1
    # over tiers
    for x in sorted(an.keys()):
        # skip tier without time info
        if an[x]['type'] not in typeMap:
            continue
        tg['head']['size'] += 1
        tg['item_name'][x] = ii
        tg['item'][ii] = {'name': x, 'size': 0, 'xmin': 0, 'xmax': -1,
                          'class': typeMap[an[x]['type']]}
        z = itemMap[an[x]['type']]
        # becomes tg['item'][ii]['points'|'intervals']
        tt = {}
        # point or interval tier content
        if z == 'points':
            # j: tier items (j+1 in TextGrid output)
            for j in numkeys(an[x]['items']):
                y = an[x]['items'][j]
                tt[j+1] = {'time': y['t'],
                           'mark': y['label']}
                tg['item'][ii]['size'] += 1
                tg['item'][ii]['xmax'] = y['t']
        else:
            j = 1
            # initial pause
            y = an[x]['items'][0]
            if y['t_start'] > 0:
                tt[j] = {'xmin': tg['item'][ii]['xmin'],
                         'xmax': y['t_start'], 'text': ''}
                j += 1
            # i: input tier idx, j: output tier idx
            for i in numkeys(an[x]['items']):
                y = an[x]['items'][i]
                # pause insertions
                if ((j-1 in tt) and
                        y['t_start'] > tt[j-1]['xmax']):
                    tt[j] = {'xmin': tt[j-1]['xmax'],
                             'xmax': y['t_start'], 'text': ''}
                    j += 1
                tt[j] = {'xmin': y['t_start'],
                         'xmax': y['t_end'], 'text': y['label']}
                tg['item'][ii]['xmax'] = tt[j]['xmax']
                j += 1

            # size
            tg['item'][ii]['size'] = j-1

        # copy to interval/points subdict
        tg['item'][ii][z] = tt

        # xmax
        tg['head']['xmax'] = max(tg['head']['xmax'], tg['item'][ii]['xmax'])
        ii += 1

    # uniform xmax, final silent interval
    for ii in tg['item']:
        # add silent interval
        if (tg['item'][ii]['class'] == 'IntervalTier' and
                tg['item'][ii]['xmax'] < tg['head']['xmax']):
            tg['item'][ii]['size'] += 1
            j = max(tg['item'][ii]['intervals'])+1
            xm = tg['item'][ii]['intervals'][j-1]['xmax']
            tg['item'][ii]['intervals'][j] = {'text': '', 'xmin': xm,
                                              'xmax': tg['head']['xmax']}
        tg['item'][ii]['xmax'] = tg['head']['xmax']
    return tg


def par2tg(par_in):

    '''
    as inter2tg() but omitting header item
    
    Args:
      par dict from i_par()
    
    Returns:
      tg dict as with i_tg()
    '''

    par = cp.deepcopy(par_in)
    del par['header']
    return inter2tg(par)


def tg_item_keys(t):

    '''
    returns item-related subkeys: 'intervals', 'text' or 'points', 'mark'
    
    Args:
      t tier
    
    Returns:
      x key1
      y key2
    '''

    if 'intervals' in t:
        return 'intervals', 'text'
    return 'points', 'mark'


def tg2par(tg, fs):

    '''
    wrapper around tg2inter() + adding 'header' item / 'class' key
    WARNING: information loss! MAU tier does not contain any wordIdx reference!
    
    Args:
     tg: dict read by tg_in
     fs: sample rate
    
    Returns:
     par: par dict (copa-xml format)
    REMARK: output cannot contain wordIdx refs!
       thus MAU is class 2 in this case, without 'i' field
    '''

    par = tg2inter(tg)
    # add 'class' key
    for tn in par:
        par[tn]['class'] = par_class(par, tn)
    par['header'] = par_header(fs)
    return par



def par_class(par, tn):

    '''
    returns class of a par tier to be added to dict coming from tg2inter()
    
    Args:
     par: par dict
     tn: tierName
    
    Returns:
     c: tierClass
    '''

    t = par[tn]['type']
    if t == 'null':
        return 1
    n = numkeys(par[tn]['items'])
    if 'i' in par[tn]['items'][n[0]]:
        wordRef = True
    else:
        wordRef = False
    if t == 'event':
        if wordRef:
            return 5
        return 3
    else:
        if wordRef:
            return 4
        return 2



def par_header(fs):

    '''
    returns header dict for par dict
    
    Args:
      fs: sample rate
    
    Returns:
      h: header dict
    '''

    return {'type': 'header', 'class': 0, 'items': {'SAM': fs}}



def tg2inter(tg, opt={}):

    '''
    transforms textgrid dict to interchange format
     same as i_par(), i_copa_xml() output
    
    Args:
      tg dict from i_tg
      [opt]:
        snap: <False> (if True, also empty-label intervals are kept)
    
    Returns:
      an dict:
    event tier:
      dict [myTierName]['type'] = 'event'
                       ['items'][itemIdx]['label']
                                         ['t']
                                         ['class']
    segment tier
           [myTierName]['type'] = 'segment'
                       ['items'][itemIdx]['label']
                                         ['t_start']
                                         ['t_end']
    '''

    opt = opt_default(opt, {'snap': False})
    an = {}
    # over tier names
    for tn in tg['item_name']:
        t = tg['item'][tg['item_name'][tn]]
        an[tn] = {'items': {}}
        # item idx in an[tn]['items']
        ii = 0
        if 'intervals' in t:
            k = 'intervals'
            an[tn]['type'] = 'segment'
        elif 'point' in t:
            k = 'points'
            an[tn]['type'] = 'event'
        # empty tier
        else:
            continue
        for i in numkeys(t[k]):
            if k == 'intervals':
                if len(t[k][i]['text']) == 0:
                    if not opt['snap']:
                        continue
                an[tn]['items'][ii] = {'label': t[k][i]['text'],
                                       't_start': t[k][i]['xmin'],
                                       't_end': t[k][i]['xmax']}
            else:
                an[tn]['items'][ii] = {'label': t[k][i]['mark'],
                                       't': t[k][i]['time']}
            ii += 1
    return an



def i_tg(ff):

    '''
    TextGrid from file
    
    Args:
      s fileName
    
    Returns:
      dict tg
         type: TextGrid
         format: short|long
         name: s name of file
         head: hoh
             xmin|xmax|size|type
         item_name ->
                 myTiername -> myItemIdx
         item
            myItemIdx ->        (same as for item_name->myTiername)
                 class
                 name
                 size
                 xmin
                 xmax
                 intervals   if class=IntervalTier
                       myIdx -> (xmin|xmax|text) -> s
                 points
                       myIdx -> (time|mark) -> s
    '''

    if i_tg_format(ff) == 'long':
        return i_tg_long(ff)
    else:
        return i_tg_short(ff)



def i_tg_short(ff):

    '''
    TextGrid short format input
    
    Args:
     s fileName
    Out:
     dict tg (see i_tg)
    '''

    tg = {'name': ff, 'format': 'short', 'head': {},
          'item_name': {}, 'item': {}, 'type': 'TextGrid'}
    (key, fld, skip, state, nf) = ('head', 'xmin', True, 'head', tg_nf())
    idx = {'item': 0, 'points': 0, 'intervals': 0}
    with open(ff, encoding='utf-8') as f:
        for z in f:
            z = re.sub('\s*\n$', '', z)
            if re.search('object\s*class', z, re.I):
                fld = nf[state]['#']
                skip = False
                continue
            else:
                if ((skip == True) or re.search('^\s*$', z) or
                        re.search('<exists>', z)):
                    continue
            if re.search('(interval|text)tier', z, re.I):
                if re.search('intervaltier', z, re.I):
                    typ = 'interval'
                else:
                    typ = 'text'
                z = re.sub('"', '', z)
                key = 'item'
                state = 'item'
                fld = nf[state]['#']
                idx[key] += 1
                idx['points'] = 0
                idx['intervals'] = 0
                if not (idx[key] in tg[key]):
                    tg[key][idx[key]] = {}
                tg[key][idx[key]][fld] = z
                if re.search('text', typ, re.I):
                    subkey = 'points'
                else:
                    subkey = 'intervals'
                fld = nf[state][fld]
            else:
                z = re.sub('"', '', z)
                if fld == 'size':
                    z = int(z)
                elif fld in ['xmin', 'xmax', 'time']:
                    z = float(z)
                if state == 'head':
                    tg[key][fld] = z
                    fld = nf[state][fld]
                elif state == 'item':
                    tg[key] = add_subdict(tg[key], idx[key])
                    tg[key][idx[key]][fld] = z
                    if fld == 'name':
                        tg['item_name'][z] = idx[key]
                    # last fld of item reached
                    if nf[state][fld] == '#':
                        state = subkey
                        fld = nf[state]['#']
                    else:
                        fld = nf[state][fld]
                elif re.search('(points|intervals)', state):
                    # increment points|intervals idx if first field adressed
                    if fld == nf[state]['#']:
                        idx[subkey] += 1
                    tg[key][idx[key]] = add_subdict(tg[key][idx[key]], subkey)
                    tg[key][idx[key]][subkey] = add_subdict(
                        tg[key][idx[key]][subkey], idx[subkey])
                    tg[key][idx[key]][subkey][idx[subkey]][fld] = z
                    if nf[state][fld] == '#':
                        fld = nf[state]['#']
                    else:
                        fld = nf[state][fld]
    return tg



def i_tg_long(ff):

    '''
    TextGrid long format input
    
    Args:
     s fileName
    
    Returns:
     dict tg (see i_tg)
    '''

    tg = {'name': ff, 'format': 'long', 'head': {},
          'item_name': {}, 'item': {}}
    (key, skip) = ('head', True)
    idx = {'item': 0, 'points': 0, 'intervals': 0}
    with open(ff, encoding='utf-8') as f:
        for z in f:
            z = re.sub('\s*\n$', '', z)
            if re.search('object\s*class', z, re.I):
                skip = False
                continue
            else:
                if ((skip == True) or re.search('^\s*$', z) or
                        re.search('<exists>', z)):
                    continue
            if re.search('item\s*\[\s*\]:?', z, re.I):
                key = 'item'
            elif re.search('(item|points|intervals)\s*\[(\d+)\]\s*:?', z, re.I):
                m = re.search(
                    '(?P<typ>(item|points|intervals))\s*\[(?P<idx>\d+)\]\s*:?', z)
                i_type = m.group('typ').lower()
                idx[i_type] = int(m.group('idx'))
                if i_type == 'item':
                    idx['points'] = 0
                    idx['intervals'] = 0
            elif re.search('([^\s+]+)\s*=\s*\"?(.*)', z):
                m = re.search('(?P<fld>[^\s+]+)\s*=\s*\"?(?P<val>.*)', z)
                (fld, val) = (m.group('fld').lower(), m.group('val'))
                fld = re.sub('number', 'time', fld)
                val = re.sub('[\"\s]+$', '', val)
                # type cast
                if fld == 'size':
                    val = int(val)
                elif fld in ['xmin', 'xmax', 'time']:
                    val = float(val)
                # head specs
                if key == 'head':
                    tg[key][fld] = val
                else:
                    # link itemName to itemIdx
                    if fld == 'name':
                        tg['item_name'][val] = idx['item']
                    # item specs
                    if ((idx['intervals'] == 0) and (idx['points'] == 0)):
                        tg[key] = add_subdict(tg[key], idx['item'])
                        tg[key][idx['item']][fld] = val
                    # points/intervals specs
                    else:
                        tg[key] = add_subdict(tg[key], idx['item'])
                        tg[key][idx['item']] = add_subdict(
                            tg[key][idx['item']], i_type)
                        tg[key][idx['item']][i_type] = add_subdict(
                            tg[key][idx['item']][i_type], idx[i_type])
                        tg[key][idx['item']][i_type][idx[i_type]][fld] = val
    return tg


def tg_nf():

    '''
    tg next field init
    '''

    return {'head':
            {'#': 'xmin',
             'xmin': 'xmax',
             'xmax': 'size',
             'size': '#'},
            'item':
            {'#': 'class',
             'class': 'name',
             'name': 'xmin',
             'xmin': 'xmax',
             'xmax': 'size',
             'size': '#'},
            'points':
            {'#': 'time',
             'time': 'mark',
             'mark': '#'},
            'intervals':
            {'#': 'xmin',
             'xmin': 'xmax',
             'xmax': 'text',
             'text': '#'}}



def i_tg_format(ff):

    '''
    decides whether TextGrid is in long or short format
    
    Args:
      s textGridfileName
    
    Returns:
      s 'short'|'long'
    '''

    with open(ff, encoding='utf-8') as f:
        for z in f:
            if re.search('^\s*<exists>', z):
                f.close
                return 'short'
            elif re.search('xmin\s*=', z):
                f.close
                return 'long'
    return 'long'



def opt_default(c, d):

    '''
    recursively adds default fields of dict d to dict c
       if not yet specified in c
    
    Args:
     c someDict
     d defaultDict
    
    Returns:
     c mergedDict (defaults added to c)
    '''

    for x in d:
        if x not in c:
            c[x] = d[x]
        elif type(c[x]) is dict:
            c[x] = opt_default(c[x], d[x])
    return c



def list_rm(x, r):

    '''
    removes elements of set R from list X
    
    Args:
     x list
     r string or set of items to be removed
    
    Returns:
     y: x purified
    '''

    y = []
    r = set(r)
    for e in x:
        if e in r:
            continue
        y.append(e)
    return y



def rm_pau(x):

    '''
    removes pause labels from input list x
    '''

    return list_rm(x, {'<p:>', '<p>', '<P>'})



def halx(x, l):

    '''
    length adjustment
    
    Args:
      x: list
      l: required length
    
    Returns:
      x adjusted
     if x is longer than l, x[0:l] is returned
     if x is shorter than l: horizontal extrapolation
    '''

    if len(x) == l:
        return x
    if len(x) > l:
        return x[0:l]

    if len(x) == 0:
        a = 0
    else:
        a = x[-1]
    if type(x) is np.ndarray:
        while len(x) < l:
            x = np.append(x, a)
    else:
        while len(x) < l:
            x.append(a)
    return x



def hal(x, y):

    '''
    hack: length adjustment
    
    Args:
      x list
      y list
    
    Returns:
      x,y shortened to same length if needed
    '''

    if len(x) > len(y):
        x = x[0:len(y)]
    elif len(y) > len(x):
        y = y[0:len(x)]
    return x, y


def hal_old(x, y):
    while (len(x) > len(y)):
        x = x[0:len(x)-1]
    while (len(y) > len(x)):
        y = y[0:len(y)-1]
    return x, y


def check_var(c):

    '''
    diverse variable checks and respective reactions
    
    Args:
     c['var'] - any variable or variable container
      ['env'] - main context of variable check, e.g. 'copasul'
      ['spec']- optional, string or dictionary specifying sub-contexts,
                actions, etc.
    '''


    # copasul
    if c['env'] == 'copasul':
        copa = c['var']
        s = c['spec']

        # clustering
        if s['step'] == 'clst':
            dom = s['dom']
            if ((dom not in copa['clst']) or
                ('c' not in copa['clst'][dom]) or
                    len(copa['clst'][dom]['c']) == 0):
                sys.exit(
                    "ERROR! Clustering of {} contours requires stylization step.\nSet navigate.do_styl_{} to true.".format(dom, dom))

        # styl - dom = gnl|glob|loc|bnd|rhy...
        elif s['step'] == 'styl':
            dom = s['dom']

            # copa['data'][ii][i] available?
            # -> first file+channelIdx (since in file/channel-wise augmentation steps does not start from
            #       copa is initialized for single varying ii and i)
            ii, i, err = check_var_numkeys(copa)
            if err:
                sys.exit(
                    "ERROR! {} feature extraction requires preprocessing step.\nSet navigate.do_preproc to true".format(dom))

            # preproc fields available?
            if check_var_copa_preproc(copa, ii, i):
                sys.exit(
                    "ERROR! {} feature extraction requires preprocessing step.\nSet navigate.do_preproc to true".format(dom))

            # domain field initialization required? (default True)
            if (('req' not in s) or (s['req'] == True)):
                if check_var_copa_dom(copa, dom, ii, i):
                    sys.exit(
                        "ERROR! {} feature extraction requires preprocessing step.\nSet navigate.do_preproc to true".format(dom))

            # dependencies on other subdicts
            if 'dep' in s:
                for x in s['dep']:
                    if check_var_copa_dom(copa, x, ii, i):
                        sys.exit(
                            "ERROR! {} feature extraction requires {} features.\nSet navigate.do_{} to true".format(dom, x, x))

            # ideosyncrasies
            if re.search('^rhy', dom):
                if ('rate' not in copa['data'][ii][i]):
                    sys.exit(
                        "ERROR! {} feature extraction requires an update of the preprocessing step. Set navigate.do_preproc to true".format(dom))
            if dom == 'bnd':
                if ((copa['config']['styl']['bnd']['residual']) and
                        ('r' not in copa['data'][0][0]['f0'])):
                    sys.exit("ERROR! {} feature extraction based on f0 residuals requires a previous global contour stylization so that the register can be subtracted from the f0 contour. Set navigate.do_styl_glob to true, or set styl.bnd.residual to false".format(dom))



def check_var_copa_preproc(copa, ii, i):

    '''
    check blocks called by check_var()
    preproc fields given?
    returns True in case of violation
    '''

    if (('f0' not in copa['data'][ii][i]) or
        ('t' not in copa['data'][ii][i]['f0']) or
            ('y' not in copa['data'][ii][i]['f0'])):
        return True
    return False



def check_var_numkeys(copa):

    '''
    checks whether copa['data'][x][y] exists and returns lowest numkeys and err True|False
    
    Args:
      copa
    
    Returns:
      i: lowest numkey in copa['data']
      j: lowest numkey in copa['data'][i]
      err: True if there is no copa['data'][i][j], else False
    '''

    if type(copa['data']) is not dict:
        return True
    a = numkeys(copa['data'])
    if len(a) == 0 or (type(copa['data'][a[0]]) is not dict):
        return -1, -1, True
    b = numkeys(copa['data'][a[0]])
    if len(b) == 0 or (type(copa['data'][a[0]][b[0]]) is not dict):
        return -1, -1, True
    return a[0], b[0], False



def check_var_copa_dom(copa, dom, ii, i):

    '''
    domain substruct initialized?
    '''

    if dom not in copa['data'][ii][i]:
        return True
    return False



def sig_preproc(y, opt={}):

    '''
    to be extended signal preproc function
    
    Args:
      y signal vector
      opt['rm_dc'] - <True> centralize
    '''

    dflt = {'rm_dc': True}
    opt = opt_default(opt, dflt)
    # remove DC
    if opt['rm_dc'] == True:
        y = y-np.mean(y)

    return y



def prob_oov():

    '''
    returns low number as probability for unseen events
    '''

    return np.exp(-100)


def assign_prob(w, p):

    '''
    assigns prob to event
    
    Args:
     w - event
     p - prob dict (from list2prob())
    
    Returns:
     pw - probability of w
    '''

    if w in p:
        return p[w]
    return prob_oov()


def fsys_stm(opt, s):

    '''
    concats dir and stem of fsys-subdirectory
    
    Args:
      opt with subdir ['fsys'][s][dir|stm]
      s - subdir name (e.g. 'out', 'export' etc
    '''

    fo = os.path.join(opt['fsys'][s]['dir'],
                      opt['fsys'][s]['stm'])
    return fo


def verbose_stm(copa, ii):

    '''
    prints stem of current f0 file in copasul framework
    '''

    if 'data' in copa:
        print(copa['data'][ii][0]['fsys']['f0']['stm'])
    else:
        print(copa[ii][0]['fsys']['f0']['stm'])



def copa_yseg(copa, dom, ii, i, j, t=[], y=[]):

    '''
    returns f0 segment from vector according to copa time specs
    
    Args:
      copa
      dom 'glob'|'loc'
      ii fileIdx
      i channelIdx
      j segmentIdx
      t time vector of channel i <[]>
      y f0 vector of channel i   <[]>
    
    Returns:
      ys f0-segment in segment j
    '''

    if len(t) == 0:
        t = copa['data'][ii][i]['f0']['t']
        y = copa['data'][ii][i]['f0']['y']
    tb = copa['data'][ii][i][dom][j]['t']
    yi = find_interval(t, tb[[0, 1]])
    return y[yi]



def o_copa_xml(an, f):

    '''
    copa xml output (appended if file exists, else from scratch)
    
    Args:
      an - dict generated by i_copa_xml()
      f  - file name
    
    Returns:
      file output xml into f
    '''

    # identation
    i1 = "  "
    i2 = "{}{}".format(i1, i1)
    i3 = "{}{}".format(i2, i1)
    i4 = "{}{}".format(i2, i2)
    i5 = "{}{}".format(i3, i2)
    # subfields for tier type
    fld = {'event': ['label', 't'], 'segment': ['label', 't_start', 't_end']}
    # output
    h = open(f, mode='w', encoding='utf-8')
    h.write("{}\n<annotation>\n{}<tiers>\n".format(xml_hd(), i1))
    # over tiers
    for x in sorted(an.keys()):
        h.write("{}<tier>\n".format(i2))
        h.write("{}<name>{}</name>\n{}<items>\n".format(i3, x, i3))
        # over items
        for i in numkeys(an[x]['items']):
            h.write("{}<item>\n".format(i4))
            for y in fld[an[x]['type']]:
                z = an[x]['items'][i][y]
                if y == 'label':
                    z = xs.escape(z)
                h.write("{}<{}>{}</{}>\n".format(i5, y, z, y))
            h.write("{}</item>\n".format(i4))
        h.write("{}</items>\n{}</tier>\n".format(i3, i2))
    h.write("{}</tiers>\n</annotation>\n".format(i1))
    h.close()
    return



def xml_hd():

    '''
    returns xml header
    '''

    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"


def int2bool(x):

    '''
    maps >0|0 to True|False
    '''

    if x > 0:
        return True
    elif x == 0:
        return False
    else:
        sys.exit("{} cannot be transformed to booelan".format(x))


def retFalse(typ='bool'):

    '''
    returns different forms of False
    
    Args:
      typ: <'bool'>|'str'|'num'
    
    Returns:
      False|''|0
    '''

    if typ == 'str':
        return ''
    if typ == 'num':
        return 0
    return False


def i_list(f):

    '''
    returns file content as 1-dim list of strings (one element per row)
    
    Args:
      filename
    
    Returns:
      list
    '''

    with open(f, encoding='utf-8') as h:
        l = [x.strip() for x in h]
        h.close()
    return l



def i_lol(f, sep='', frm='2d'):

    '''
    reads any text file as 1-dim or 2-dim list (of strings)
    Remark: no np.asarray, thus element indexing is x[i][j],
            i.e. [i,j] does not work
    
    Args:
      s fileName
      sep separator (<''> -> whitespace split)
          regex need to be marked by 'r:sep'
          e.g.: 'r:\\t+' - split at tabs (double backslash needed!)
      frm <'2d'>|'1d'
    
    Returns:
      lol x (sublists: rows splitted at whitespace)
         or l x (rows concat to 1 list)
    '''

    if re.search('^r:', sep):
        is_re = True
        sep = re.sub('^r:', '', sep)
    else:
        is_re = False
    d = []
    with open(f, encoding='utf-8') as h:
        for z in h:
            # default or customized split
            if len(sep) == 0:
                x = z.split()
            else:
                # whitspace standardizing
                z = str_standard(z)
                # regex or string split
                if is_re:
                    x = re.split(sep, z)
                else:
                    x = z.split(sep)
            if len(x) > 0:
                if frm == '2d':
                    d.append(x)
                else:
                    for y in x:
                        d.append(y)
        h.close()
    return d


def str_standard(x, a=False):

    '''
    standardizes strings:
      removes initial, final (and multiple) whitespaces
    
    Args:
      x: someString
      a: <False>; if True: replace all whitespace(-sequences) by single blank
    
    Returns:
      x: standardizedString
    '''

    x = re.sub('^\s+', '', x)
    x = re.sub('\s+$', '', x)
    if a:
        x = re.sub('\s+', ' ', x)
    return x



def o_par(par, f):

    '''
    BAS partiture file output
    
    Args:
      par: dict in i_copa_xml format (read by i_par)
      fil: file name
    
    Returns:
      par file output to f
    Tier class examples
      1: KAS:	0	v i:6
      2: IPA:    4856    1228    @
      3: LBP: 1651 PA
      4: USP:    3678656 14144   48;49   PAUSE_WORD
      5: PRB:    54212    5   TON: H*; FUN: NA
    '''

    # sample rate
    fs = par['header']['items']['SAM']
    # init output list
    o = [["LHD: Partitur 1.3"],
         ["SAM: {}".format(fs)],
         ["LBD:"]]
    for tn in sorted_keys(par):
        if tn == 'header':
            continue
        tier = cp.deepcopy(par[tn])
        # print(sorted(tier.keys()))
        c = tier['class']
        # add to par output list
        for i in numkeys(tier['items']):
            a = o_par_add(tier, tn, i, c, fs)
            # unsnap for segments
            o = o_par_unsnap(o, a, c)
            o.append(a)

    # -> 1dim list
    for i in idx(o):
        # print(o[i])
        o[i] = ' '.join(map(str, o[i]))

    # output
    output_wrapper(o, f, 'list')
    return



def o_par_unsnap(o, a, c):

    '''
    for segment tiers reduces last time offset in o by 1 sample
    if equal to onset in a
    
    Args:
      o: par list output so far
      a: new list to add
      c: tier class
    
    Returns:
      o: last sublist updated
    '''

    # no MAU segment tier
    if c != 4 or a[0] != 'MAU:':
        return o
    # diff tiers
    if o[-1][0] != a[0]:
        return o
    t = o[-1][1]
    dur = o[-1][2]
    if t+dur != a[1]-1:
        o[-1][2] = a[1]-1-t
    return o


def o_par_add(tier, tn, i, c, fs):

    '''
    adds row to par output file
    
    Args:
      tier: tier subdict from par
      tn: current tier name
      i: current item index in tier
      c: tier class 1-5
      fs: sample rate
    '''

    x = tier['items'][i]
    s = "{}:".format(tn)
    if c == 1:
        # no time reference
        a = [s, x['i'], x['label']]
    elif c == 2 or c == 4:
        # segment
        t = int(x['t_start']*fs)
        dur = int((x['t_end']-x['t_start'])*fs)
        if c == 2:
            a = [s, t, dur, x['label']]
        else:
            a = [s, t, dur, x['i'], x['label']]
    elif c == 3 or c == 5:
        # event
        t = int(x['t']*fs)
        if c == 3:
            a = [s, t, x['label']]
        else:
            a = [s, t, x['i'], x['label']]
    return a



def i_par(f, opt={}):

    '''
    BAS partiture file input
    
    Args:
      s fileName
      opt
          snap: <False>
            if True off- and onset of subsequent intervals in MAU are set equal
            (of use for praat-TextGrid conversion so that no empty segments
             are outputted)
          header: <True>
          fs: sample rate, needed if no header provided, i.e. header=False
    OUT (same as i_copa_xml + 'class' for tier class and 'i' for word idx)
    event tier:
      dict [myTierName]['type'] = 'event'
                       ['class'] = 3,5
                       ['items'][itemIdx]['label']
                                         ['t']
                                         ['i']
    segment tier
           [myTierName]['type'] = 'segment'
                       ['class'] = 2,4
                       ['items'][itemIdx]['label']
                                         ['t_start']
                                         ['t_end']
                                         ['i']
    no time info tier
           [myTierName]['type'] = 'null'
                       ['class'] = 1
                       ['items'][itemIdx]['label']
                                         ['i']
    header
           [header]['type'] = 'header'
                   ['class'] = 0
                   ['items']['SAM'] = mySampleRate
    symbolic reference to word idx 'i' is always of type 'str'
       (since it might include [,;])
    for all other tier classes t, t_start, t_end are floats; unit: seconds
    Only tiers defined in dict tt are considered
    Tier class examples
      1: KAS:	0	v i:6
      2: IPA:    4856    1228    322     @
      3: LBP: 1651 PA
      4: USP:    3678656 14144   48;49   PAUSE_WORD
      5: PRB:    54212    5   TON: H*; FUN: NA
    '''

    opt = opt_default(opt, {'snap': False, 'header': True})
    if (opt['header'] == False and ('fs' not in opt)):
        sys.exit('specify sample rate fs for headerless par files.')

    par = dict()
    # tier classes (to be extended if needed)
    tc = {'KAN': 1, 'KAS': 1, 'ORT': 1, 'MAU': 4, 'DAS': 1,
          'PRB': 5, 'PRS': 1, 'LBP': 3, 'LBG': 3, 'PRO': 1,
          'POS': 1, 'TRN': 4, 'TRS': 1, 'PRM': 3, 'MAS': 4}
    # current item idx for each tier
    ii = {}
    # sample rate
    if 'fs' in opt:
        fs = opt['fs']
    else:
        fs = -1
    # start to build dict if True
    if opt['header']:
        headOff = False
    else:
        headOff = True
    # join sep
    js = ' '
    for z in i_lol(f):
        if z[0] == 'SAM:':
            fs = int(z[1])
        elif z[0] == 'LBD:':
            headOff = True
        elif headOff == False or (not re.search(':$', z[0])):
            continue
        tn = re.sub(':$', '', z[0])
        if tn not in tc:
            continue
        if tn not in par:
            if tc[tn] == 1:
                typ = 'null'
            elif tc[tn] in {3, 5}:
                typ = 'event'
            else:
                typ = 'segment'
            par[tn] = {'type': typ, 'items': {}, 'class': tc[tn]}
            ii[tn] = 0

        par[tn]['items'][ii[tn]] = {}
        if tc[tn] == 1:
            par[tn]['items'][ii[tn]]['label'] = js.join(z[2:len(z)])
            par[tn]['items'][ii[tn]]['i'] = z[1]
        elif tc[tn] == 2:
            par[tn]['items'][ii[tn]]['label'] = js.join(z[3:len(z)])
            if opt['snap'] and ii[tn] > 0:
                par[tn]['items'][ii[tn]
                                 ]['t_start'] = par[tn]['items'][ii[tn]-1]['t_end']
            else:
                par[tn]['items'][ii[tn]]['t_start'] = int(z[1])/fs
            par[tn]['items'][ii[tn]]['t_end'] = (int(z[1])+int(z[2])-1)/fs
        elif tc[tn] == 3:
            par[tn]['items'][ii[tn]]['label'] = js.join(z[2:len(z)])
            par[tn]['items'][ii[tn]]['t'] = int(z[1])/fs
        elif tc[tn] == 4:
            par[tn]['items'][ii[tn]]['label'] = js.join(z[4:len(z)])
            if opt['snap'] and ii[tn] > 0:
                par[tn]['items'][ii[tn]
                                 ]['t_start'] = par[tn]['items'][ii[tn]-1]['t_end']
            else:
                par[tn]['items'][ii[tn]]['t_start'] = int(z[1])/fs
            par[tn]['items'][ii[tn]]['t_end'] = (int(z[1])+int(z[2])-1)/fs
            par[tn]['items'][ii[tn]]['i'] = z[3]
        elif tc[tn] == 5:
            par[tn]['items'][ii[tn]]['label'] = js.join(z[3:len(z)])
            par[tn]['items'][ii[tn]]['t'] = int(z[1])/fs
            par[tn]['items'][ii[tn]]['i'] = z[2]
        ii[tn] += 1

    # add header subdict
    par['header'] = par_header(fs)

    return par



def par_mau2wrd(par):

    '''
    returns wrd dict derived from MAU tier
    
    Args:
      par: dict from i_par()
    
    Returns:
      wrd: dict derived from MAU (<{}>)
         ['type'] = 'segment'
         ['class'] = 4
         ['items'][itemIdx]['label']
                           ['t_start']
                           ['t_end']
      [itemIdx] corresponds to MAU... ['i']
    '''

    if 'MAU' not in par:
        return {}
    w = {'type': 'segment', 'class': 4, 'items': {}}
    for i in numkeys(par['MAU']['items']):
        j = int(par['MAU']['items'][i]['i'])
        if j < 0:
            continue
        if j not in w['items']:
            w['items'][j] = {}
            for x in ['label', 't_start', 't_end']:
                w['items'][j][x] = par['MAU']['items'][i][x]
        else:
            w['items'][j]['label'] = w['items'][j]['label'] + \
                ' '+par['MAU']['items'][i]['label']
            w['items'][j]['t_end'] = par['MAU']['items'][i]['t_end']
    return w


def tab2copa_xml_tier(x, lab=[]):

    '''
    transforms 1- or 2dim table to [myTierName]-subdict (see i_copa_xml())
    
    Args:
      x: 1- or 2-dim matrix with time info
      lab: corresp label ([]) if empty uniform label 'x'
    
    Returns:
      y dict
         ['type']: event|segment
         ['items'][itemIdx]['label']
                           ['t'] or ['t_start']
                                    ['t_end']
    call:
      myAnnot[myTierName] = tab2copa_xml_tier(myTab,myLab)
    '''

    if ncol(x) == 2:
        typ = 'segment'
    else:
        typ = 'event'
    tier = {'type': typ, 'items': {}}
    for i in range(len(x)):
        tier['items'][i] = {}
        if len(lab) > i:
            tier['items'][i]['label'] = lab[i]
        else:
            tier['items'][i]['label'] = 'x'
        if typ == 'segment':
            tier['items'][i]['t_start'] = x[i, 0]
            tier['items'][i]['t_end'] = x[i, 1]
        else:
            tier['items'][i]['t'] = x[i, 0]
    return tier



def i_copa_xml(f):

    '''
    copa xml input
    <annotation><tiers><tier><name>
                             <items><item><label>
                                          <t_start>
                                          <t_end>
                                          <t>
    
    Args:
      inputFileName
    
    Returns:
    event tier:
      dict [myTierName]['type'] = 'event'
                       ['items'][itemIdx]['label']
                                         ['t']
    segment tier
           [myTierName]['type'] = 'segment'
                       ['items'][itemIdx]['label']
                                         ['t_start']
                                         ['t_end']
    '''

    annot = {}
    tree = et.parse(f)
    root = tree.getroot()
    tiers = extract_xml_element(root, 'tiers')
    for tier in tiers:
        name_el = extract_xml_element(tier, 'name')
        name = name_el.text
        annot[name] = {'items': {}}
        itemIdx = 0
        for item in tier.iter('item'):
            annot[name]['items'][itemIdx] = {
                'label': extract_xml_element(item, 'label', 'text')}
            t = extract_xml_element(item, 't', 'text')
            if t:
                annot[name]['type'] = 'event'
                annot[name]['items'][itemIdx]['t'] = t
            else:
                annot[name]['type'] = 'segment'
                t_start = extract_xml_element(item, 't_start', 'text')
                t_end = extract_xml_element(item, 't_end', 'text')
                annot[name]['items'][itemIdx]['t_start'] = t_start
                annot[name]['items'][itemIdx]['t_end'] = t_end
            itemIdx += 1
    return annot



def extract_xml_element(myTree, elementName, ret='element'):

    '''
    returns element object or its content string
    
    Args:
      xml.etree.ElementTree (sub-)object
      elementName (string)
      ret 'element'|'string'
    
    Returns:
      elementContent (object or string dep on ret)
    '''

    for e in myTree:
        if e.tag == elementName:
            if ret == 'element':
                return e
            else:
                return e.text
    return False


def myLog(o, task, msg=''):

    '''
    log file opening/closing/writing
    
    Args:
      opt: dict, that needs to contain log file name in o['fsys']['log']
      task: 'open'|'write'|'print'|'close'
      msg: message <''>
    
    Returns:
      for task 'open': opt with new field opt['log'] containing file handle
      else: True
    '''

    if task == 'open':
        h = open(o['fsys']['log'], 'a')
        h.write("\n## {}\n".format(isotime()))
        o['log'] = h
        return o
    elif task == 'write':
        o['log'].write("{}\n".format(msg))
    elif task == 'print':
        o['log'].write("{}\n".format(msg))
        print(msg)
    else:
        o['log'].close()
    return True


def log_exit(f_log, msg):
    f_log.write("{}\n".format(msg))
    print(msg)
    f_log.close()
    sys.exit()



def cntr_classif(c, d, opt={}):

    '''
    nearest centroid classification
    
    Args:
      c dict
         [myClassKey] -> myCentroidVector
      d feature matrix (ncol == ncol(c[myClassKey])
      opt dict
         ['wgt']  -> feature weights (ncol == ncol(c[myClassKey]))
         ['attract'] -> class attractiveness (>0 !)
                     [myClassKey] -> attractiveness <1>
                     distance d to class is multiplied derived from
                     1 - attract/sum(attract)
    
    Returns:
      a list with answers (+/- numeric depending on key type in C)
    '''

    opt = opt_default(opt, {'wgt': np.ones(ncol(d)), 'attract': {}})

    # calculate class prior distance from attractiveness
    cdist = {}
    if len(opt['attract'].keys()) > 0:
        attract_sum = 0
        for x in c:
            if x not in opt['attract']:
                opt['attract'][x] = 1
            attract_sum += opt['attract'][x]
        for x in c:
            cdist[x] = 1-opt['attract'][x]/attract_sum
    else:
        for x in c:
            cdist[x] = 1

    a = []
    # over rows
    for i in range(len(d)):
        # assign class of nearest centroid
        # min dist
        mdst = -1
        for x in c:
            # weighted eucl distance
            dst = dist_eucl(d[i, :], c[x], opt['wgt']) * cdist[x]
            if (mdst < 0 or dst < mdst):
                nc = x
                mdst = dst
        a.append(nc)
    return a



def wgt_mean(x, w):

    '''
    weighted mean
    
    Args:
      x: vector
      w: weight vector of same length
    
    Returns:
      y: weighted mean of x
    '''

    return np.sum(w*x)/np.sum(w)



def feature_weights_unsup(x, mtd='corr'):

    '''
    estimates feature weights for unsupervised learning
    
    Args:
      x: n x m feature matrix, columns-variables, rows-observations
      mtd: <'corr'>, nothing else supported
    OUT
      w: 1 x m weight vector
    Method 'corr':
      - for 'informed clustering' with uniform feature trend, e.g. all
         variables are expected to have high values for cluster A and
         low values for cluster B
      - wgt(feat):
           if corr(feat,rowMeans)<=0: 0
           if corr(feat,rowMeans)>0: its corr
    wgts are normalized to w/sum(w)
    '''

    # row means
    m = np.mean(x, axis=1)
    w = np.asarray([])
    for i in range(ncol(x)):
        r = np.corrcoef(x[:, i], m)
        r = np.nan_to_num(r)
        w = np.append(w, max(r[0, 1], 0))
    # unusable -> uniform weight
    if np.sum(w) == 0:
        return np.ones(len(m))
    return w/np.sum(w)



def dist_eucl(x, y, w=[]):

    '''
    weighted euclidean distance between x and y
    
    Args:
      x: array
      y: array
      w: weight array <[]>
    
    Returns:
      d: distance scalar
    '''

    if len(w) == 0:
        w = np.ones(len(x))
    q = x-y
    # print('!!!',w) #!cl
    # stopgo() #!cl
    # print('q:', q)
    # print('q2:', q*q)
    # print('wq2:', w*q*q)
    # print('max(wq2):', max(w*q*q))
    # print('sum(wq2):', (w*q*q).sum())
    # print('sqrt(sum(wq2)):', np.sqrt((w*q*q).sum()))
    # stopgo()
    return np.sqrt((w*q*q).sum())



def calc_delta(x):

    '''
    calculates delta values x[i,:]-x[i-1,:] for values in x
    
    Args:
     x: feature matrix
    
    Returns:
     dx: delta matrix
    REMARK:
     first delta is calculated by x[0,:]-x[1,:]
    '''

    if len(x) <= 1:
        return x
    dx = np.asarray([x[0, :]-x[1, :]])
    for i in range(1, len(x)):
        dx = push(dx, x[i, :]-x[i-1, :])
    return dx



def cmat(x, y, a=0):

    '''
    table merge
    always returns 2-dim nparray
    input can be 1- or 2-dim
    
    Args:
      x nparray
      y nparray
      a axis <0>  (0: append rows, 1: append columns)
    
    Returns:
      z = [x y]
    '''

    if len(x) == 0 and len(y) == 0:
        return ea()
    elif len(x) == 0:
        return lol(y)
    elif len(y) == 0:
        return lol(x)
    return np.concatenate((lol(x), lol(y)), a)



def non_empty_val(d, x):

    '''
    returns True if
      - x is key in dict d
      - d[x] is non-empty list or string
    '''

    if (x not in d) or len(d[x]) == 0:
        return False
    return True



def ea(n=1):

    '''
    return empty np array(s)
    
    Args:
      n: <1> how many (up to 5)
    
    Returns:
      n x np.asarray([])
    '''

    if n == 1:
        return np.asarray([])
    if n == 2:
        return np.asarray([]), np.asarray([])
    if n == 3:
        return np.asarray([]), np.asarray([]), np.asarray([])
    if n == 4:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
    if n == 5:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
    return



def eai():

    '''
    return empty np integer array
    '''

    return np.array([]).astype(int)


def eai(n=1):

    '''
    return empty np array(s) of type int
    
    Args:
      n: <1> how many (up to 5)
    
    Returns:
      n x np.asarray([])
    '''

    if n == 1:
        return np.asarray([]).astype(int)
    if n == 2:
        return np.asarray([]).astype(int), np.asarray([]).astype(int)
    if n == 3:
        return np.asarray([]).astype(int), np.asarray([]).astype(int), np.asarray([]).astype(int)
    if n == 4:
        return np.asarray([]).astype(int), np.asarray([]).astype(int), np.asarray([]).astype(int), np.asarray([]).astype(int)
    if n == 5:
        return np.asarray([]).astype(int), np.asarray([]).astype(int), np.asarray([]).astype(int), np.asarray([]).astype(int), np.asarray([]).astype(int)
    return



def robust_split(x, p):

    '''
    robustly splits vector in data above '>' and below '<=' value
    so that always two non-empty index vectors are returned (except of empty
    list input)
    
    Args:
     x: vector
     p: splitValue
    
    Returns:
     i0, i1: 2 index vectors of items in x <= and > p
    '''

    if len(x) == 1:
        return 0, 0
    if len(x) == 2:
        return min_i(x), max_i(x)
    else:
        for px in [p, np.mean(x)]:
            i0 = find(x, '<=', px)
            i1 = find(x, '>', px)
            if min(len(i0), len(i1)) > 0:
                return i0, i1
    i = np.arange(0, len(x))
    return i, i


def str2logic(x):

    '''
    replace strings (read e.g. from json) to logic constants
    
    Args:
      x string
    
    Returns:
      y logic constant (if available)
    '''

    if not is_type(x) == 'str':
        return x
    if not re.search('(TRUE|FALSE|NONE)', x):
        return x
    elif x == 'TRUE':
        return True
    elif x == 'FALSE':
        return False
    return None


def is_type(x):

    '''
    quick hack to return variable type str|int|float|list (incl np array)
    fallback 'unk'
    
    Args:
      x var
    
    Returns:
      t type of x
    '''

    tx = str(type(x))
    for t in ['str', 'int', 'float', 'list']:
        if re.search(t, tx):
            return t
    if re.search('ndarray', tx):
        return 'list'
    return 'unk'



def max_kv(h):

    '''
    returns key, value pair from dict with maximum value
    (first one found in alphabetically sorted key list)
    
    Args:
      h - dict with numeric values
    
    Returns:
      k - key with max value
      v - max value
    '''

    k, v = np.nan, np.nan
    for x in sorted_keys(h):
        if np.isnan(v) or h[x] > v:
            k, v = x, h[x]
    return k, v



def min_i(x):

    '''
    returns index of first minimum in list or nparray
    
    Args:
      x: list or nparray
    
    Returns:
      i: index of 1st minimum
    '''

    return list(x).index(min(x))



def max_i(x):

    '''
    returns index of first maximum in list or nparray
    
    Args:
      x: list or nparray
    
    Returns:
      i: index of 1st maximum
    '''

    return list(x).index(max(x))



def rob0(x):

    '''
    robust 0 for division, log, etc
    '''

    if x == 0:
        return 0.00000001
    return x



def of_list_type(x):

    '''
    returns True if x is list or np.array
    '''

    if (type(x) is list) or (type(x) is np.ndarray):
        return True
    return False



def aslist(x):

    '''
    returns any input as list
    
    Args:
     x
    
    Returns:
     x if list, else [x]
    '''

    if type(x) is list:
        return x
    return [x]



def two_dim_array(x):

    '''
    1-dim -> 2-dim array
    
    Args:
      x: 1-dim nparray
    
    Returns:
      y: 2-dim array
    '''

    return x.reshape((-1, x.size))



def trs_pattern(x, pat, lng):

    '''
    identifies transcription major classes: 'vow'
    
    Args:
      x: phoneme string
      pat: pattern type 'vow'
      lng: language 'deu'
    
    Returns:
      True if x contains pat, else False
    '''

    ref = {'vow': {'deu': '[aeiouyAEIOUY269@]'}}
    if re.search(ref[pat][lng], x):
        return True
    return False



def root_path():

    '''
    returns '/'
    '''

    return os.path.abspath(os.sep)



def myHome():

    '''
    returns home directory
    '''

    return str(Path.home())



def g2p(opt):

    '''
    wrapper around g2p (local or webservice)
    
    Args:
      opt dict
        all parameters accepted by g2p.pl
       + local True|False (if False, webservice is called)
    
    Returns:
      success True|False
    '''


    if opt['local']:
        cmd = "/homes/reichelu/repos/repo_pl/src/prondict/g2p.pl -task apply"
    else:
        cmd = "curl -v -X POST -H 'content-type: multipart/form-data'"

    for x in opt:
        if (re.search('local', x) or (x == 'o' and (not opt['local']))):
            continue
        y = ''
        if x == 'i':
            y = '@'
        if opt['local']:
            z = " -{} {}".format(x, opt[x])
        else:
            z = " -F {}={}{}".format(x, y, opt[x])
        cmd += z

    if not opt['local']:
        cmd += " 'http://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runG2P'"

    return webservice_output(cmd, opt)


def bas_webservice(opt, o):

    '''
    call any BAS webservice and write result in output file
    
    Args:
      opt:
        'tool' -> service url
        'param' -> parameter dict (param names as required by webservice)
                 !! files to be uploaded already need to have initial @
      o: output file name
    
    Returns:
      success: True|False
      file: sevice output into o
    '''

    cmd = "curl -v -X POST -H 'content-type: multipart/form-data'"

    # concatenate curl call
    for x in opt['param']:
        cmd += " -F {}={}".format(x, opt['param'][x])
    cmd += " {}".format(opt['tool'])

    # call webservice + output results
    return webservice_output(cmd, {'out': o, 'local': False})


def webmaus(opt):

    '''
    wrapper around webmaus general
    
    Args:
      opt dict
        .signal: signalFileName
        .bpf: par file name
        .out: par output file
        .lng: language iso
        .outformat 'TextGrid'|<'mau-append'>
        .insorttextgrid <true>
        .inskantextgrid <true>
        .usetrn <false>
        .modus <standard>|align   (formerly: canonly true|false)
        .weight <default>
        .outipa <false>
        .minpauslen <5>
        .insprob <0.0>
        .mausshift <10.0>
        .noinitialfinalsilence <false>
        .local <False>; if true maus is called not as webservice
    
    Returns:
        sucess True|False
        par file written to opt.par_out
    '''

    opt = opt_default(opt, {'outformat': 'mau-append',
                            'insorttextgrid': 'true',
                            'inskantextgrid': 'true',
                            'usetrn': 'false',
                            'modus': 'standard',
                            'weight': 'default',
                            'outipa': 'false',
                            'minpauslen': 5,
                            'insprob': 0,
                            'mausshift': 10.0,
                            'noinitialfinalsilence': 'false',
                            'local': False})
    optmap = {'OUTFORMAT': 'outformat',
              'INSORTTEXTGRID': 'insorttextgrid',
              'INSKANTEXTGRID': 'inskantextgrid',
              'USETRN': 'usetrn',
              'STARTWORD': 'startword',
              'ENDWORD': 'endword',
              'INSPROB': 'insprob',
              'OUTFORMAT': 'outformat',
              'MAUSSHIFT': 'mausshift',
              'OUTIPA': 'outipa',
              'MINPAUSLEN': 'minpauslen',
              'MODUS': 'modus',
              'LANGUAGE': 'lng',
              'BPF': 'bpf',
              'WEIGHT': 'weight',
              'SIGNAL': 'signal',
              'NOINITIALFINALSILENCE': 'noinitialfinalsilence'}

    if opt['local']:
        cmd = "maus"
    else:
        cmd = "curl -v -X POST -H 'content-type: multipart/form-data'"

    for x in optmap:
        if optmap[x] in opt:
            y = ''
            if re.search('^(BPF|SIGNAL)$', x):
                y = '@'
            if opt['local']:
                z = " {}={}".format(x, opt[optmap[x]])
            else:
                z = " -F {}={}{}".format(x, y, opt[optmap[x]])

            cmd += z

    if opt['local']:
        cmd += " > {}".opt['out']
    else:
        cmd += " 'http://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUS'"

    return webservice_output(cmd, opt)


def webservice_output(cmd, opt):

    '''
    collect output from g2p or MAUS
    write into file opt['out'], resp opt['o']
    
    Args:
      cmd: service call string
      opt:
        fatal -> <True>|False exit if unsuccesful
        local -> True|False
        o(ut) -> outPutFile
    
    Returns:
      (if not exited)
        True if succesful, else false
    '''


    opt = opt_default(opt, {'fatal': False})

    print(cmd)
    ans = subprocess.check_output(cmd, shell=True, universal_newlines=True)
    print(ans)

    # local call -> done
    if opt['local']:
        return

    # webservice answer parse
    m = re.search('<downloadLink>(?P<res>(.+?))<\/downloadLink>', ans)

    if m is None:
        if opt['fatal']:
            sys.exit("error: {}".format(ans))
        else:
            print("error: {}".format(ans))
            return False

    res = m.group('res')
    if res is None:
        if opt['fatal']:
            sys.exit("error: {}".format(ans))
        else:
            print("error: {}".format(ans))
            return False

    # unifying across maus and g2p
    if 'out' in opt:
        o = opt['out']
    else:
        o = opt['o']

    os.system("wget {} -O {}".format(res, o))

    return True


def i_eaf(f, opt={}):

    '''
    reads eaf files
    
    Args:
     f: fileName
     opt: <{}>
        .min_l: minimum segment length in sec
                (some elan files contain blind segments)
        .embed: <'none'>,'igc',... makro to cope with corpus-specific issues
    
    Returns:
     eaf dict
        .myTierName
          .i: tierIndex (starting with 0)
          .type: 'segment' (so far only segment supported!)
          .removed: set()  ANNOTATION_IDs that are removed depending on embed
          .items
            .myItemIdx
              .'label'
              .'t_start' in sec
              .'t_end'   in sec
              .'ts_start' - timeSloptID
              .'ts_end'
              .'id' annot Id
    '''

    opt = opt_default(opt, {'min_l': 0, 'embed': 'none'})
    eaf = {}
    d = et.parse(f)
    root = d.getroot()
    # time stamps dict: id -> time in sec
    ts = i_eaf_ts(root)
    tier_i = 0
    for xe in root:
        if xe.tag != 'TIER':
            continue
        ae = xe.attrib
        # tier name
        tn = ae['TIER_ID']
        eaf[tn] = {'type': 'segment', 'i': tier_i,
                   'items': {}, 'removed': set()}
        tier_i += 1
        # item index
        i = 0
        for xa in xe:
            if xa.tag == 'ANNOTATION':
                for xaa in xa:
                    if xaa.tag == 'ALIGNABLE_ANNOTATION':
                        aa = xaa.attrib
                        t_start = ts[aa['TIME_SLOT_REF1']]
                        t_end = ts[aa['TIME_SLOT_REF2']]
                        aid = aa['ANNOTATION_ID']
                        # vs blind elan segments
                        if t_end - t_start < opt['min_l']:
                            eaf[tn]['removed'].add(aid)
                            continue
                        xav = extract_xml_element(xaa, 'ANNOTATION_VALUE')
                        lab = xav.text
                        if lab is None:
                            lab = ''
                        if opt['embed'] == 'igc':
                            lab = re.sub('\s*\-\-+\s*', ', ', lab)
                            # lab = re.sub('\s*\?\?+\s*','',lab)
                            # lab = re.sub('\(.+?\)','',lab)
                            lab = re.sub('\s*\?\?+\s*', ' <usb> ', lab)
                            lab = re.sub('\(.+?\)', ' <usb> ', lab)
                            lab = string_std(lab)
                            # remove | sep at start and end of text
                            lab = re.sub('^\s*\|', '', lab)
                            lab = re.sub('\|\s*$', '', lab)
                            if len(lab) == 0:
                                eaf[tn]['removed'].add(aid)
                                continue
                        eaf[tn]['items'][i] = {'label': lab,
                                               't_start': t_start,
                                               't_end': t_end,
                                               'ts_start': aa['TIME_SLOT_REF1'],
                                               'ts_end': aa['TIME_SLOT_REF2'],
                                               'id': aid}
                        i += 1
                        # print(t_start,t_end,lab)
                        # stopgo()
    return eaf



def i_eaf_ts(root):

    '''
    returns eaf time stamp dict
    
    Args:
      eaf xml root object
    
    Returns:
      ts dict
        timeStamp -> timeValue (onset already added, in sec)
    '''

    ts = {}
    # time onset (in ms)
    ons = i_eaf_ons(root)
    tsx = extract_xml_element(root, 'TIME_ORDER')
    for e in tsx:
        if e.tag == 'TIME_SLOT':
            a = e.attrib
            ts[a['TIME_SLOT_ID']] = (int(a['TIME_VALUE'])+ons)/1000
    return ts



def i_eaf_ons(root):

    '''
    returns eaf time onset from header
    
    Args:
      eaf xml root object
    
    Returns:
      int onset (in ms)
    '''

    hd = extract_xml_element(root, 'HEADER')
    md = extract_xml_element(hd, 'MEDIA_DESCRIPTOR')
    mda = md.attrib
    return int(mda['TIME_ORIGIN'])



def string_std(s):

    '''
    standardise strings, i.e. remove initial and final blanks
    and replace multiple by single blanks
    
    Args:
      s string
    
    Returns:
      s standardized string
    '''

    s = re.sub('\s+', ' ', s)
    s = re.sub('^\s+', '', s)
    s = re.sub('\s+$', '', s)
    return s



def purge_dir(d):

    '''
    empty directory by removing + regenerating
    '''

    if os.path.isdir(d):
        sh.rmtree(d)
        os.makedirs(d)

    return True


def rm_nan(x):

    '''
    remove nan from array
    
    Args:
      x - 1-dim array
    
    Returns:
      x without nan
    '''

    return x[~np.isnan(x)]



def linint_nan(y, r='i'):

    '''
    linear interpolation over NaN
    
    Args:
      y - np array 1-dim
      r - 'i'|'e'
          what to return if only NaNs contained:
           <'i'> - input list
           'e' - empty list
    
    Returns:
      y with NaNs replaced by linint
        (or input if y consists of NaNs only)
    '''

    xi = find(y, 'is', 'nan')

    # no NaN
    if len(xi) == 0:
        return y

    # only NaN
    if len(xi) == len(y):
        if r == 'i':
            return y
        return ea()

    # interpolate
    xp = find(y, 'is', 'finite')
    yp = y[xp]
    yi = np.interp(xi, xp, yp)
    y[xi] = yi
    return y



def list_switch(l, i, j):

    '''
    switches elements with idx i and j in list l
    
    Args:
      l list
      i, j indices
    
    Returns:
      l updated
    '''

    bi = cp.deepcopy(l[i])
    l[i] = cp.deepcopy(l[j])
    l[j] = bi
    return l



def copasul_call(c, as_sudo=False):

    '''
    executes copasul.py from other scripts
    
    Args:
       c config filename (full path)
       as_sudo: boolean, if True: sudo-call
    
    Returns:
       as specified in c
    '''

    d = os.path.dirname(os.path.realpath(__file__))
    cmd = os.path.join(d, 'copasul.py')
    if as_sudo:
        cmd = "sudo {}".format(cmd)
    os.system("{} -c {}".format(cmd, c))



def as_sudo(opt):

    '''
    returns True if opt dict contains as_sudo key with value True
    '''

    if ext_true(opt, 'as_sudo'):
        return True
    return False



def ext_false(d, k):

    '''
    extended not:
    returns True if (key not in dict) OR (dict.key is False)
    
    Args:
     d: dict
     k: key
    
    Returns:
     b: True|False
    '''

    if ((k not in d) or (not d[k])):
        return True
    return False



def ext_true(d, k):

    '''
    returns True if (key in dict) AND (dict.key is True)
    
    Args:
     d: dict
     k: key
    
    Returns:
     b: True|False
    '''

    if (k in d) and d[k]:
        return True
    return False



def pipeline_copasul_call(opt, x):

    '''
    calls copasul in pipeline
    
    Args:
      opt pipeline config
             must contain ['fsys']['config'][myKey]
             with location of copasul config file
      myKey key in opt['fsys']['config'] to be adressed
    
    Returns:
      -
    '''

    copasul_call(opt['fsys']['config'][x])



def colors():

    '''
    
    Returns: list of default 10 item color circle + some additional ones
    '''

    return ['blue', 'orange', 'green', 'red', 'purple',
            'brown', 'pink', 'gray', 'olive', 'cyan',
            'lime', 'darkgreen', 'maroon', 'salmon',
            'darkcyan', 'powderblue', 'lightgreen']



def timeSort(x, t):

    '''
    sorts elements in list x according to their time points in list t
    
    Args:
      x - list of elements to be sorted
      t - list of their time stamps
    
    Returns:
      x sorted
    '''

    for i in range(0, len(x)-1):
        for j in range(i+1, len(x)):
            if t[i] > t[j]:
                b = x[j]
                x[j] = x[i]
                x[i] = b
    return x



def is_punc(s):

    '''
    returns True if input string is punctuation mark
    '''

    if re.search('^[<>,!\|\.\?:\-\(\);\"\']+$', s):
        return True
    return False



def lng_map(task='expand'):

    '''
    task: 'expand'|'shorten'
      from/to iso<->fullform
    so far for all languages supported by snowball stemmer
    '''

    lm = {'nld': 'dutch',
          'dan': 'danish',
          'eng': 'english',
          'fin': 'finnish',
          'fra': 'french',
          'deu': 'german',
          'hun': 'hungarian',
          'ita': 'italian',
          'nor': 'norwegian',
          'porter': 'porter',
          'por': 'portuguese',
          'ron': 'romanian',
          'rus': 'russian',
          'spa': 'spanish',
          'swe': 'swedish'}
    if task == 'expand':
        return lm
    else:
        lmr = {}
        for x in lm:
            lmr[lm[x]] = x
        return lmr



def lists2condEntrop(x, y):

    '''
    returns condition entropy H(y|x) for input lists x and y
    
    Args:
     x
     y
    
    Returns:
     H(y|x)
    '''

    c = lists2count(x, y)
    return count2condEntrop(c)



def count2condEntrop(c):

    '''
    
    Args:
      c: output of lists2count()
        e.g. c['joint'][x][y]
    
    Returns:
      H(Y|X) (beware order!)
    '''

    h = 0
    for x in c['joint']:
        nx = c['margin'][x]
        if nx == 0:
            continue
        pys = 0
        for y in c['joint'][x]:
            py = c['joint'][x][y]/nx
            if py == 0:
                continue
            pys += (py*binlog(py))
        h -= ((nx/c['N'])*pys)
    return h



def lists2redun(x, y):

    '''
    returns redundancy given two input lists
    
    Args:
     x
     y
    
    Returns:
     r = I(x;y)/(H(x)+H(y)
    '''

    ixy = count2mi(lists2count(x, y))
    hx = list2entrop(x)
    hy = list2entrop(y)
    return ixy/(hx+hy)



def lists2mi(x, y):

    '''
    returns mutual infor of 2 input lists
    
    Args:
      x: list
      y: list
    
    Returns:
      mi: mutual info
    '''

    return count2mi(lists2count(x, y))



def lists2count(x, y):

    '''
    returns dict with 2 layers for cooc counts in 2 input lists
    
    Args:
      x
      y (same lengths!)
    
    Returns:
      c['joint'][x][y]=myCount f.a. x,y in <xs, ys>
       ['margin_x'][x]=myCount
       ['margin_y'][y]=myCount
       ['N']: len(x)
    '''

    c, n = {'joint': {}, 'margin_x': {}, 'margin_y': {}}, 0
    for i in idx_a(len(x)):
        if x[i] not in c['margin_x']:
            c['margin_x'][x[i]] = 0
        if y[i] not in c['margin_y']:
            c['margin_y'][y[i]] = 0
        if x[i] not in c['joint']:
            c['joint'][x[i]] = dict()
        if y[i] not in c['joint'][x[i]]:
            c['joint'][x[i]][y[i]] = 0
        c['joint'][x[i]][y[i]] += 1
        c['margin_x'][x[i]] += 1
        c['margin_y'][y[i]] += 1
    c['N'] = len(x)
    return c



def c2p(what, c, x='', y=''):

    '''
    returns (max likelihood) probability from list2count() dict
    
    Args:
      what: 'x','y','x|y','y|x' which prob to calculate
      c: dict returned from myl.lists2count()
      x
      y
    
    Returns:
      p prob
    '''

    if what == 'x':
        return c['margin_x'][x]/c['N']
    if what == 'y':
        return c['margin_y'][y]/c['N']
    if what == 'y|x':
        try:
            return c['joint'][x][y]/c['margin_x'][x]
        except:
            return 0
    if what == 'x|y':
        try:
            return c['joint'][x][y]/c['margin_y'][y]
        except:
            return 0
    if what == '-x':
        return (c['N'] - c['margin_x'][x])/c['N']
    if what == '-y':
        return (c['N'] - c['margin_y'][y])/c['N']
    if what == 'y|-x':
        try:
            c_joint = c['margin_y'][y] - c['joint'][x][y]
        except:
            c_joint = c['margin_y'][y]
        c_hist = c['N'] - c['margin_x'][x]
        return c_joint/c_hist
    if what == 'x|-y':
        try:
            c_joint = c['margin_x'][x] - c['joint'][x][y]
        except:
            c_joint = c['margin_x'][x]
        c_hist = c['N'] - c['margin_y'][y]
        return c_joint/c_hist
    return np.nan



def infoGain(x_in, y_in):

    '''
    information gain
    for categorical variables x (e.g. words) about y (e.g. sentiment)
    
    Args:
      x: list or lol of word tokens
      y: corresponding classes
    
    Returns:
      g: word -> it's information gain
    '''


    # flatten list
    if listType(x_in[0]):
        a, b = [], []
        for i in idx(x_in):
            a.extend(x_in[i])
            b.extend([y_in[i]]*len(x_in[i]))
        xx, yy = a, b
    else:
        xx, yy = cp.copy(x_in), cp.copy(y_in)

    # c.joint|margin_x|margin_y|N
    c = lists2count(xx, yy)
    # word/class types
    xc, yc = set(xx), set(yy)

    # class entropy: h = - sum_y P(y) log P(y)
    h = 0
    for y in yc:
        p = c2p('y', c, '', y)
        h -= p*binlog(p)

    # infogain per word
    g = {}
    for x in xc:
        # px: p(x)
        # pn: p(-x)
        px = c2p('x', c, x, y)
        pn = c2p('-x', c, x, y)
        # gx: sum_y y|x
        # gn: sum_y y|-x
        gx, gn = 0, 0
        for y in yc:
            p = c2p('y|x', c, x, y)
            pn = c2p('y|-x', c, x, y)
            if p > 0:
                gx += p*binlog(p)
            if pn > 0:
                gn += pn*binlog(pn)
        g[x] = h + px*gx + pn*gn
    return g


def count2mi(c):

    '''
    mutual info based on lists2count() object
    
    Args:
      c: output of lists2count()
    
    Returns:
      mi: mutual info
    '''

    mi = 0
    n = c['N']
    for x in c['joint']:
        for y in c['joint'][x]:
            pxy = c['joint'][x][y]/n
            px = c['margin'][x]/n
            py = c['margin'][y]/n
            if min(px, py, pxy) == 0:
                continue
            mi += (pxy * binlog(pxy/(py*py)))
    return mi



def make_pulse(opt):

    '''
    wrapper around Praat pulse extractor
    config analogously to make_f0()
    
    Args:
      opt
    
    Returns:
      pulse files
    '''

    pth = opt['fsys']['data']
    par = opt['param']['pulse']

    # clean up pulse subdir
    sh.rmtree(pth['pulse'])
    os.makedirs(pth['pulse'])

    if as_sudo(opt['param']['pulse']):
        cmd = "sudo praat"
    else:
        cmd = "praat"

    os.system("{} {} {} {} {} {} wav pulse".format(cmd, par['tool'],
                                                   par['param']['min'],
                                                   par['param']['max'],
                                                   pth['aud'], pth['pulse']))
    return True



def make_formants(opt):

    '''
    wrapper around Praat formant extractor
    example call: wrapper_py/ids.py
    opt analogously to make_f0
    
    Args:
      opt
    
    Returns:
      formant table files (5 formant freq and bws)
    '''

    pth = opt['fsys']['data']
    par = opt['param']['formants']

    # clean up formants subdir
    sh.rmtree(pth['formants'])
    os.makedirs(pth['formants'])

    if as_sudo(par):
        cmd = "sudo praat"
    else:
        cmd = "praat"

    os.system("{} {} {} {} {} {} {} {} wav frm".format(cmd, par['tool'],
                                                       par['param']['winlength'],
                                                       par['param']['shift'],
                                                       par['param']['maxfrequ'],
                                                       par['param']['preemp'],
                                                       pth['aud'], pth['formants']))
    return True



def make_f0(opt):

    '''
    wrapper around Praat f0 extractor
    example call: wrapper_py/hgc.py
    needs opt of format config/hgc.json
    F0 extraction
    
    Args:
      opt
    
    Returns:
      f0 files
    '''

    pth = opt['fsys']['data']
    par = opt['param']['f0']

    # clean up f0 subdir
    sh.rmtree(pth['f0'])
    os.makedirs(pth['f0'])

    if as_sudo(opt['param']['f0']):
        cmd = "sudo praat"
    else:
        cmd = "praat"

    os.system("{} {} 0.01 {} {} {} {} wav f0".format(cmd, par['tool'],
                                                     par['param']['min'],
                                                     par['param']['max'],
                                                     pth['aud'], pth['f0']))
    return True


### figure init #################################


def newfig(fs=()):

    '''
    init new figure with onclick->next, keypress->exit
    figsize can be customized
    
    Args:
      fs tuple <()>
    
    Returns:
      figure object
    
    Returns:
      figureHandle
    '''

    if len(fs) == 0:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=fs)
    cid1 = fig.canvas.mpl_connect('button_press_event', fig_next)
    cid2 = fig.canvas.mpl_connect('key_press_event', fig_key)
    return fig


def fig_next(event):
    plt.close()


def fig_key(event):
    sys.exit()



# 1811
#### plotting ###############################################
def myPlot(xin=[], yin=[], opt={}):

    '''
    plots x against y(s)
    
    Args:
      x: array or dict of x values
      y: array or dict of y values
      opt: <{}>
         bw: boolean; False; black-white
         nrm_x: boolean; False; minmax normalize all x values
         nrm_y: boolean; False; minmax normalize all y values
         ls: dict; {}; linespecs (e.g. '-k'), keys as in y
         lw: dict; {}; linewidths, keys as in y
            ls and lw for y as dict input
         legend_order; []; order of keys in x and y to appear in legend
         legend_lab; {}; mapping of keys to labels in dict
    
    Returns:
      True
    REMARKS:
    if y is dict:
        myKey1 -> [myYvals]
        myKey2 -> [myYvals] ...
    if x is array: same x values for all y.myKey*
                   if empty: indices 1:len(y) taken
            dict: same keys as for y
    opt.ls|lw can only be used if y is passed as dict
    '''

    opt = opt_default(opt, {'nrm_x': False,
                            'nrm_y': False,
                            'bw': False,
                            'ls': {},
                            'lw': {},
                            'legend_order': [],
                            'legend_lab': {},
                            'legend_loc': 'best',
                            'fs_legend': 40,
                            'fs': (25, 15),
                            'fs_ylab': 20,
                            'fs_xlab': 20,
                            'fs_title': 30,
                            'title': '',
                            'xlab': '',
                            'ylab': ''})
    xin, yin = cp.deepcopy(xin), cp.deepcopy(yin)
    # uniform dict input (default key 'y')
    if type(yin) is dict:
        y = yin
    else:
        y = {'y': yin}
    if type(xin) is dict:
        x = xin
    else:
        x = {}
        for lab in y:
            xx = xin
            if len(xin) == 0:
                xx = np.arange(0, len(y[lab]))
            x[lab] = xx
    # minmax normalization
    nopt = {'mtd': 'minmax', 'rng': [0, 1]}
    if opt['nrm_x']:
        for lab in x:
            x[lab] = nrm_vec(x[lab], nopt)
    if opt['nrm_y']:
        for lab in y:
            y[lab] = nrm_vec(y[lab], nopt)
    # plotting
    fig = newfig(fs=opt["fs"])
    ax = fig.add_subplot(111)
    if len(opt["title"]) > 0:
        ax.set_title(opt["title"], fontsize=opt["fs_title"])
    if len(opt["xlab"]) > 0:
        ax.set_xlabel(opt["xlab"], fontsize=opt["fs_xlab"])
    if len(opt["ylab"]) > 0:
        ax.set_ylabel(opt["ylab"], fontsize=opt["fs_ylab"])

    # line specs/widths
    # defaults
    if opt['bw']:
        lsd = ['-k', '-k', '-k', '-k', '-k', '-k', '-k']
    else:
        lsd = ['-b', '-g', '-r', '-c', '-m', '-k', '-y']
    while len(lsd) < len(y.keys()):
        lsd.extend(lsd)
    lwd = 4
    i = 0

    leg = []
    if len(opt["legend_order"]) > 0:
        labelKeys = opt["legend_order"]
        for lk in labelKeys:
            leg.append(opt["legend_lab"][lk])
    else:
        labelKeys = sorted(y.keys())

    # plot per label
    for lab in labelKeys:
        if lab in opt['ls']:
            cls = opt['ls'][lab]
        else:
            cls = opt['ls'][lab] = lsd[i]
            i += 1
        if lab in opt['lw']:
            clw = opt['lw'][lab]
        else:
            clw = lwd
        plt.plot(x[lab], y[lab], cls, linewidth=clw)

    if len(leg) > 0:
        plt.legend(leg,
                   fontsize=opt['fs_legend'],
                   loc=opt['legend_loc'])

    plt.show()



#### audio processing #######################################
def sox_mono(opt):

    '''
    converts fi to mono and writes it to fo
    
    Args:
      opt:
        i: input file name
        o: output file name
        c: int channel to be extracted
            0: both channels mixed
            1, 2: channel 1, 2
        as_sudo: False
    '''

    cmd = myCmd("sox", opt)
    if opt["c"] == 0:
        os.system("{} {} {} channels 1".format(cmd, opt["i"], opt["o"]))
    else:
        os.system("{} {} {} remix {}".format(cmd, opt["i"],
                                             opt["o"], opt["c"]))
    return True



def sox_trim(opt):

    '''
    trimming audio file
    
    Args:
      opt:
        i: input file name
        o: output file name
        ons: start time stamp (sec)
        dur: duration time stamp (sec)
        as_sudo: False
    '''

    cmd = myCmd("sox", opt)
    os.system("{} {} {} trim {} {}".format(cmd, opt["i"], opt["o"],
                                           opt["ons"], opt["dur"]))
    return True


def myCmd(cmd, opt):

    '''
    returns command root +/- sudo
    
    Args:
      cmd: name of command
      opt: dict that can contain key as_sudo with boolean value
    
    Returns:
      o: command name +/- sudo prefix
    '''

    if as_sudo(opt):
        return "sudo {}".format(cmd)
    return (cmd)


##### wrapper around copasul-related functionalities #############


def copa_wrapper(opt, wopt={}):

    '''
    option structure e.g. as in config/emo.json
    integrated in some other wrapper pipeline e.g. as follows:
    def myWrapper(args):
       opt = myl.args2opt(args)
       if opt["navigate"]["myCorpusSpecificProcess"]:
           fun_myCorpusSpecificProcess(opt)
       else:
           myl.copa_wrapper(opt)
    Can be extended for any navigation field myFld that
    allows for the call: copasul_call(opt["fsys"]["config"][myFld]
    opt and call then should look like this:
        opt["navigate"][myField]=True
        opt["fsys"]["config"][myField] = myConfigFile.json
        copa_wrapper(opt,{"process": [myField]})
    
    Args:
      opt with .navigate etc
      wopt wrapper workflow options
              .rm_copy: <True> purge augment security copies
              .process: <[]> list of navigation steps to be processed
                 by copasul_call() if True; standard steps not needed
    '''

    # standard keys
    navi_standard = {"f0": False, "pulse": False,
                     "augment": False, "feat": False}
    opt = opt_default(opt, {"navigate": {}})
    opt["navigate"] = opt_default(opt["navigate"], navi_standard)
    wopt = opt_default(wopt, {"rm_copy": True,
                              "process": []})

    # standard navigation
    if opt["navigate"]["f0"]:
        make_f0(opt)
    if opt["navigate"]["pulse"]:
        make_pulse(opt)
    if opt["navigate"]["augment"]:
        copasul_call(opt["fsys"]["config"]["augment"])
        if wopt["rm_copy"]:
            os.system("rm {}/*copy*".format(opt["fsys"]["data"]["annot"]))
    if opt["navigate"]["feat"]:
        copasul_call(opt["fsys"]["config"]["feat"])

    # customized navigation
    for x in wopt["process"]:
        if x in navi_standard or ext_false(opt["navigate"], x):
            continue
        if x in opt["fsys"]["config"] and is_file(opt["fsys"]["config"][x]):
            copasul_call(opt["fsys"]["config"][x])


##### file/dir operations #######################


def rm_file(x):

    '''
    remove file x
    
    Args:
      x: filename
    
    Returns:
      True if file existed, else False
    '''

    if is_file(x):
        os.remove(x)
        return True
    return False



def rm_dir(x):

    '''
    remove directory x
    
    Args:
      x: dirname
    
    Returns:
      True if dir existed, else False
    '''

    if is_dir(x):
        sh.rmtree(x)
        return True
    return False



def is_file(x):

    '''
    returns true if x is file, else false
    '''

    if os.path.isfile(x):
        return True
    return False



def is_dir(x):

    '''
    returns true if x is dir, else false
    '''

    if os.path.isdir(x):
        return True
    return False



def is_mac_dir(x):

    '''
    returns true if x is mac index dir
    '''

    if re.search("__MACOSX", x):
        return True
    return False



def make_dir_rec(x):

    '''
    recursively create folders/subfolders
    '''

    os.makedirs(x, exist_ok=True)



def make_dir(x, purge=False):

    '''
    create directory
    
    Args:
      x: dir name
      purge: delete before so that x is empty
    '''

    if purge and is_dir(x):
        sh.rmtree(x)
    if not is_dir(x):
        os.mkdir(x)



def cp_file(x, y):

    '''
    copy file x to y (y needs to be file name, too, and not just dir)
    '''

    if is_file(x):
        sh.copyfile(x, y)
        return True
    return False



def mv_file(x, y):

    '''
    copy file x to y (y needs to be file name, too, and not just dir)
    '''

    if is_file(x):
        sh.move(x, y)
        return True
    return False


def tool_wrapper(opt):

    '''
    wrapper around my perl g2p, perma etc. tools
    
    Args:
      opt:
        param.
          <all_params>: value
        cmd: command base string
        exec: <True>|False
    
    Returns:
      cmd+options
    '''

    opt = opt_default(opt, {'exec': True})
    cmd = opt['cmd']
    for o in opt['param']:
        cmd += " -{} {}".format(o, opt['param'][o])
    print(cmd)
    if opt["exec"]:
        os.system(cmd)
    return cmd



def myBoxplot(x, y, opt={}):

    '''
    creates boxplot
    
    Args:
      x: vector of variable to be plotted (necessarily already imputed!)
      y: it's grouping vector (same length)
      opt:
         title - <""> plot title
         save - <""> output file name; if empty, not saved
         xlab - <"">
         ylab - <"">
         show - <True> present figure on screen
         fs - <()> figure size tuple
         v - <True> verbose plot
    '''

    opt = opt_default(opt, {"title": "", "save": "", "show": True,
                            "notch": True, "showfliers": False,
                            "xlab": "", "ylab": "", "fs": (),
                            "v": True})
    # prep boxplot data format
    lab = uniq(y)
    s = []

    x = np.asarray(x)
    y = np.asarray(y)

    # over grp levels
    for lev in lab:
        i = find(y, "==", lev)
        s.append(x[i])

    fig = newfig(fs=opt["fs"])
    ax = fig.add_subplot(111)
    if len(opt["title"]) > 0:
        ax.set_title(opt["title"], fontsize=18)
    if len(opt["xlab"]) > 0:
        ax.set_xlabel(opt["xlab"], fontsize=14)
    if len(opt["ylab"]) > 0:
        ax.set_ylabel(opt["ylab"], fontsize=14)

    plt.boxplot(s, labels=lab, notch=opt["notch"],
                showfliers=opt["showfliers"])
    # if len(opt["title"])>0:
    #    fig.suptitle(opt["title"])
    if opt["v"]:
        plt.show()
    if len(opt["save"]) > 0:
        fig.savefig(opt["save"])


def myHistogram(x, opt={}):

    '''
    histogram
    '''

    opt = opt_default(opt, {"title": "", "save": "", "show": True,
                            "lw": 4,
                            "xlab": "", "ylab": "", "fs": (),
                            "v": True, "facecolor": "blue",
                            "num_bins": 10, "normed": 1,
                            "add_fit": False, "alpha": 0.5})

    fig = newfig(fs=opt["fs"])
    ax = fig.add_subplot(111)

    n, bins, patches = plt.hist(x, opt["num_bins"], normed=opt["normed"],
                                facecolor=opt["facecolor"], alpha=opt["alpha"])

    if opt["add_fit"]:
        mu, sigma = np.mean(x), np.std(x)
        y = mlab.normpdf(bins, mu, sigma)
        plt.plot(bins, y, 'r-', linewidth=opt["lw"])

    if len(opt["title"]) > 0:
        ax.set_title(opt["title"], fontsize=30)
    if len(opt["xlab"]) > 0:
        ax.set_xlabel(opt["xlab"], fontsize=20)
    if len(opt["ylab"]) > 0:
        ax.set_ylabel(opt["ylab"], fontsize=20)

    if opt["v"]:
        plt.show()
    if len(opt["save"]) > 0:
        fig.savefig(opt["save"])


def myScatter(x, y, opt={}):

    '''
    scatterplot
    '''


    opt = opt_default(opt, {"title": "", "save": "", "show": True,
                            "lw": 4,
                            "xlab": "", "ylab": "", "fs": (),
                            "v": True})

    fig = newfig(fs=opt["fs"])
    ax = fig.add_subplot(111)
    if len(opt["title"]) > 0:
        ax.set_title(opt["title"], fontsize=30)
    if len(opt["xlab"]) > 0:
        ax.set_xlabel(opt["xlab"], fontsize=20)
    if len(opt["ylab"]) > 0:
        ax.set_ylabel(opt["ylab"], fontsize=20)

    plt.scatter(x, y, linewidths=opt["lw"])
    if opt["v"]:
        plt.show()
    if len(opt["save"]) > 0:
        fig.savefig(opt["save"])

