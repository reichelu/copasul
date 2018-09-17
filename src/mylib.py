#!/usr/bin/env python3

# author: Uwe Reichel, Budapest, 2016

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

###########################################################
### collection of general purpose functions ###############
###########################################################

####### I/O functions
### input_wrapper()
# i_tg()       -> tg dict
# i_par()      -> interchange dict
# i_copa_xml() -> interchange dict
# i_lol()      -> 1- or 2dim list of strings
# i_seg_lab()  -> d.t [[on off]...]
#                  .lab [label ...]
# i_numpy: calls np.loadtxt() returns np.array list of floats
#          (1 col -> 1-dim array; else 1 sublist per row)
#       'pandas_csv': csv file into dict colName -> colContent (using pandas)
### output_wrapper()
# o_tg()
# o_par()
# o_copa_xml()
### annotation processing
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
### format transformations
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
### pipeline: add tiers to existing TextGrid (w/o replacement)
# myTg = myl.input_wrapper(myInputFile,'TextGrid')
# myAdd = myl.input_wrapper(myAddFile,'TextGrid')
# with tier replacement:
# opt = {'repl':True}
# without replacement
# opt = {'repl':False}
# for x in tg_tn(myAdd):
#    myTg = tg_add(myTg,tg_tier(myAdd,x),opt)
#### processing pandas_csv dict, e.g. in copasul context
# split_by_grp() - if data to be split e.g. by analysis tier
#                  -> dd[myGrpLevel] = subDict
# profile_wrapper() - transforming it into profile incl plotting
#  calls the following functions that can also be applied in isolation
#    pw_preproc(): wrapper around the following functions
#    pw_str2float(): panda is biased towards strings -> corrections, incl. NaNs
#    pw_nan2mean(): replacing NaNs by column means/medians (can be further grouped by 1 variable)
#    pw_abs(): abs-transform of selected columns
#    pw_centerScale(): column-wise center/scaling


### basic matrix op functions
# cmat(): 2-dim matrix concat, converts if needed all input to 2-dim arrays
# lol(): ensure always 2-dim array (needed e.g. for np.concatenate of 2-dim and
#        1-sim array)
# push(): appending without dimensioning flattening (vs append)
# find(): simulation of matlab find functionality
# find_interval(): indices of values within range, pretty slow
# first_interval(): first index of row containint specific element


# pushes 1 additional element y to array x (default: row-wise)
# !! if x is not empty, i.e. not []: yDim must be xDim-1, e.g.
#       if x 1-dim: y must be scalar
#       if x 2-dim: y must 1-dim
# if x is empty, i.e. [], the dimension of the output is yDim+1
# Differences to np.append:
#   append flattens arrays if dimension of x,y differ, push does not
# REMARK: cmat() might be more appropriate if 2-dim is to be returned
def push(x,y,a=0):
    if (listType(y) and len(y)==0):
        return x
    if len(x)==0:
        return np.array([y])
    return np.concatenate((x,[y]),axis=a)
    
# returns True if input is numpy array or list; else False
def listType(y):
    if (type(y)==np.ndarray or type(y)==list):
        return True
    return False

# returns isotime format, e.g. 2016-12-05T18:25:54Z
def isotime():
    return datetime.datetime.now().isoformat()

# transforms seconds to numpy array indices (=samplesIdx-1)
def sec2idx(i,fs,ons=0):
    return np.round(i*fs+ons-1).astype(int)

# transforms seconds to sample indices (arrayIdx+1)
def sec2smp(i,fs,ons=0):
    return np.round(i*fs+ons).astype(int)

# transforms numpy array indices (=samplesIdx-1) to seconds
def idx2sec(i,fs,ons=0):
    return (i+1+ons)/fs
    
# transforms sample indices (arrayIdx+1) to seconds
def smp2sec(i,fs,ons=0):
    return (i+ons)/fs

# IN:
#   s file name
# OUT:
#   TRUE if file exists, else FALSE
def fileExists(n):
    if os.path.isfile(n):
        return True
    return False


# returns predefined lists or sets (e.g. to address keys in copa dict)
# ret: return 'set' or 'list'
def lists(typ='register',ret='list'):
    ll = {'register': ['bl','ml','tl','rng'],
          'bndtyp': ['std','win','trend'],
          'bndfeat': ['r', 'rms','rms_pre','rms_post',
                      'sd_prepost','sd_pre','sd_post',
                      'corrD','corrD_pre','corrD_post',
                      'rmsR','rmsR_pre','rmsR_post',
                      'aicI','aicI_pre','aicI_post'],
          'bgd': ['bnd','gnl_f0','gnl_en','rhy_f0','rhy_en','voice'],
          'featsets': ['glob','loc','bnd','gnl_f0','gnl_en',
                       'rhy_f0','rhy_en','voice'],
          'afa': ['aud','f0','annot','pulse'],
          'fac': ['glob','loc','bnd','gnl_f0','gnl_en',
                  'rhy_f0','rhy_en','augment','chunk','voice'],
          'facafa': ['glob','loc','bnd','gnl_f0','gnl_en',
                     'rhy_f0','rhy_en','augment','chunk',
                     'aud','f0','annot','pulse'],
          'factors': ['class','ci','fi','si','gi','stm','tier','spk',
                      'is_init','is_fin','is_init_chunk','is_fin_chunk']}
    if typ in ll:
        if ret=='list':
            return ll[typ]
        else:
            return set(ll[typ])
    if ret=='list':
        return []
    return set()

# splits dict d by levels of column gv
# IN:
#   dd: dict derived from input_wrapper(...,'pandas_csv')
#   gc: grouping column name
# OUT:
#   ds: dict
#      myGrpLevel -> dSubset with gv = myGrpLevel (same keys as d)
#   if gv is not in d, then myGrpLevel = 'x'
def split_by_grp(dd,gc):
    d = cp.deepcopy(dd)
    ds = {}
    if gc not in d:
        ds['x'] = d
        return ds
    # colnames
    cn = d.keys()
    # make lists to arrays for indexing
    for n in cn:
        d[n]=np.array(d[n])
    # grouping column
    g = d[gc]
    # over grouping levels
    for lev in np.unique(g):
        ds[lev]={}
        # assign subset
        for n in cn:
            ds[lev][n] = d[n][(g == lev)]
    return ds


# wrapper around profile plotting of opt.mean values for selected features
# IN:
#   d: dict derived from input_wrapper(...,'pandas_csv')
#   opt:
#      'navigate': processing steps
#            str2float: ensure that all values in d are floats and not strings
#                       (relevant to correctly interprete np.nan) <True>
#            nan2mean: replace NaN by column opt.mean <True>
#            nrm: column-wise centering+scaling <True>
#            dict2df: create pandas data frame (internally set to True)
#            plot: output plot files <False> (requires subdict 'plot')
#      'feat': [list of feature column names
#               (in wanted output order from top to bottom)] !
#      'absfeat': [list of features for which absolute values to be taken] <[]>
#      'abs_add': add (=True) or replace (=False) absfeat column names <False>
#      'grp': [list of grouping variable column names] !
#      'stat': <'median'>|'mean'
#      'plot': dict
#           'stm': dir/stem ! of output file
#           'figsize': <()> tuple width x height
#           'bw': <False>|True
#           'fs_ytick': <20> fontsize ytick
#           'fs_legend': <30> fontsize legend
#           'title': <''> figure title
#           'lw': <5> line width
#           'concat_grp': <True> concatenate grouping variable name to plot['title']
#     (! = obligatory)
# OUT:
#   p: profile dict
#      lab -> [featureNames]
#      grp[myGrpVar][myGrpLevel] -> [values] same length and order as lab
# REMARKS:
#   it is not controlled whether or not input D is already split by analysis
#   tiers. If such a splitting is needed (e.g. for gnl or bnd feature sets)
#   use ds = split_by_grp(d,'tier') first and
#   apply profile_wrapper() separately for each ds[myTierName]
def profile_wrapper(dd,opt):
    
    ### opt init
    opt = opt_default(opt,{'stat':'median', 'navigate':{},
                           'absfeat':[], 'plot':{}})
    opt['navigate'] = opt_default(opt['navigate'],
                                  {'str2float': True,
                                   'nan2mean': True,
                                   'abs_add': False,
                                   'nrm': True,
                                   'plot': False})
    opt['plot'] = opt_default(opt['plot'], {'bw': False, 'fs_ytick': 20,
                                            'fs_legend': 30, 'title': '',
                                            'lw': 5, 'figsize':(),
                                            'concat_grp': False})
    opt['navigate']['dict2df']=True
    if 'stm' not in opt['plot']:
        opt['navigate']['plot'] = False
        
    ### preprocessing
    # opt['feat'] might be updated by *_abs columns
    # d is now pd.dataframe
    d, opt = pw_preproc(dd,opt)
    
    ### profile
    p = pw_prof(d,opt)
    
    ### plotting
    if opt['navigate']['plot']:
        pw_plot(p,opt['plot'])

    return p

# called by profile_wrapper() for plotting
# IN:
#   p: profile dict
#      lab -> [featureNames]
#      grp[myGrpVar][myGrpLevel] -> [values] same length and order as lab
#   opt: opt['plot'] dict from profile_wrapper()

def pw_plot(p,opt):
    
    ### plotting
    if opt['bw']:
        cols = ['k-', 'k--', 'k-.', 'k:']
    else:
        cols = ['-g','-r','-b','-k']

    # y-axis ticks
    yi = np.asarray(range(1,len(p['lab'])+1))
        
    # over grouping variables
    for g in p['grp']:
        
        colI = 0
        y = p['grp'][g]
        
        fo = "{}_{}.pdf".format(opt['stm'],g)
        fig = newfig(opt['figsize'])
        # over levels
        for lev in sorted(y.keys()):
            plt.plot(y[lev],yi,"{}".format(cols[colI]),label=lev,linewidth=5)
            colI+=1

        plt.yticks(yi,p['lab'],fontsize = opt['fs_ytick'])
        if opt['concat_grp']:
            plt.title("{}: {}".format(opt['title'],g))
        else:
            plt.title(opt['title'])
        plt.legend(fontsize = opt['fs_legend'])
        plt.show()
        fig.savefig(fo)
        #stopgo()
    return


# called by profile_wrapper()
# IN, OUT, cf there
def pw_prof(d,opt):
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


# wrapper around preprocessing for profile generation.
# called by profile_wrapper() or standalone
# IN:
#   d: dict by input_wrapper(...,'pandas_csv')
#   opt:
#      'navigate': processing steps
#            str2float: ensure that all values in d are floats and not strings
#                       (relevant to correctly interprete np.nan) <True>
#            nan2mean: replace NaN by column opt.mean <True>
#            nrm: column-wise centering+scaling <True>
#            dict2df: create pandas data frame (always <True> if called by
#                     profile_wrapper(); if not set: <False>
#      'feat': [list of feature column names
#               (in wanted output order from top to bottom)] !
#      'absfeat': [list of features for which absolute values to be taken] <[]>
#      'abs_add': <False> add (=True) or replace (=False) absfeat column names in d
#      'stat': <'median'>|'mean'
#      'grp_n2m': grouping column name for nan2mean separately for each factor level <''>
# OUT:
#   d preprocessed
#   opt: evtl. with updated 'feat' list (in case of abs_add
def pw_preproc(d,opt):
    
    ### opt init
    opt = opt_default(opt,{'stat':'median', 'navigate':{},
                           'absfeat':[], 'grp_n2m': ''})
    opt['navigate'] = opt_default(opt['navigate'],
                                  {'str2float': True,
                                   'nan2mean': True,
                                   'abs_add': False,
                                   'nrm': True,
                                   'dict2df': False})
    
    d = cp.deepcopy(d)
    if opt['navigate']['str2float']:
        d = pw_str2float(d,opt['feat'])
    if opt['navigate']['nan2mean']:
        d = pw_nan2mean(d,opt['feat'],opt['stat'],opt['grp_n2m'])
    if len(opt['absfeat'])>0:
        d, cn = pw_abs(d,opt['absfeat'],opt['navigate']['abs_add'])
        if opt['navigate']['abs_add']:
            # element-wise to keep pre-specified order
            for x in cn:
                if x in opt['feat']:
                    continue
                opt['feat'].append(x)
    if opt['navigate']['nrm']:
        d = pw_centerScale(d,opt['feat'])
    if opt['navigate']['dict2df']:
        d = pd.DataFrame(d)
    
    return d, opt

        
# column-wise string to float
# IN:
#   d: dict from input_wrapper(...,'pandas_csv')
#   cn: list of names of columns to be processed
# OUT:
#   d: with processed columns
def pw_str2float(d,cn):
    for x in cn:
        v = ea()
        for i in range(len(d[x])):
            v = np.append(v,float(d[x][i]))
        d[x] = v
    return d

# replaces np.nan by column medians
# IN:
#   d: dict from input_wrapper(...,'pandas_csv')
#   cn: list of names of columns to be processed
#   mv: type of mean value '<median>'|'mean'
#   grp: grouping variable for nan2mean by factor-level
# OUT:
#   d: with processed columns
def pw_nan2mean(d,cn,mv='median',grp=''):
    ### factor-level nan2mean
    if len(grp)>0 and grp in d:
        # grpLevel -> row indices in d
        gri = {}
        for g in uniq(d[grp]):
            gri[g] = find(d[grp],'==',g)

        # over features
        for x in cn:
            xina = find(d[x],'is','nan')
            if len(xina)==0:
                continue
            xifi = find(d[x],'is','finite')
            # over grouping levels
            for g in gri:
                ina = intersect(xina,gri[g])
                ifi = intersect(xifi,gri[g])
                if len(ina)==0 or len(ifi)==0:
                    continue
                if mv=='median':
                    m = np.median(d[x][ifi])
                else:
                    m = np.mean(d[x][ifi])
                d[x][ina]=m

    ### global nan2mean
    # redo also for factor level nan2mean to catch
    # non-replacement cases 
    for x in cn:
        ina = find(d[x],'is','nan')
        if len(ina)==0:
            continue
        ifi = find(d[x],'is','finite')
        if mv=='median':
            m = np.median(d[x][ifi])
        else:
            m = np.mean(d[x][ifi])
        d[x][ina]=m
    return d                    

# replace column content by abs values or add column with name *_abs
# IN:
#   d: dict from input_wrapper(...,'pandas_csv')
#   cn: list of names of columns to be processed
#   do_add: <False>; if True add abs val columns, if False, replace
#       column content
# OUT:
#   d: with processed columns
#   cna: list with updated column names (only different for do_add=True)
def pw_abs(d,cn,do_add=False):
    cna = cp.deepcopy(cn)
    for c in cn:
        if do_add:
            ca = "{}_abs".format(c)
            cna.append(ca)
            d[ca] = np.abs(d[c])
        else:
            d[c] = np.abs(d[c])
    return d, cna

# center/scale feature values
# IN:
#  dict from input_wrapper(...,'pandas_csv')
#  cn - relevant column names
# OUT:
#  d - with relevant columns centered+scaled
def pw_centerScale(d,cn):
    for x in cn:
        d[x] = d[x].reshape(-1, 1)
        cs = sp.RobustScaler().fit(d[x])
        d[x] = cs.transform(d[x])
        d[x] = d[x][:,0]
    return d


# simulation of matlab find for 1-dim data
# IN:
#  1-dim array
#  op
#  value (incl. 'max'|'min'|'nan' etc. in combination with op 'is')
# OUT:
#  1-dim index array
def find(x_in,op,v):
    x = np.asarray(x_in)
    if op=='==':
        xi = np.asarray((x==v).nonzero())[0,:]
    elif op=='!=':
        xi = np.asarray((x!=v).nonzero())[0,:]
    elif op=='>':
        xi = np.asarray((x>v).nonzero())[0,:]
    elif op=='>=':
        xi = np.asarray((x>=v).nonzero())[0,:]
    elif op=='<':
        xi = np.asarray((x<v).nonzero())[0,:]
    elif op=='<=':
        xi = np.asarray((x<=v).nonzero())[0,:]
    elif (op=='is' and v=='nan'):
        xi = np.asarray(np.isnan(x).nonzero())[0,:]
    elif (op=='is' and v=='finite'):
        xi = np.asarray(np.isfinite(x).nonzero())[0,:]
    elif (op=='is' and v=='inf'):
        xi = np.asarray(np.isinf(x).nonzero())[0,:]
    elif (op=='is' and v=='max'):
        return find(x,'==',np.max(x))
    elif (op=='is' and v=='min'):
        return find(x,'==',np.min(x))
    return xi.astype(int)

# returns mean absolute error of vectors
# or of one vector and zeros (=mean abs dev)
# IN:
#   x
#   y <zeros>
# OUT:
#   meanAbsError(x,y)
def mae(x,y=[]):
    if len(y)==0:
        y=np.zeros(len(x))
    x=np.asarray(x)
    return np.mean(abs(x-y))

# residual squared deviation
# IN:
#   x: data vector
#   y: prediction vector (e.g. fitted line) or []
# OUT:
#   r: residual squared deviation
def rss(x,y=[]):
    if len(y)==0:
        y=np.zeros(len(x))
    x=np.asarray(x)
    y=np.asarray(y)
    return np.sum((x-y)**2)


# aic information criterion for least squares fit
# for model comparison, i.e. without constant terms
# IN:
#   x: underlying data
#   y: predictions (same length as x!)
#   k: number of parameters (<3> for single linear fit)
def aic_ls(x,y,k=3):
    n=len(x)
    r = rss(x,y)
    if r==0:
        return 2*k
    aic = 2*k + n*np.log(r)
    return aic

# robust natural log
# log(<=0) is np.nan
def robust_log(y):
    if y<=0:
        return np.nan
    return np.log(y)

# robust division
# returns np.nan for 0-divisions
def robust_div(x,y):
    if y == 0:
        return np.nan
    return x/y

# mean squared error of vectors
# or of one vector and zeros (=mean squared dev)
def mse(x,y=[]):
    if len(y)==0:
        y=np.zeros(len(x))
    x=np.asarray(x)
    return np.mean((x-y)**2)

# returns RMSD of two vectors
# or of one vector and zeros
def rmsd(x,y=[]):
    if len(y)==0:
        y=np.zeros(len(x))
    x=np.asarray(x)
    return np.sqrt(np.mean((x-y)**2))

# ~opposite of which_interval()
# returns indices of values in 1-dim x that are >= iv[0] and <= iv[1]
# IN:
#   x: 1-dim array
#   iv: 2-element interval array
# OUT:
#   1-dim index array of elements in x within (inclusive) range of iv 
def find_interval(x,iv,fs=-1):
    xi = sorted(intersect(find(x,'>=',iv[0]),
                          find(x,'<=',iv[1])))
    return np.asarray(xi).astype(int)

# ~opposite of find_interval()
# returns row index of seg containing t (only first in case of multiple)
# IN:
#  x number
#  iv 2-dim array, interval [on offset]
# OUT:
#  rowIdx <-1>
def first_interval(x,iv):
    ri = -1
    xi = sorted(intersect(find(iv[:,0],'<=',x),
                          find(iv[:,1],'>=',x)))
    if len(xi)>0:
        ri = xi[0]

    return int(ri)

# vectorized version of windowing
# IN:
#   s
#     .win: window length
#     .rng: [on, off] range of indices to be windowed
# OUT: [[on off] ...]
def seq_windowing(s):
    vecwin = np.vectorize(windowing)
    r = s['rng']
    ww = np.asarray(vecwin(range(r[0],r[1]),s))
    return ww.T
    

# window of length wl on and offset around single index in range [on off]
# vectorized version: seq_windowing
# IN:
#   i current index
#   s
#    .win window length
#    .rng [on, off] range of indices to be windowed
# OUT:
#  on, off of window around i
def windowing(i,s):
    # half window
    wl = max([1,math.floor(s['win']/2)])
    r = s['rng']
    on = max([r[0],i-wl])
    off = min([i+wl,r[1]])
    # extend window
    d = (2*wl-1) - (off-on)
    if d>0:
        if on>r[0]:
            on = max([r[0], on-d])
        elif off < r[1]:
            off = min([off+d, r[1]])
    return on, off

# as windowing(), but returning all indices from onset to offset
#   i current index
#   s
#    .win window length
#    .rng [on, off] range of indices to be windowed
# OUT:
#  [on:1:off] in window around i
def windowing_idx(i,s):
    on, off = windowing(i,s)
    return np.arange(on,off,1)
    

# returns intersection list of two 1-dim lists
def intersect(a,b):
    return list(set(a) & set(b))

# for flexible command line vs embedded call of some function
# either: args contains a 'config' key, then opt is read from config file
# or: args contains key <reqKey>, then opt is set to args
# IN:
#   args dict
#   reqKey required key
# OUT:
#   opt dict read from args.config or equal args
def args2opt(args,reqKey):
    if 'config' in args:
        opt = input_wrapper(args['config'],'json')
    elif reqKey in args:
        opt = args
    else:
        sys.exit("args cannot be transformed to opt")
    return opt


# printing arbitrary number of variables
def pr(*args):
    for v in args:
        print(v)

# returns sorted list of numeric (more general: same data-type) keys
# IN:
#   x dict with numeric keys
# OUT:
#   sortedKeyList
def numkeys(x):
    return sorted(list(x.keys()))

# same as numkeys(), only for name clarity purpose
def sorted_keys(x):
    return sorted(list(x.keys()))

# add key to empty subdict if not yet part of dict
# IN:
#   d dict
#   s key
# OUT:
#   d incl key spointing to empty subdict
def add_subdict(d,s):
    if not (s in d):
        d[s] = {}
    return d

# for debugging
# wait until <return>
# IN:
#   x - optional message
def stopgo(x=''):
    z = input(x)
    return

# returns files incl full path as list (recursive dir walk)
# IN:
#   d - string, directory
#   e - string, extension
# OUT:
#   ff - list of fullPath-filenames
def file_collector(d,e):
    ff=[]
    for root, dirs, files in os.walk(d):
        files.sort()
        for f in files:
            if f.endswith(e):
                ff.append(os.path.join(root, f))
    return sorted(ff)

# renames key names for intercompatibility of opt dictionaries 
# IN: dict opt
#     dict maps oldName -> newName
# OUT: dict mopt with replaced keynames
def optmap(opt,maps):
    mopt={}
    for key in maps:
        mopt[maps[key]] = opt[key]
    return mopt


# removes outliers in 1-dim array
# IN:
#   y - 1dim array
#   opt: 'f' -> factor of min deviation
#        'm' -> from 'mean' or 'median'
# OUT:
#   z - y without outliers
def outl_rm(y,opt):
    opt['zi']=False
    oi = outl_idx(y,opt)
    mask=np.ones(len(y),np.bool)
    mask[oi]=False
    return y[mask]

# marks outliers in arrayreturns idx of outliers
# IN:
# y - numeric array
# opt - 'f' -> factor of min deviation
#       'm' -> from 'mean', 'median' or 'fence'
#           (mean: m +/- f*sd,
#            median: med +/- f*iqr,
#            fence: q1-f*iqr, q3+f*iqr)
#       'zi' -> true|false - ignore zeros in m and outlier calculation
# OUT:
# io - indices of outliers
def outl_idx(y,opt):
    if opt['zi']==True:
        i = (y!=0).nonzero()
    else:
        i = range(np.size(y))

    f=opt['f']

    if np.size(i)==0:
        return ea()

    # getting lower and upper boundary lb, ub
    if opt['m'] == 'mean':
        # mean +/- f*sd
        m = np.mean(y.take(i))
        r = np.std(y.take(i))
        lb, ub = m-f*r, m+f*r
    else:
        m = np.median(y.take(i))
        q1, q3 = np.percentile(y.take(i), [25,75])
        r = q3 - q1
        if opt['m'] == 'median':
            # median +/- f*iqr
            lb, ub = m-f*r, m+f*r
        else:
            # Tukey's fences: q1-f*iqr , q3+f*iqr
            lb, ub = q1-f*r, q3+f*r
        
    if opt['zi']==False:
        io = ((y>ub) | (y<lb)).nonzero()
    else:
        io = ((y>0) & ((y>ub) | (y<lb))).nonzero()

    #stopgo("m: {}, lb: {}, ub: {}, n: {}".format(trunc2(m),trunc2(lb),trunc2(ub),len(io[0])))

    return io

# output wrapper, extendable, so far only working for typ 'pickle'
# 'TextGrid' and 'list' (1-dim)
# IN:
#   anyVariable (except json: dict type)
#   fileName
#   typ: 'pickle'|'TextGrid'|'json'|'list'|'string'
# OUT:
#   <fileOutput>
def output_wrapper(v,f,typ):
    if typ == 'pickle': m = 'wb'
    else: m = 'w'
    if typ == 'pickle':
        with open(f,m) as h:
            pickle.dump(v,h)
            h.close()
    elif typ == 'TextGrid':
        o_tg(v,f)
    elif re.search('xml',typ):
        o_copa_xml(v,f)
    elif re.search('(string|list)',typ):
        if typ=='list':
            x = "\n".join(v)
        else:
            x = v
        h = open(f,mode='w',encoding='utf-8')
        h.write(x)
        h.close()
    elif typ == 'json':
        with open(f,m) as h:
            json.dump(v, h, indent="\t", sort_keys=True)
            h.close()


# TextGrid output of dict read in by i_tg()
# (appended if file exists, else from scratch)
# IN:
#   tg dict
#   f fileName
# OUT:
#   intoFile
def o_tg(tg,fil):
    h = open(fil,mode='w', encoding='utf-8')
    idt = '    '
    fld = tg_fields()
    ## head
    if tg['format'] == 'long':
        h.write("File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n")
        h.write("xmin = {}\n".format(tgv(tg['head']['xmin'],'xmin')))
        h.write("xmax = {}\n".format(tgv(tg['head']['xmax'],'xmax')))
        h.write("tiers? <exists>\n")
        h.write("size = {}\n".format(tgv(tg['head']['size'],'size')))
    else:
        h.write("File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n")
        h.write("{}\n".format(tgv(tg['head']['xmin'],'xmin')))
        h.write("{}\n".format(tgv(tg['head']['xmax'],'xmax')))
        h.write("<exists>")
        h.write("{}\n".format(tgv(tg['head']['size'],'size')))

    ## item
    if (tg['format'] == 'long'):
        h.write("item []:\n")

    for i in numkeys(tg['item']):
        # subkey := intervals or points?
        if re.search(tg['item'][i]['class'],'texttier',re.I):
            subkey = 'points'
        else:
            subkey = 'intervals'
        if tg['format'] == 'long':
            h.write("{}item [{}]:\n".format(idt,i))
        for f in fld['item']:
            if tg['format'] == 'long':
                if f == 'size':
                    h.write("{}{}{}: size = {}\n".format(idt,idt,subkey,tgv(tg['item'][i]['size'],'size')))
                else:
                    h.write("{}{}{} = {}\n".format(idt,idt,f,tgv(tg['item'][i][f],f)))
            else:
                h.write("{}\n".format(tgv(tg['item'][i][f],f)))
        
        # empty tier
        if subkey not in tg['item'][i]: continue
        for j in numkeys(tg['item'][i][subkey]):
            if tg['format'] == 'long':
                h.write("{}{}{} [{}]:\n".format(idt,idt,subkey,j))
            for f in fld[subkey]:
                if (tg['format'] == 'long'):
                    h.write("{}{}{}{} = {}\n".format(idt,idt,idt,f,tgv(tg['item'][i][subkey][j][f],f)))
                else:
                    h.write("{}\n".format(tgv(tg['item'][i][subkey][j][f],f)))
    h.close()

    
# returns field names of TextGrid head and items
# OUT:
#   hol fieldNames
def tg_fields():
    return {'head': ['xmin','xmax','size'],
            'item': ['class','name','xmin','xmax','size'],
            'points': ['time', 'mark'],
            'intervals': ['xmin', 'xmax', 'text']}


# rendering of TextGrid values
# IN:
#   s value
#   s attributeName
# OUT:
#   s renderedValue
def tgv(v,a):
    if re.search('(xmin|xmax|time|size)',a):
        return v
    else:
        return "\"{}\"".format(v)

# returns tier subdict from TextGrid
# IN:
#   tg: dict by i_tg()
#   tn: name of tier
# OUT:
#   t: dict tier
def tg_tier(tg,tn):
    if tn not in tg['item_name']:
        return {}
    return tg['item'][tg['item_name'][tn]]

# returns list of TextGrid tier names
# IN:
#   tg: textgrid dict
# OUT:
#   tn: sorted list of tiernames
def tg_tn(tg):
    return sorted(list(tg['item_name'].keys()))
    
# creates chunk tier (interpausal units) from MAU tier in TextGrid
# IN:
#   tg: textgrid dict
#   tn: MAUS tier name <'MAU'>
#   cn: CHUNK tier name <'CHUNK'>
#   cl: chunk label <'c'>
# OUT:
#   c: chunk tier dict
# REMARK:
#   c can be added to tg by myl.tg_add(tg,c)
def tg_mau2chunk(tg,tn='MAU',cn='CHUNK',cl='c'):
    pau = '<p:>'
    # MAU tier
    t = tg_tier(tg,tn)
    #t = tg['item'][tg['item_name'][tn]]
    # CHUNK tier
    k = 'intervals'
    c = {'size':0,'name':cn,k:{}}
    for x in ['xmin','xmax','class']:
        c[x] = t[x]
    # t onset, idx in chunk tier
    to, j = 0, 1
    kk = numkeys(t[k])
    for i in kk:
        t1 = trunc4(t[k][i]['xmin'])
        t2 = trunc4(t[k][i]['xmax'])
        if t[k][i]['text']==pau:
            if to<t1:
                c[k][j] = {'xmin':to,'xmax':t1,'text':cl}
                j+=1
            c[k][j] = {'xmin':t1,'xmax':t2,'text':pau}
            j+=1
            to=t2
    # final non-pause segment
    if t[k][kk[-1]]['text'] != pau:
        c[k][j] = {'xmin':to,'xmax':t[k][kk[-1]]['xmax'],'text':cl}
        j+=1
    # size
    c['size']=j-1
    return c

# replaces NA... values in list according to mapping in map
# IN:
#   x: 1-dim list
#   mp: mapping dict (default NA, NaN, nan -> np.nan)
# OUT:
#   x: NANs, INFs etc replaced
def nan_repl(x,mp={}):
    if len(mp.keys())==0:
        mp={'NA':np.nan, 'NaN':np.nan, 'nan':np.nan}
    x = np.array(x)
    for z in mp:
        x[ x==z ] = mp[z]
    return x

# replaces NaN and INF by column median values
# IN:
#   x: vector
# OUT:
#   y: vector with replaced NaN
def nan2mean(x):
    inan = find(x,'is','nan')
    iinf = find(x,'is','inf')
    if max(len(inan),len(iinf))==0:
        return x
    ifi = find(x,'is','finite')
    m = np.median(x[ifi])
    if len(inan)>0:
        x[inan]=m
    if len(iinf)>0:
        x[iinf]=m
    
    return x

# input wrapper, extendable, so far working for
# 'json', 'pickle', 'TextGrid', 'tab'/'lol', 'par', 'xml', 'l_txt',
# 'lol_txt', 'seg_lab', 'list','csv', 'copa_csv', 'pandas_csv'
#   xml is restricted to copa_xml format 
#   *csv: recommended: 'pandas_csv', since 'csv' treats all values as strings
#   and 'copa_csv' is more explicit than 'pandas_csv' in type def but
#   might get deprecated
#   Current diffs between copa_csv and pandas_csv: idx columns as ci, fi etc 
#   and  numeric speakerIds will be treated as integer in pandas_csv but as
#   strings in cop_csv
# IN:
#    fileName
#    typ
#    opt <{}> additional options
# OUT:
#    containedVariable
# REMARK:
#    diff between lol and lol_txt:
#    lol: numeric 2-dim np.array (text will be ignored)
#    lol_txt: any 2-dim array
#    l_txt: 1-dim array splitting input text at blanks
#    seg_lab: on off label
#    i_numpy: output of np.loadtxt (1- or 2-dim array)
#    csv: returns dict with keys defined by column titles 
#    csv: NA, NaN, Inf ... are kept as strings. In some
#        context need to be replaced by np.nan... (e.g. machine learning)
#        use myl.nan_repl() for this purpose
def input_wrapper(f,typ,opt={}):
    # col separator
    if 'sep' in opt:
        sep = opt['sep']
    else:
        if typ=='i_numpy':
            sep = None
        else:
            sep = ','
    # 1-dim list of rows
    if typ=='list':
        return i_list(f)
    # 1-dim nparray of floats
    if typ=='i_numpy':
        return np.loadtxt(f,delimiter=sep)
    # TextGrid
    if typ=='TextGrid':
        return i_tg(f)
    # xml
    if typ=='xml':
        return i_copa_xml(f)
    # csv into dict (BEWARE: everything is treaten as strings!)
    if typ=='csv':
        o = {}
        for row in csv.DictReader(open(f,'r'),delimiter=sep):
            for a in row:
                if a not in o:
                    o[a] = []
                o[a].append(row[a])
        return o
    # copa csv into dict (incl. type conversion)
    if typ=='copa_csv':
        o = {}
        for row in csv.DictReader(open(f,'r'),delimiter=sep):
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
    if typ=='pandas_csv':
        o = pd.read_csv(f,sep=sep)
        return o.to_dict('list')
    # par
    if typ=='par_as_tab':
        return i_par(f)
    if typ=='pickle':
        m = 'rb'
    else:
        m = 'r'
    if (typ == 'lol' or typ == 'tab'):
        return lol(f,opt)
    # 1 dim list of blanksep items
    if typ == 'l_txt':
        return i_lol(f,sep='',frm='1d')
    if typ == 'lol_txt':
        return i_lol(f)
    if typ == 'seg_lab':
        return i_seg_lab(f)
    with open(f,m) as h:
        if typ=='json':
            return json.load(h)
        elif typ=='pickle':
            return pickle.load(h)
    return False

# decides whether name belongs to categorical variable 
def copa_categ_var(x):
    if ((x in lists('factors','set')) or
        re.search('^(grp|lab|class|spk|tier)',x) or
        re.search('_(grp|lab|class|tier)$',x) or
        re.search('_(grp|lab|class|tier)_',x)):
        return True
    return False

# dynamically adjusts 'fsys' part on copa options based on input
# requires uniform config format as in wrapper_py/config/ids|hgc.json
# IN:
#   task: subfield in opt[fsys][config] pointing to copa opt json file
#   opt: in which to find this subfield
# OUT:
#   copa_opt: copasul options with adjusted fsys specs
def copa_opt_dynad(task,opt):

    # read copa config file
    copa_opt = input_wrapper(opt['fsys']['config'][task],'json')

    # adjust fsys specs according to opt
    for d in ['aud','f0','annot','export','pic']:
        if d in opt['fsys']['data']:
            copa_opt['fsys'][d]['dir'] = opt['fsys']['data'][d]
    copa_opt['fsys']['export']['stm'] = opt['fsys']['data']['stm']
    copa_opt['fsys']['pic']['stm'] = opt['fsys']['data']['stm']
    
    return copa_opt


# [on off label] rows converted to 2-dim np.array and label list
# IN:
#   f - file name
#   sep - <''> separator, 
#       regex need to be marked by 'r:sep'
#       e.g.: 'r:\\t+' - split at tabs (double backslash needed!)
# OUT:
#   d.t [[on off]...]
#    .lab [label ...]
def i_seg_lab(f,sep=''):
    if re.search('^r:',sep):
        is_re = True
        sep = re.sub('^r:','',sep)
    else:
        is_re = False
    d = {'t':ea(), 'lab':[]}
    js = ' '
    with open(f, encoding='utf-8') as h:
        for z in h:
            # default or customized split
            if len(sep)==0:
                x = z.split()
            else:
                # whitspace standardizing
                z = str_standard(z,True)
                # regex or string split
                if is_re:
                    x = re.split(sep,z)
                else:
                    x = z.split(sep)
            x[0] = float(x[0])
            x[1] = float(x[1])
            d['t'] = push(d['t'],x[0:2])
            if len(x)>=3:
                d['lab'].append(js.join(x[2:len(x)]))
            else:
                d['lab'].append('')
        h.close()
    return d

# truncates float n to precision 2
def trunc2(n):
    return float('%.2f'%(n))
    
# truncates float n to precision 4
def trunc4(n):
    return float('%.4f'%(n))

# truncates float n to precision 6
def trunc6(n):
    return float('%.6f'%(n))

# counting events from input list
# IN:
#   x - list of strings
# OUT:
#   c - dict
#    myType -> myCount
#   n - number of tokens in x
def counter(x):
    c={}
    for y in x:
        if y not in c:
            c[y]=1
        else:
            c[y]+=1
    return c, len(x)

# counts to MLE probs
# IN:
#    c - counter dict from counter()
#    n - number of tokens from counter()
# OUT:
#    p - dict
#     myType -> myProb
def count2prob(c,n):
    p={}
    for x in c:
        p[x] = c[x]/n
    return p

# probs to entropy
# IN:
#    p - dict: myType -> myProb from count2prob()
# OUT:
#    h - unigramEntropy
def prob2entrop(p):
    h = 0
    for x in p:
        if p[x]==0:
            continue
        h -= (p[x] * math.log(p[x],2))
    return h

# information radius of two probability distributions
# IN:
#    p - dict: myType -> myProb from count2prob()
#    q - same
# OUT:
#    r - information radius between p and q
def prob2irad(p_in,q_in):
    p = cp.deepcopy(p_in)
    q = cp.deepcopy(q_in)

    # mutual filling of probmodels
    for e in set(p.keys()):
        if e not in q:
            q[e]=0
    for e in set(q.keys()):
        if e not in p:
            p[e]=0

    r = 0 
    for e in set(p.keys()):
        m = (p[e]+q[e])/2
        if m==0:
            continue
        if p[e]==0:
            a = q[e]*binlog(q[e]/m)
        elif q[e]==0:
            a = p[e]*binlog(p[e]/m)
        else:
            a = p[e]*binlog(p[e]/m)+q[e]*binlog(q[e]/m)
        r += a
    return r


# shortcut to log2
def binlog(x):
    return math.log(x,2)


# wrapper around counter() and count2prob()
# IN:
#   x - list of strings
# OUT:
#   p - dict
#     myType -> myProb
def list2prob(x):
    c, n = counter(x)
    return count2prob(c,n)

# wrapper around counter() and count2prob(), prob2entrop()
# IN:
#   x - list of strings
# OUT:
#   myUnigramEntropy
def list2entrop(x):
    c, n = counter(x)
    p = count2prob(c,n)
    return prob2entrop(p)

# returns sorted + unique element list
# IN:
#   x - list
# OUT:
#   x - list unique
def uniq(x):
    return sorted(list(set(x)))

# returns precision, recall, and F1
# from input dict C
# IN:
#   c: dict with count for 'hit', 'ans', 'ref'
# OUT:
#   val: dict
#      'precision', 'recall', 'f1'
def precRec(c):
    p = c['hit']/c['ans']
    r = c['hit']/c['ref']
    f = 2*(p*r)/(p+r)
    return {'precision':p,'recall':r,'f1':f}


# apply scalar input functions to 1d or 2d arrays
# IN:
#   f - function
#   x - array variable
def cellwise(f,x):
    if len(x)==0:
        return x
    vf = np.vectorize(f)
    # 1d input
    if np.asarray(x).ndim==1:
        return vf(x)
    # 2d input
    return np.apply_along_axis(vf,1,x)


# outputs also one-line inputs as 2 dim array
# IN: fileName or list
#     opt:
#       'colvec' <FALSE> +/- enforce to return column vector
#             (for input files with 1 row or 1 column
#              FALSE returns [[...]], and TRUE returns [[.][.]...])
# OUT: numpy 2-dim array
# BEWARE
def lol(f,opt={}):
    opt = opt_default(opt,{'colvec':False})
    try:
        x = np.loadtxt(f)
    except:
        x = f

    if x.ndim==1:
        x=x.reshape((-1,x.size))
    x=np.asarray(x)

    if opt['colvec'] and len(x)==1:
        x = np.transpose(x)

    return x

# returns robust median from 1- or 2-dim array
# IN:
#   x array
#   opt: 'f' -> factor of min deviation
#        'm' -> from 'mean' or 'median'
# OUT:
#   m median scalar or vector
def robust_median(x,opt):
    nc = ncol(x)
    # 1-dim array
    if nc==1:
        return np.median(outl_rm(x,opt))
    # 2-dim
    m=ea()
    for i in range(nc):
        m = np.append(m, np.median(outl_rm(x[:,i],opt)))
    return m

# returns number of columns    
# 1 if array is one dimensional
# IN:
#   x: 2-dim array
# OUT:
#   numfOfColumns
def ncol(x):
    if np.ndim(x)==1:
        return 1
    return len(x[0,:])

# returns number of rows
# 1 for 1-dim array
# IN:
#   x: array
# OUT:
#   numfOfRows
def nrow(x):
    if np.ndim(x)==1:
        return 1
    return len(x)

# outputs a data frame assigning each column a title
# IN:
#    s F   - fileName
#    l COL - colNames
# OUT:
#    dataFrame DF with content of x
def df(f,col):
    x = lol(f)
    col_names = {}
    for i in range(0,len(x[0,:])):
        col_names[col[i]]=x[:,i]
    df = pd.DataFrame(col_names)
    return df

# returns dir, stm, ext from input
# IN:
#   x: fullPathString
# OUT:
#   dir
#   stm
#   ext
def dfe(x):
    dd = os.path.split(x)
    d = dd[0]
    s = os.path.splitext(os.path.basename(dd[1]))
    e = s[1]
    e = re.sub('\.','',e)
    return d, s[0], e


# returns file name stem
# IN:
#   f fullPath/fileName
# OUT:
#   s stem
def stm(f):
    s = os.path.splitext(os.path.basename(f))[0]
    return s

# normalizes vector x according to
#   opt.mtd|(rng|max|min)
# mtd: 'minmax'|'zscore'|'std'
#   'minmax' - normalize to opt.rng
#   'zscore' - z-transform
#   'std' - divided by std (whitening)
# IN:
#   x - vector
#   opt - dict 'mtd'|'rng'|'max'|'min'
# OUT:
#   x normalized
def nrm_vec(x,opt):
    if opt['mtd']=='minmax':
        r = opt['rng']
        if 'max' in opt:
            ma = opt['max']
        else:
            ma = max(x)
        if 'min' in opt:
            mi = opt['min']
        else:
            mi = min(x)
        if ma>mi:
            x = (x-mi)/(ma-mi)
            x = r[0] + x*(r[1]-r[0]);
    elif opt['mtd']=='zscore':
        x = st.zscore(x)
    elif opt['mtd']=='std':
        x = sc.whiten(x)
    return x

# normalizes scalar to range opt.min|max set to opt.rng
# supports minmax only
def nrm(x,opt):
    if opt['mtd']=='minmax':
        mi = opt['min']
        ma = opt['max']
        r = opt['rng']
        if ma>mi:
            x = (x-mi)/(ma-mi)
            x = r[0] + x*(r[1]-r[0]);
    return x

# maps integers from -32768 to 32767 to interval [-1 1]
def wav_int2float(s):
    #return nrm_vec(s,{'mtd':'minmax','min':-32768,'max':32767,'rng':[-1,1]})
    s = s/32768
    s[find(s,'<',-1)]=-1
    s[find(s,'>',1)]=1
    return s

# normalisation of T to range specified in vector RANGE
# opt
#     .t0  zero is placed to value t0
#     .rng [min max] val for t nrmd, must span interval  
# RNG must span interval including 0
def nrm_zero_set(t,opt):
    if len(t)==0: return t
    # t halves
    t1 = t[find(t,'<=',opt['t0'])]
    t2 = t[find(t,'>',opt['t0'])]

    if len(t1)==0 or len(t2)==0:
        return nrm_vec(t,{'mtd':'minmax','rng':opt['rng']})

    # corresponding ranges
    r1=[opt['rng'][0], 0];
    r2=[opt['rng'][1]/len(t2), opt['rng'][1]];

    # separate normalisations for t-halves
    o = {}
    o['mtd'] = 'minmax';
    o['rng'] = r1
    t1n = nrm_vec(t1,o)
    
    o['rng'] = r2
    t2n = nrm_vec(t2,o);

    return np.concatenate((t1n,t2n))

# returns index array for vector of length len() l
# thus highest idx is l-1
def idx_a(l,sts=1):
    return np.arange(0,l,sts)
#    return np.asarray(range(l))
    
# returns index array between on and off (both included)
def idx_seg(on,off,sts=1):
    return np.arange(on,off+1,sts)

# returns index iterable of list L
def idx(l):
    return range(len(l))


# copy key-value pairs from dict A to dict B
def cp_dict(a,b):
    for x in list(a.keys()):
        b[x] = a[x]
    return b

# returns dimension of numpy array
# IN:
#   array
# OUT:
#   int for number of dims
def ndim(x):
    return len(x.shape)

# transforms TextGrid tier to 2 arrays
# point -> 1 dim + lab
# interval -> 2 dim (one row per segment) + lab
# IN:
#   t: tg tier (by tg_tier())
# OUT:
#   x: 1- or 2-dim array of time stamps
#   lab: corresponding labels
# REMARK:
#   empty intervals are skipped
def tg_tier2tab(t):
    x = ea()
    lab = []
    if 'intervals' in t:
        for i in numkeys(t['intervals']):
            z = t['intervals'][i]
            if len(z['text'])==0:
                continue
            x = push(x,[z['xmin'],z['xmax']])
            lab.append(z['text'])
    else:
        for i in numkeys(t['points']):
            z = t['points'][i]
            x = push(x,z['time'])
            lab.append(z['mark'])
    return x, lab


# transforms table to TextGrid tier
# IN:
#    t - numpy 1- or 2-dim array with time info
#    lab - list of labels <[]>
#    specs['class'] <'IntervalTier' for 2-dim, 'TextTier' for 1-dim>
#         ['name']
#         ['xmin'] <0>
#         ['xmax'] <max tab>
#         ['size'] - will be determined automatically
#         ['lab_pau'] - <''>
# OUT:
#    dict tg tier (see i_tg() subdict below myItemIdx) 
# for 'interval' tiers gaps between subsequent intervals will be bridged
# by lab_pau
def tg_tab2tier(t,lab,specs):
    tt = {'name':specs['name']}
    nd = ndim(t)
    # 2dim array with 1 col
    if nd==2: nd = ncol(t)
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
        tt['xmin']=0
    else:
        tt['xmin'] = specs['xmin']
    if 'xmax' not in specs:
        if nd==1:
            tt['xmax'] = t[-1]
        else:
            tt['xmax'] = t[-1,1]
    else:
        tt['xmax'] = specs['xmax']
    # point tier content
    if nd==1:
        for i in idx_a(len(t)):
            # point tier content might be read as [[x],[x],[x],...] or [x,x,x,...]
            if of_list_type(t[i]):
                z = t[i,0]
            else:
                z = t[i]
            tt['points'][i+1] = {'time':z, 'mark':lab[i]}
        tt['size']=len(t)
    # interval tier content
    else:
        j=1
        # initial pause
        if t[0,0] > tt['xmin']:
            tt['intervals'][j]={'xmin':tt['xmin'],'xmax':t[0,0],'text':lp}
            j+=1
        for i in idx_a(len(t)):
            # pause insertions
            if ((j-1 in tt['intervals']) and
                t[i,0]>tt['intervals'][j-1]['xmax']):
                tt['intervals'][j]={'xmin':tt['intervals'][j-1]['xmax'],
                                    'xmax':t[i,0],'text':lp}
                j+=1
            tt['intervals'][j]={'xmin':t[i,0],'xmax':t[i,1],'text':lab[i]}
            j+=1
        # final pause
        if tt['intervals'][j-1]['xmax'] < tt['xmax']:
            tt['intervals'][j]={'xmin':tt['intervals'][j-1]['xmax'],
                                'xmax':tt['xmax'],'text':lp}
            j+=1 # so that uniform 1 subtraction for size
        # size
        tt['size']=j-1
    return tt

# add tier to TextGrid
# IN:
#   tg dict from i_tg(); can be empty dict
#   tier subdict to be added:
#       same dict form as in i_tg() output, below 'myItemIdx'
#   opt 
#      ['repl'] <True> - replace tier of same name 
# OUT:
#   tg updated
# REMARK:
#   !if generated from scratch head xmin and xmax are taken over from the tier
#    which might need to be corrected afterwards! 
def tg_add(tg,tier,opt={'repl':True}):

    # from scratch
    if 'item_name' not in tg:
        fromScratch = True
        tg = {'name':'', 'format':'long', 'item_name':{}, 'item':{},
              'head':{'size':0,'xmin':0,'xmax':0,'type':'ooTextFile'}}
    else:
        fromScratch = False

    # tier already contained?
    if (opt['repl']==True and (tier['name'] in tg['item_name'])):
        i = tg['item_name'][tier['name']]
        tg['item'][i] = tier
    else:
        # item index
        ii = numkeys(tg['item'])
        if len(ii)==0: i=1
        else: i = ii[-1]+1
        tg['item_name'][tier['name']] = i
        tg['item'][i] = tier
        tg['head']['size'] += 1

    if fromScratch and 'xmin' in tier:
        for x in ['xmin','xmax']:
            tg['head'][x] = tier[x]

    return tg

# transform interchange format to TextGrid
# transforms table to TextGrid tier
# IN:
#    an: annot dict e.g. by i_par() or i_copa_xml()
# OUT:
#    tg: TextGrid dict
# for 'interval' tiers gaps between subsequent intervals are bridged
# only tiers with time information are taken over!
def inter2tg(an):
    typeMap = {'segment':'IntervalTier', 'event':'TextTier'}
    itemMap = {'segment':'intervals', 'event':'points'}
    tg = {'type':'TextGrid','format':'long',
          'head':{'xmin':0,'xmax':-1,'size':0},
          'item_name':{},'item':{}}
    # item idx
    ii=1
    # over tiers
    for x in sorted(an.keys()):
        # skip tier without time info
        if an[x]['type'] not in typeMap:
            continue
        tg['head']['size']+=1
        tg['item_name'][x]=ii
        tg['item'][ii]={'name':x, 'size':0, 'xmin':0, 'xmax':-1,
                        'class':typeMap[an[x]['type']]}
        z = itemMap[an[x]['type']]
        # becomes tg['item'][ii]['points'|'intervals']
        tt={}
        # point or interval tier content
        if z=='points':
            # j: tier items (j+1 in TextGrid output)
            for j in numkeys(an[x]['items']):
                y = an[x]['items'][j]
                tt[j+1]={'time':y['t'],
                         'mark':y['label']}
                tg['item'][ii]['size'] += 1
                tg['item'][ii]['xmax'] = y['t']
        else:
            j=1
            # initial pause
            y = an[x]['items'][0]
            if y['t_start'] > 0:
                tt[j]={'xmin':tg['item'][ii]['xmin'],
                       'xmax':y['t_start'], 'text':''}
                j+=1
            # i: input tier idx, j: output tier idx
            for i in numkeys(an[x]['items']):
                y = an[x]['items'][i]
                # pause insertions
                if ((j-1 in tt) and
                    y['t_start'] > tt[j-1]['xmax']):
                    tt[j]={'xmin':tt[j-1]['xmax'],
                           'xmax':y['t_start'], 'text':''}
                    j+=1
                tt[j]={'xmin':y['t_start'],'xmax':y['t_end'],'text':y['label']}
                tg['item'][ii]['xmax']=tt[j]['xmax']
                j+=1
            
            # size
            tg['item'][ii]['size']=j-1

        # copy to interval/points subdict
        tg['item'][ii][z] = tt

        # xmax
        tg['head']['xmax'] = max(tg['head']['xmax'],tg['item'][ii]['xmax'])
        ii+=1

    # uniform xmax, final silent interval
    for ii in tg['item']:
        # add silent interval
        if (tg['item'][ii]['class']=='IntervalTier' and
            tg['item'][ii]['xmax'] < tg['head']['xmax']):
            tg['item'][ii]['size'] += 1
            j = max(tg['item'][ii]['intervals'])+1
            xm = tg['item'][ii]['intervals'][j-1]['xmax']
            tg['item'][ii]['intervals'][j] = {'text':'','xmin':xm,
                                              'xmax':tg['head']['xmax']}
        tg['item'][ii]['xmax']=tg['head']['xmax']
    return tg


# as inter2tg() but omitting header item
# IN:
#   par dict from i_par()
# OUT:
#   tg dict as with i_tg()
def par2tg(par_in):
    par = cp.deepcopy(par_in)
    del par['header']
    return inter2tg(par)


# returns item-related subkeys: 'intervals', 'text' or 'points', 'mark'
# IN:
#   t tier
# OUT:
#   x key1
#   y key2
def tg_item_keys(t):
    if 'intervals' in t:
        return 'intervals', 'text'
    return 'points', 'mark'


# wrapper around tg2inter() + adding 'header' item / 'class' key
# WARNING: information loss! MAU tier does not contain any wordIdx reference!
# IN:
#  tg: dict read by tg_in
#  fs: sample rate
# OUT:
#  par: par dict (copa-xml format)
# REMARK: output cannot contain wordIdx refs!
#    thus MAU is class 2 in this case, without 'i' field
def tg2par(tg,fs):
    par = tg2inter(tg)
    # add 'class' key
    for tn in par:
        par[tn]['class'] = par_class(par,tn)
    par['header'] = par_header(fs)
    return par

# returns class of a par tier to be added to dict coming from tg2inter()
# IN:
#  par: par dict
#  tn: tierName
# OUT:
#  c: tierClass
def par_class(par,tn):
    t = par[tn]['type']
    if t=='null':
        return 1
    n = numkeys(par[tn]['items'])
    if 'i' in par[tn]['items'][n[0]]:
        wordRef = True
    else:
        wordRef = False
    if t=='event':
        if wordRef:
            return 5
        return 3
    else:
        if wordRef:
            return 4
        return 2

# returns header dict for par dict
# IN:
#   fs: sample rate
# OUT:
#   h: header dict
def par_header(fs):
    return {'type':'header','class':0,'items':{'SAM':fs}}

# transforms textgrid dict to interchange format
#  same as i_par(), i_copa_xml() output
# IN:
#   tg dict from i_tg
#   [opt]:
#     snap: <False> (if True, also empty-label intervals are kept)
# OUT:
#   an dict:
# event tier:
#   dict [myTierName]['type'] = 'event'
#                    ['items'][itemIdx]['label']
#                                      ['t']
#                                      ['class']
# segment tier
#        [myTierName]['type'] = 'segment'
#                    ['items'][itemIdx]['label']
#                                      ['t_start']
#                                      ['t_end']
def tg2inter(tg,opt={}):
    opt = opt_default(opt,{'snap':False})
    an = {}
    # over tier names
    for tn in tg['item_name']:
        t = tg['item'][tg['item_name'][tn]]
        an[tn]={'items':{}}
        # item idx in an[tn]['items']
        ii=0
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
            if k=='intervals':
                if len(t[k][i]['text'])==0:
                    if not opt['snap']:
                        continue
                an[tn]['items'][ii]={'label':t[k][i]['text'],
                                     't_start':t[k][i]['xmin'],
                                     't_end':t[k][i]['xmax']}
            else:
                an[tn]['items'][ii]={'label':t[k][i]['mark'],
                                     't':t[k][i]['time']}
            ii+=1
    return an

# TextGrid from file
# IN:
#   s fileName
# OUT:
#   dict tg
#      type: TextGrid
#      format: short|long
#      name: s name of file
#      head: hoh
#          xmin|xmax|size|type
#      item_name -> 
#              myTiername -> myItemIdx
#      item
#         myItemIdx ->        (same as for item_name->myTiername)
#              class
#              name
#              size
#              xmin
#              xmax
#              intervals   if class=IntervalTier
#                    myIdx -> (xmin|xmax|text) -> s
#              points
#                    myIdx -> (time|mark) -> s
def i_tg(ff):
    if i_tg_format(ff) == 'long':
        return i_tg_long(ff)
    else:
        return i_tg_short(ff)

# TextGrid short format input
# IN:
#  s fileName
# Out:
#  dict tg (see i_tg)
def i_tg_short(ff):
    tg = {'name':ff, 'format':'short', 'head':{},
          'item_name':{},'item':{},'type':'TextGrid'}
    (key,fld,skip,state,nf)=('head','xmin',True,'head',tg_nf())
    idx = {'item':0, 'points':0, 'intervals':0}
    with open(ff, encoding='utf-8') as f:
        for z in f:
            z = re.sub('\s*\n$','',z)
            if re.search('object\s*class',z,re.I):
                fld = nf[state]['#']
                skip = False
                continue
            else:
                if ((skip==True) or re.search('^\s*$',z) or
                    re.search('<exists>',z)):
                    continue
            if re.search('(interval|text)tier',z,re.I):
                if re.search('intervaltier',z,re.I):
                    typ = 'interval'
                else:
                    typ = 'text'
                z = re.sub('"','',z)
                key = 'item'
                state = 'item'
                fld = nf[state]['#']
                idx[key]+=1
                idx['points'] = 0
                idx['intervals'] = 0
                if not (idx[key] in tg[key]):
                    tg[key][idx[key]] = {}
                tg[key][idx[key]][fld] = z
                if re.search('text',typ,re.I):
                    subkey = 'points'
                else:
                    subkey = 'intervals'
                fld = nf[state][fld]
            else:
                z = re.sub('"','',z)
                if fld == 'size':
                    z=int(z)
                elif fld in ['xmin','xmax','time']:
                    z=float(z)
                if state == 'head':
                    tg[key][fld] = z
                    fld = nf[state][fld]
                elif state == 'item':
                    tg[key] = add_subdict(tg[key],idx[key])
                    tg[key][idx[key]][fld] = z
                    if fld=='name':
                        tg['item_name'][z] = idx[key]
                    # last fld of item reached
                    if nf[state][fld] == '#':
                        state = subkey
                        fld = nf[state]['#']
                    else:
                        fld = nf[state][fld]
                elif re.search('(points|intervals)',state):
                    # increment points|intervals idx if first field adressed
                    if fld == nf[state]['#']:
                        idx[subkey]+=1
                    tg[key][idx[key]] = add_subdict(tg[key][idx[key]],subkey)
                    tg[key][idx[key]][subkey] = add_subdict(tg[key][idx[key]][subkey],idx[subkey])
                    tg[key][idx[key]][subkey][idx[subkey]][fld] = z
                    if nf[state][fld] == '#':
                        fld = nf[state]['#']
                    else:
                        fld = nf[state][fld]
    return tg

# TextGrid long format input
# IN:
#  s fileName
# OUT:
#  dict tg (see i_tg)
def i_tg_long(ff):
    tg = {'name':ff, 'format':'long', 'head':{},
          'item_name':{},'item':{}}
    (key,skip)=('head',True)
    idx = {'item':0, 'points':0, 'intervals':0}
    with open(ff, encoding='utf-8') as f:
        for z in f:
            z = re.sub('\s*\n$','',z)
            if re.search('object\s*class',z,re.I):
                skip = False
                continue
            else:
                if ((skip==True) or re.search('^\s*$',z) or
                    re.search('<exists>',z)):
                    continue
            if re.search('item\s*\[\s*\]:?',z,re.I):
                key = 'item'
            elif re.search('(item|points|intervals)\s*\[(\d+)\]\s*:?',z,re.I):
                m = re.search('(?P<typ>(item|points|intervals))\s*\[(?P<idx>\d+)\]\s*:?',z)
                i_type = m.group('typ').lower()
                idx[i_type] = int(m.group('idx'))
                if i_type == 'item':
                    idx['points'] = 0
                    idx['intervals'] = 0
            elif re.search('([^\s+]+)\s*=\s*\"?(.*)',z):
                m = re.search('(?P<fld>[^\s+]+)\s*=\s*\"?(?P<val>.*)',z)
                (fld,val)=(m.group('fld').lower(),m.group('val'))
                fld = re.sub('number','time',fld)
                val = re.sub('[\"\s]+$','',val)
                # type cast
                if fld == 'size':
                    val = int(val)
                elif fld in ['xmin','xmax','time']:
                    val = float(val)
                # head specs
                if key == 'head':
                    tg[key][fld]=val
                else:
                    # link itemName to itemIdx
                    if fld == 'name':
                        tg['item_name'][val] = idx['item']
                    # item specs
                    if ((idx['intervals']==0) and (idx['points']==0)):
                        tg[key] = add_subdict(tg[key],idx['item'])
                        tg[key][idx['item']][fld]=val
                    # points/intervals specs
                    else:
                        tg[key] = add_subdict(tg[key],idx['item'])
                        tg[key][idx['item']] = add_subdict(tg[key][idx['item']],i_type)
                        tg[key][idx['item']][i_type] = add_subdict(tg[key][idx['item']][i_type],idx[i_type])
                        tg[key][idx['item']][i_type][idx[i_type]][fld]=val
    return tg


# tg next field init
def tg_nf():
    return {'head':
            {'#':'xmin',
             'xmin':'xmax',
             'xmax':'size',
             'size':'#'},
            'item':
            {'#':'class',
             'class':'name',
             'name':'xmin',
             'xmin':'xmax',
             'xmax':'size',
             'size':'#'},
            'points':
            {'#':'time',
             'time':'mark',
             'mark':'#'},
            'intervals':
            {'#':'xmin',
             'xmin':'xmax',
             'xmax':'text',
             'text':'#'}}

# decides whether TextGrid is in long or short format
# IN:
#   s textGridfileName
# OUT:
#   s 'short'|'long'
def i_tg_format(ff):
    with open(ff, encoding='utf-8') as f:
        for z in f:
            if re.search('^\s*<exists>',z):
                f.close
                return 'short'
            elif re.search('xmin\s*=',z):
                f.close
                return 'long'
    return 'long'

# recursively adds default fields of dict d to dict c 
#    if not yet specified in c
# IN:
#  c someDict
#  d defaultDict
# OUT:
#  c mergedDict (defaults added to c)
def opt_default(c,d):
    for x in d:
        if x not in c:
            c[x] = d[x]
        elif type(c[x]) is dict:
            c[x] = opt_default(c[x],d[x])
    return c

# removes elements of set R from list X
# IN:
#  x list
#  r string or set of items to be removed
# OUT:
#  y: x purified
def list_rm(x,r):
    y = []
    r = set(r)
    for e in x:
        if e in r:
            continue
        y.append(e)
    return y

# removes pause labels from input list x
def rm_pau(x):
    return list_rm(x,{'<p:>','<p>','<P>'})

# length adjustment
# IN:
#   x: list
#   l: required length
# OUT:
#   x adjusted
#  if x is longer than l, x[0:l] is returned
#  if x is shorter than l: horizontal extrapolation
def halx(x,l):
    if len(x)==l:
        return x
    if len(x)>l:
        return x[0:l]
    
    if len(x)==0:
        a=0
    else:
        a=x[-1]
    if type(x) is np.ndarray:
        while len(x)<l:
            x = np.append(x,a)
    else:
        while len(x)<l:
            x.append(a)
    return x
            
# hack: length adjustment
# IN:
#   x list
#   y list
# OUT:
#   x,y shortened to same length if needed
def hal(x,y):
    if len(x)>len(y):
        x=x[0:len(y)]
    elif len(y)>len(x):
        y=y[0:len(x)]
    return x, y

def hal_old(x,y):
    while(len(x)>len(y)):
        x=x[0:len(x)-1]
    while(len(y)>len(x)):
        y=y[0:len(y)-1]
    return x, y


# diverse variable checks and respective reactions
# IN:
#  c['var'] - any variable or variable container
#   ['env'] - main context of variable check, e.g. 'copasul'
#   ['spec']- optional, string or dictionary specifying sub-contexts,
#             actions, etc.
def check_var(c):

    ## copasul
    if c['env']=='copasul':
        copa = c['var']
        s = c['spec']

        # clustering
        if s['step']=='clst':
            dom = s['dom']
            if ((dom not in copa['clst']) or
                ('c' not in copa['clst'][dom]) or
                len(copa['clst'][dom]['c'])==0):
                sys.exit("ERROR! Clustering of {} contours requires stylization step.\nSet navigate.do_styl_{} to 1.".format(dom,dom))

        # styl - dom = gnl|glob|loc|bnd|rhy...
        elif s['step']=='styl':
            dom = s['dom']

            # copa['data'][ii][i] available?
            # -> first file+channelIdx (since in file/channel-wise augmentation steps does not start from 
            #       copa is initialized for single varying ii and i)
            ii,i,err = check_var_numkeys(copa)
            if err:
                sys.exit("ERROR! {} feature extraction requires preprocessing step.\nSet navigate.do_preproc to 1".format(dom))

            # preproc fields available?
            if check_var_copa_preproc(copa,ii,i):
                sys.exit("ERROR! {} feature extraction requires preprocessing step.\nSet navigate.do_preproc to 1".format(dom))

            # domain field initialization required? (default True)
            if (('req' not in s) or (s['req']==True)):
                if check_var_copa_dom(copa,dom,ii,i):
                    sys.exit("ERROR! {} feature extraction requires preprocessing step.\nSet navigate.do_preproc to 1".format(dom))

            # dependencies on other subdicts
            if 'dep' in s:
                for x in s['dep']:
                    if check_var_copa_dom(copa,x,ii,i):
                        sys.exit("ERROR! {} feature extraction requires {} features.\nSet navigate.do_{} to 1".format(dom,x,x))
                    
            # ideosyncrasies
            if re.search('^rhy',dom):
                if ('rate' not in copa['data'][ii][i]):
                    sys.exit("ERROR! {} feature extraction requires an update of the preprocessing step. Set navigate.do_preproc to 1".format(dom))
            if dom=='bnd':
                if ((copa['config']['styl']['bnd']['residual']) and
                    ('r' not in copa['data'][0][0]['f0'])):
                    sys.exit("ERROR! {} feature extraction based on f0 residuals requires a previous global contour stylization so that the register can be subtracted from the f0 contour. Set navigate.do_styl_glob to 1, or set styl.bnd.residual to 0".format(dom))

## check blocks called by check_var()
# preproc fields given?
# returns True in case of violation
def check_var_copa_preproc(copa,ii,i):
    if (('f0' not in copa['data'][ii][i]) or
        ('t' not in copa['data'][ii][i]['f0']) or
        ('y' not in copa['data'][ii][i]['f0'])):
        return True
    return False

# checks whether copa['data'][x][y] exists and returns lowest numkeys and err True|False
# IN:
#   copa
# OUT:
#   i: lowest numkey in copa['data']
#   j: lowest numkey in copa['data'][i]
#   err: True if there is no copa['data'][i][j], else False
def check_var_numkeys(copa):
    if type(copa['data']) is not dict: return True 
    a = numkeys(copa['data'])
    if len(a)==0 or (type(copa['data'][a[0]]) is not dict):
        return -1, -1, True
    b = numkeys(copa['data'][a[0]])
    if len(b)==0 or (type(copa['data'][a[0]][b[0]]) is not dict):
        return -1, -1, True
    return a[0], b[0], False

# domain substruct initialized?
def check_var_copa_dom(copa,dom,ii,i):
    if dom not in copa['data'][ii][i]:
        return True
    return False

# to be extended signal preproc function
# IN:
#   y signal vector
#   opt['rm_dc'] - <True> centralize
def sig_preproc(y,opt={}):
    dflt = {'rm_dc':True}
    opt = opt_default(opt,dflt)
    # remove DC
    if opt['rm_dc']==True:
        y = y-np.mean(y)

    return y

# returns low number as probability for unseen events
def prob_oov():
    return np.exp(-100)


# assigns prob to event
# IN:
#  w - event
#  p - prob dict (from list2prob())
# OUT:
#  pw - probability of w
def assign_prob(w,p):
    if w in p:
        return p[w]
    return prob_oov()


# concats dir and stem of fsys-subdirectory
# IN:
#   opt with subdir ['fsys'][s][dir|stm]
#   s - subdir name (e.g. 'out', 'export' etc
def fsys_stm(opt,s):
    fo = os.path.join(opt['fsys'][s]['dir'],
                      opt['fsys'][s]['stm'])
    return fo


# prints stem of current f0 file in copasul framework
def verbose_stm(copa,ii):
    if 'data' in copa:
        print(copa['data'][ii][0]['fsys']['f0']['stm'])
    else:
        print(copa[ii][0]['fsys']['f0']['stm'])

# returns f0 segment from vector according to copa time specs
# IN:
#   copa
#   dom 'glob'|'loc'
#   ii fileIdx
#   i channelIdx
#   j segmentIdx
#   t time vector of channel i <[]>
#   y f0 vector of channel i   <[]>
# OUT:
#   ys f0-segment in segment j
def copa_yseg(copa,dom,ii,i,j,t=[],y=[]):
    if len(t)==0:
        t = copa['data'][ii][i]['f0']['t']
        y = copa['data'][ii][i]['f0']['y']
    tb = copa['data'][ii][i][dom][j]['t']
    yi = find_interval(t,tb[[0,1]])
    return y[yi]

# copa xml output (appended if file exists, else from scratch)
# IN:
#   an - dict generated by i_copa_xml()
#   f  - file name
# OUT:
#   file output xml into f
def o_copa_xml(an,f):
    # identation
    i1 = "  "
    i2 = "{}{}".format(i1,i1)
    i3 = "{}{}".format(i2,i1)
    i4 = "{}{}".format(i2,i2)
    i5 = "{}{}".format(i3,i2)
    # subfields for tier type
    fld = {'event':['label','t'], 'segment':['label','t_start','t_end']}
    # output
    h = open(f,mode='w',encoding='utf-8')
    h.write("{}\n<annotation>\n{}<tiers>\n".format(xml_hd(),i1))
    # over tiers
    for x in sorted(an.keys()):
        h.write("{}<tier>\n".format(i2))
        h.write("{}<name>{}</name>\n{}<items>\n".format(i3,x,i3))
        # over items
        for i in numkeys(an[x]['items']):
            h.write("{}<item>\n".format(i4))
            for y in fld[an[x]['type']]:
                z = an[x]['items'][i][y]
                if y=='label':
                    z = xs.escape(z)
                h.write("{}<{}>{}</{}>\n".format(i5,y,z,y))
            h.write("{}</item>\n".format(i4))
        h.write("{}</items>\n{}</tier>\n".format(i3,i2))
    h.write("{}</tiers>\n</annotation>\n".format(i1))
    h.close()
    return

# returns xml header
def xml_hd():
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"


# maps >0|0 to True|False
def int2bool(x):
    if x>0:
        return True
    elif x==0:
        return False
    else:
        sys.exit("{} cannot be transformed to booelan".format(x))


# returns different forms of False
# IN:
#   typ: <'bool'>|'str'|'num'
# OUT:
#   False|''|0
def retFalse(typ='bool'):
    if typ=='str': return ''
    if typ=='num': return 0
    return False
        

# returns file content as 1-dim list of strings (one element per row)
# IN:
#   filename
# OUT:
#   list
def i_list(f):
    with open(f, encoding='utf-8') as h:
        l = [x.strip() for x in h]
        h.close()
    return l

# reads any text file as 1-dim or 2-dim list (of strings)
# Remark: no np.asarray, thus element indexing is x[i][j],
#         i.e. [i,j] does not work 
# IN:
#   s fileName
#   sep separator (<''> -> whitespace split)
#       regex need to be marked by 'r:sep'
#       e.g.: 'r:\\t+' - split at tabs (double backslash needed!)
#   frm <'2d'>|'1d'
# OUT:
#   lol x (sublists: rows splitted at whitespace)
#      or l x (rows concat to 1 list)
def i_lol(f,sep='',frm='2d'):
    if re.search('^r:',sep):
        is_re = True
        sep = re.sub('^r:','',sep)
    else:
        is_re = False
    d = []
    with open(f, encoding='utf-8') as h:
        for z in h:
            # default or customized split
            if len(sep)==0:
                x = z.split()
            else:
                # whitspace standardizing
                z = str_standard(z)
                # regex or string split
                if is_re:
                    x = re.split(sep,z)
                else:
                    x = z.split(sep)
            if len(x)>0:
                if frm=='2d':
                    d.append(x)
                else:
                    for y in x:
                        d.append(y)
        h.close()
    return d


# standardizes strings:
#   removes initial, final (and multiple) whitespaces
# IN:
#   x: someString
#   a: <False>; if True: replace all whitespace(-sequences) by single blank
# OUT:
#   x: standardizedString
def str_standard(x,a=False):
    x = re.sub('^\s+','',x)
    x = re.sub('\s+$','',x)
    if a:
        x = re.sub('\s+',' ',x)
    return x

# BAS partiture file output
# IN:
#   par: dict in i_copa_xml format (read by i_par)
#   fil: file name
# OUT:
#   par file output to f
# Tier class examples
#   1: KAS:	0	v i:6
#   2: IPA:    4856    1228    @
#   3: LBP: 1651 PA
#   4: USP:    3678656 14144   48;49   PAUSE_WORD
#   5: PRB:    54212    5   TON: H*; FUN: NA
def o_par(par,f):
    # sample rate
    fs = par['header']['items']['SAM']
    # init output list
    o = [["LHD: Partitur 1.3"],
         ["SAM: {}".format(fs)],
         ["LBD:"]]
    for tn in sorted_keys(par):
        if tn=='header':
            continue
        tier = cp.deepcopy(par[tn])
        #print(sorted(tier.keys()))
        c = tier['class']
        # add to par output list
        for i in numkeys(tier['items']):
            a = o_par_add(tier,tn,i,c,fs)
            # unsnap for segments
            o = o_par_unsnap(o,a,c)
            o.append(a)

    # -> 1dim list
    for i in idx(o):
        #print(o[i])
        o[i] = ' '.join(map(str,o[i]))

    # output
    output_wrapper(o,f,'list')
    return

# for segment tiers reduces last time offset in o by 1 sample
# if equal to onset in a
# IN:
#   o: par list output so far
#   a: new list to add
#   c: tier class
# OUT:
#   o: last sublist updated 
def o_par_unsnap(o,a,c):
    # no MAU segment tier
    if c!=4 or a[0] != 'MAU:':
        return o
    # diff tiers
    if o[-1][0]!=a[0]:
        return o
    t = o[-1][1]
    dur = o[-1][2]
    if t+dur != a[1]-1:
        o[-1][2] = a[1]-1-t
    return o
        

# adds row to par output file
# IN:
#   tier: tier subdict from par
#   tn: current tier name
#   i: current item index in tier
#   c: tier class 1-5
#   fs: sample rate
def o_par_add(tier,tn,i,c,fs):
    x = tier['items'][i]
    s = "{}:".format(tn)
    if c==1:
        # no time reference
        a = [s,x['i'],x['label']]
    elif c==2 or c==4:
        # segment
        t = int(x['t_start']*fs)
        dur = int((x['t_end']-x['t_start'])*fs)
        if c==2:
            a = [s,t,dur,x['label']]
        else:
            a = [s,t,dur,x['i'],x['label']]
    elif c==3 or c==5:
        # event
        t = int(x['t']*fs)
        if c==3:
            a = [s,t,x['label']]
        else:
            a = [s,t,x['i'],x['label']]
    return a

# BAS partiture file input
# IN:
#   s fileName
#   opt
#       snap: <False>
#         if True off- and onset of subsequent intervals in MAU are set equal
#         (of use for praat-TextGrid conversion so that no empty segments
#          are outputted)
#       header: <True>
#       fs: sample rate, needed if no header provided, i.e. header=False
# OUT (same as i_copa_xml + 'class' for tier class and 'i' for word idx)
# event tier:
#   dict [myTierName]['type'] = 'event'
#                    ['class'] = 3,5
#                    ['items'][itemIdx]['label']
#                                      ['t']
#                                      ['i']
# segment tier
#        [myTierName]['type'] = 'segment'
#                    ['class'] = 2,4
#                    ['items'][itemIdx]['label']
#                                      ['t_start']
#                                      ['t_end']
#                                      ['i']
# no time info tier
#        [myTierName]['type'] = 'null'
#                    ['class'] = 1
#                    ['items'][itemIdx]['label']
#                                      ['i']
# header
#        [header]['type'] = 'header'
#                ['class'] = 0
#                ['items']['SAM'] = mySampleRate
# symbolic reference to word idx 'i' is always of type 'str'
#    (since it might include [,;])
# for all other tier classes t, t_start, t_end are floats; unit: seconds
# Only tiers defined in dict tt are considered
# Tier class examples
#   1: KAS:	0	v i:6
#   2: IPA:    4856    1228    322     @
#   3: LBP: 1651 PA
#   4: USP:    3678656 14144   48;49   PAUSE_WORD
#   5: PRB:    54212    5   TON: H*; FUN: NA
def i_par(f,opt={}):
    opt = opt_default(opt,{'snap':False,'header':True})
    if (opt['header']==False and ('fs' not in opt)):
        sys.exit('specify sample rate fs for headerless par files.')
    
    par = dict();
    # tier classes (to be extended if needed)
    tc = {'KAN':1, 'KAS':1, 'ORT':1, 'MAU':4, 'DAS':1,
          'PRB':5, 'PRS':1, 'LBP':3, 'LBG':3, 'PRO':1,
          'POS':1, 'TRN':4, 'TRS':1, 'PRM':3, 'MAS':4}
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
        elif headOff==False or (not re.search(':$',z[0])) :
            continue
        tn = re.sub(':$','',z[0])
        if tn not in tc: continue
        if tn not in par:
            if tc[tn] == 1:
                typ = 'null'
            elif tc[tn] in {3,5}:
                typ = 'event'
            else: 
                typ = 'segment'
            par[tn] = {'type':typ,'items':{},'class':tc[tn]}
            ii[tn] = 0

        par[tn]['items'][ii[tn]]={}
        if tc[tn]==1:
            par[tn]['items'][ii[tn]]['label'] = js.join(z[2:len(z)])
            par[tn]['items'][ii[tn]]['i'] = z[1]
        elif tc[tn]==2:
            par[tn]['items'][ii[tn]]['label'] = js.join(z[3:len(z)])
            if opt['snap'] and ii[tn]>0:
                par[tn]['items'][ii[tn]]['t_start']=par[tn]['items'][ii[tn]-1]['t_end']
            else:
                par[tn]['items'][ii[tn]]['t_start'] = int(z[1])/fs
            par[tn]['items'][ii[tn]]['t_end'] = (int(z[1])+int(z[2])-1)/fs
        elif tc[tn]==3:
            par[tn]['items'][ii[tn]]['label'] = js.join(z[2:len(z)])
            par[tn]['items'][ii[tn]]['t'] = int(z[1])/fs
        elif tc[tn]==4:
            par[tn]['items'][ii[tn]]['label'] = js.join(z[4:len(z)])
            if opt['snap'] and ii[tn]>0:
                par[tn]['items'][ii[tn]]['t_start']=par[tn]['items'][ii[tn]-1]['t_end']
            else:
                par[tn]['items'][ii[tn]]['t_start'] = int(z[1])/fs
            par[tn]['items'][ii[tn]]['t_end'] = (int(z[1])+int(z[2])-1)/fs
            par[tn]['items'][ii[tn]]['i'] = z[3]
        elif tc[tn]==5:
            par[tn]['items'][ii[tn]]['label'] = js.join(z[3:len(z)])
            par[tn]['items'][ii[tn]]['t'] = int(z[1])/fs
            par[tn]['items'][ii[tn]]['i'] = z[2]
        ii[tn]+=1
        
    # add header subdict
    par['header'] = par_header(fs)

    return par

# returns wrd dict derived from MAU tier
# IN:
#   par: dict from i_par()
# OUT:
#   wrd: dict derived from MAU (<{}>)
#      ['type'] = 'segment'
#      ['class'] = 4
#      ['items'][itemIdx]['label']
#                        ['t_start']
#                        ['t_end']
#   [itemIdx] corresponds to MAU... ['i']
def par_mau2wrd(par):
    if 'MAU' not in par:
        return {}
    w = {'type':'segment','class':4,'items':{}}
    for i in numkeys(par['MAU']['items']):
        j = int(par['MAU']['items'][i]['i'])
        if j < 0: continue
        if j not in w['items']:
            w['items'][j] = {}
            for x in ['label','t_start','t_end']:
                w['items'][j][x] = par['MAU']['items'][i][x]
        else:
            w['items'][j]['label'] = w['items'][j]['label']+' '+par['MAU']['items'][i]['label']
            w['items'][j]['t_end'] = par['MAU']['items'][i]['t_end']
    return w


# transforms 1- or 2dim table to [myTierName]-subdict (see i_copa_xml())
# IN:
#   x: 1- or 2-dim matrix with time info
#   lab: corresp label ([]) if empty uniform label 'x'
# OUT:
#   y dict
#      ['type']: event|segment
#      ['items'][itemIdx]['label']
#                        ['t'] or ['t_start']
#                                 ['t_end']
# call:
#   myAnnot[myTierName] = tab2copa_xml_tier(myTab,myLab)
def tab2copa_xml_tier(x,lab=[]):
    if ncol(x)==2:
        typ = 'segment'
    else:
        typ = 'event'
    tier = {'type':typ, 'items':{}}
    for i in range(len(x)):
        tier['items'][i]={}
        if len(lab)>i:
            tier['items'][i]['label']=lab[i]
        else:
            tier['items'][i]['label']='x'
        if typ == 'segment':
            tier['items'][i]['t_start']=x[i,0]
            tier['items'][i]['t_end']=x[i,1]
        else:
            tier['items'][i]['t']=x[i,0]
    return tier

# copa xml input
# <annotation><tiers><tier><name>
#                          <items><item><label>
#                                       <t_start>
#                                       <t_end>
#                                       <t>
# IN:
#   inputFileName
# OUT:
# event tier:
#   dict [myTierName]['type'] = 'event'
#                    ['items'][itemIdx]['label']
#                                      ['t']
# segment tier
#        [myTierName]['type'] = 'segment'
#                    ['items'][itemIdx]['label']
#                                      ['t_start']
#                                      ['t_end']
def i_copa_xml(f):
    annot={}
    tree = et.parse(f)
    root = tree.getroot()
    tiers = extract_xml_element(root,'tiers')
    for tier in tiers:
        name_el = extract_xml_element(tier,'name')
        name = name_el.text
        annot[name]={'items':{}}
        itemIdx = 0
        for item in tier.iter('item'):
            annot[name]['items'][itemIdx]={'label':extract_xml_element(item,'label','text')}
            t = extract_xml_element(item,'t','text')
            if t:
                annot[name]['type']='event'
                annot[name]['items'][itemIdx]['t']=t
            else:
                annot[name]['type']='segment'
                t_start = extract_xml_element(item,'t_start','text')
                t_end = extract_xml_element(item,'t_end','text')
                annot[name]['items'][itemIdx]['t_start']=t_start
                annot[name]['items'][itemIdx]['t_end']=t_end
            itemIdx+=1
    return annot

# returns element object or its content string
# IN:
#   xml.etree.ElementTree (sub-)object
#   elementName (string)
#   ret 'element'|'string'
# OUT:
#   elementContent (object or string dep on ret)
def extract_xml_element(myTree,elementName,ret='element'):
    for e in myTree:
        if e.tag == elementName:
            if ret=='element':
                return e
            else:
                return e.text
    return False


def log_exit(f_log,msg):
    f_log.write("{}\n".format(msg))
    print(msg)
    f_log.close()
    sys.exit()

# nearest centroid classification
# IN:
#   c dict
#      [myClassKey] -> myCentroidVector
#   d feature matrix (ncol == ncol(c[myClassKey])
#   opt dict
#      ['wgt']  -> feature weights (ncol == ncol(c[myClassKey]))
#      ['attract'] -> class attractiveness (>0 !)
#                  [myClassKey] -> attractiveness <1>
#                  distance d to class is multiplied derived from
#                  1 - attract/sum(attract)
# OUT:
#   a list with answers (+/- numeric depending on key type in C)
def cntr_classif(c,d,opt={}):
    opt = opt_default(opt,{'wgt':np.ones(ncol(d)),'attract':{}})

    # calculate class prior distance from attractiveness
    cdist = {}
    if len(opt['attract'].keys())>0:
        attract_sum = 0
        for x in c:
            if x not in opt['attract']:
                opt['attract'][x]=1
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
            dst = dist_eucl(d[i,:],c[x],opt['wgt']) * cdist[x]
            if (mdst<0 or dst<mdst):
                nc = x
                mdst = dst
        a.append(nc)
    return a
        
# weighted mean
# IN:
#   x: vector
#   w: weight vector of same length
# OUT:
#   y: weighted mean of x
def wgt_mean(x,w):
    return np.sum(w*x)/np.sum(w)

# estimates feature weights for unsupervised learning
# IN:
#   x: n x m feature matrix, columns-variables, rows-observations
#   mtd: <'corr'>, nothing else supported
# OUT
#   w: 1 x m weight vector
# Method 'corr':
#   - for 'informed clustering' with uniform feature trend, e.g. all 
#      variables are expected to have high values for cluster A and
#      low values for cluster B
#   - wgt(feat):
#        if corr(feat,rowMeans)<=0: 0
#        if corr(feat,rowMeans)>0: its corr
# wgts are normalized to w/sum(w)
def feature_weights_unsup(x,mtd='corr'):
    # row means
    m = np.mean(x,axis=1)
    w = np.asarray([])
    for i in range(ncol(x)):
        r = np.corrcoef(x[:,i],m)
        r = np.nan_to_num(r)
        w = np.append(w,max(r[0,1],0))
    # unusable -> uniform weight
    if np.sum(w)==0:
        return np.ones(len(m))
    return w/np.sum(w)
        
# weighted euclidean distance between x and y
# IN:
#   x: array
#   y: array
#   w: weight array <[]>
# OUT:
#   d: distance scalar
def dist_eucl(x,y,w=[]):
    if len(w)==0:
        w = np.ones(len(x))
    q = x-y
    return np.sqrt((w*q*q).sum())

# calculates delta values x[i,:]-x[i-1,:] for values in x
# IN:
#  x: feature matrix
# OUT:
#  dx: delta matrix
# REMARK:
#  first delta is calculated by x[0,:]-x[1,:]
def calc_delta(x):
    if len(x)<=1:
        return x
    dx = np.asarray([x[0,:]-x[1,:]])
    for i in range(1,len(x)):
        dx = push(dx, x[i,:]-x[i-1,:])
    return dx

# table merge
# always returns 2-dim nparray
# input can be 1- or 2-dim
# IN:
#   x nparray
#   y nparray
#   a axis <0>  (0: append rows, 1: append columns)
# OUT:
#   z = [x y]
def cmat(x,y,a=0):
    if len(x)==0 and len(y)==0:
        return ea()
    elif len(x)==0:
        return lol(y)
    elif len(y)==0:
        return lol(x)
    return np.concatenate((lol(x),lol(y)),a)

# returns True if
#   - x is key in dict d
#   - d[x] is non-empty list or string
def non_empty_val(d,x):
    if (x not in d) or len(d[x])==0:
        return False
    return True

# return empty np array(s)
# IN:
#   n: <1> how many (up to 5)
# OUT:
#   n x np.asarray([])
def ea(n=1):
    if n==1:
        return np.asarray([])
    if n==2:
        return np.asarray([]), np.asarray([])
    if n==3:
        return np.asarray([]), np.asarray([]), np.asarray([])
    if n==4:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
    if n==5:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
    return
      
# return empty np array(s) of type int
# IN:
#   n: <1> how many (up to 5)
# OUT:
#   n x np.asarray([])
def eai(n=1):
    if n==1:
        return np.asarray([]).astype(int)
    if n==2:
        return np.asarray([]).astype(int), np.asarray([]).astype(int)
    if n==3:
        return np.asarray([]).astype(int), np.asarray([]).astype(int), np.asarray([]).astype(int)
    if n==4:
        return np.asarray([]).astype(int), np.asarray([]).astype(int), np.asarray([]).astype(int), np.asarray([]).astype(int)
    if n==5:
        return np.asarray([]).astype(int), np.asarray([]).astype(int), np.asarray([]).astype(int), np.asarray([]).astype(int), np.asarray([]).astype(int)
    return

# robustly splits vector in data above '>' and below '<=' value
# so that always two non-empty index vectors are returned (except of empty
# list input)
# IN:
#  x: vector
#  p: splitValue
# OUT:
#  i0, i1: 2 index vectors of items in x <= and > p
def robust_split(x,p):
    if len(x)==1:
        return 0, 0
    if len(x)==2:
        return min_i(x), max_i(x)
    else:
        for px in [p, np.mean(x)]:
            i0 = find(x,'<=',px)
            i1 = find(x,'>',px)
            if min(len(i0),len(i1))>0:
                return i0, i1
    i = np.arange(0,len(x))
    return i,i


# replace strings (read e.g. from json) to logic constants
# IN:
#   x string
# OUT:
#   y logic constant (if available)
def str2logic(x):
    if not is_type(x)=='str':
        return x
    if not re.search('(TRUE|FALSE|NONE)',x):
        return x
    elif x=='TRUE':
        return True
    elif x=='FALSE':
        return False
    return None


# quick hack to return variable type str|int|float|list (incl np array)
# fallback 'unk'
# IN:
#   x var
# OUT:
#   t type of x
def is_type(x):
    tx = str(type(x))
    for t in ['str','int','float','list']:
        if re.search(t,tx):
            return t
    if re.search('ndarray',tx):
        return 'list'
    return 'unk'

# returns key, value pair from dict with maximum value
# (first one found in alphabetically sorted key list)
# IN:
#   h - dict with numeric values
# OUT:
#   k - key with max value
#   v - max value
def max_kv(h):
    k, v = np.nan, np.nan
    for x in sorted_keys(h):
        if np.isnan(v) or h[x]>v:
            k,v = x,h[x]
    return k,v

# returns index of first minimum in list or nparray
# IN:
#   x: list or nparray
# OUT:
#   i: index of 1st minimum
def min_i(x):
    return list(x).index(min(x))

# returns index of first maximum in list or nparray
# IN:
#   x: list or nparray
# OUT:
#   i: index of 1st maximum
def max_i(x):
    return list(x).index(max(x))

# robust 0 for division, log, etc
def rob0(x):
    if x==0:
        return 0.00000001
    return x
  
# returns True if x is list or np.array
def of_list_type(x):
    if (type(x) is list) or (type(x) is np.ndarray):
        return True
    return False

# returns any input as list
# IN:
#  x
# OUT:
#  x if list, else [x]
def aslist(x):
    if type(x) is list:
        return x
    return [x]

# 1-dim -> 2-dim array
# IN:
#   x: 1-dim nparray
# OUT:
#   y: 2-dim array
def two_dim_array(x):
    return x.reshape((-1,x.size))
    
# identifies transcription major classes: 'vow'
# IN:
#   x: phoneme string
#   pat: pattern type 'vow'
#   lng: language 'deu'
# OUT:
#   True if x contains pat, else False 
def trs_pattern(x,pat,lng):
    ref={'vow':{'deu':'[aeiouyAEIOUY269@]'}}
    if re.search(ref[pat][lng],x):
        return True
    return False

# returns '/'
def root_path():
    return os.path.abspath(os.sep)




# wrapper around g2p (local or webservice)
# IN:
#   opt dict
#     all parameters accepted by g2p.pl
#    + local True|False (if False, webservice is called)
# OUT:
#   success True|False
def g2p(opt):

    if opt['local']:
        cmd = "/homes/reichelu/repos/repo_pl/src/prondict/g2p.pl -task apply"
    else:
        cmd = "curl -v -X POST -H 'content-type: multipart/form-data'"
        
    for x in opt:
        if (re.search('local',x) or (x=='o' and (not opt['local']))):
            continue
        y = ''
        if x=='i':
            y = '@'
        if opt['local']:
            z = " -{} {}".format(x,opt[x])
        else:
            z = " -F {}={}{}".format(x,y,opt[x])
        cmd += z

    if not opt['local']:
        cmd += " 'http://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runG2P'"

    return webservice_output(cmd,opt)


# call any BAS webservice and write result in output file
# IN:
#   opt:
#     'tool' -> service url
#     'param' -> parameter dict (param names as required by webservice)
#              !! files to be uploaded already need to have initial @
#   o: output file name
# OUT:
#   success: True|False
#   file: sevice output into o
def bas_webservice(opt,o):
    cmd = "curl -v -X POST -H 'content-type: multipart/form-data'"
    
    # concatenate curl call
    for x in opt['param']:
        cmd += " -F {}={}".format(x,opt['param'][x])
    cmd += " {}".format(opt['tool'])

    # call webservice + output results
    return webservice_output(cmd,{'out':o, 'local':False})




# wrapper around webmaus general
# IN:
#   opt dict
#     .signal: signalFileName
#     .bpf: par file name
#     .out: par output file
#     .lng: language iso
#     .outformat 'TextGrid'|<'mau-append'>
#     .insorttextgrid <true>
#     .inskantextgrid <true>
#     .usetrn <false>
#     .modus <standard>|align   (formerly: canonly true|false)
#     .weight <default>
#     .outipa <false>
#     .minpauslen <5>
#     .insprob <0.0>
#     .mausshift <10.0>
#     .noinitialfinalsilence <false>
#     .local <False>; if true maus is called not as webservice
# OUT:
#     sucess True|False
#     par file written to opt.par_out
def webmaus(opt):
    opt = opt_default(opt,{'outformat':'mau-append',
                           'insorttextgrid':'true',
                           'inskantextgrid':'true',
                           'usetrn':'false',
                           'modus':'standard',
                           'weight':'default',
                           'outipa':'false',
                           'minpauslen':5,
                           'insprob':0,
                           'mausshift':10.0,
                           'noinitialfinalsilence':'false',
                           'local':False})
    optmap = {'OUTFORMAT':'outformat',
              'INSORTTEXTGRID':'insorttextgrid',
              'INSKANTEXTGRID':'inskantextgrid',
              'USETRN':'usetrn',
              'STARTWORD':'startword',
              'ENDWORD':'endword',
              'INSPROB':'insprob',
              'OUTFORMAT':'outformat',
              'MAUSSHIFT':'mausshift',
              'OUTIPA':'outipa',
              'MINPAUSLEN':'minpauslen',
              'MODUS':'modus',
              'LANGUAGE':'lng',
              'BPF':'bpf',
              'WEIGHT':'weight',
              'SIGNAL':'signal',
              'NOINITIALFINALSILENCE':'noinitialfinalsilence'}
        
    if opt['local']:
        cmd = "maus"
    else:
        cmd = "curl -v -X POST -H 'content-type: multipart/form-data'"

    for x in optmap:
        if optmap[x] in opt:
            y = ''
            if re.search('^(BPF|SIGNAL)$',x):
                y = '@'
            if opt['local']:
                z = " {}={}".format(x,opt[optmap[x]])
            else:
                z = " -F {}={}{}".format(x,y,opt[optmap[x]])

            cmd += z

    if opt['local']:
        cmd += " > {}".opt['out']
    else:
        cmd += " 'http://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUS'"

    return webservice_output(cmd,opt)


# collect output from g2p or MAUS
# write into file opt['out'], resp opt['o']
# IN:
#   cmd: service call string
#   opt:
#     fatal -> <True>|False exit if unsuccesful 
#     local -> True|False
#     o(ut) -> outPutFile
# OUT:
#   (if not exited)
#     True if succesful, else false
def webservice_output(cmd,opt):

    opt = opt_default(opt,{'fatal':True})

    print(cmd)
    ans = subprocess.check_output(cmd, shell=True, universal_newlines=True)
    print(ans)

    # local call -> done
    if opt['local']:
        return

    # webservice answer parse
    m = re.search('<downloadLink>(?P<res>(.+?))<\/downloadLink>',ans)

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

    os.system("wget {} -O {}".format(res,o))

    return True


# reads eaf files
# IN:
#  f: fileName
#  opt: <{}>
#     .min_l: minimum segment length in sec
#             (some elan files contain blind segments)
#     .embed: <'none'>,'igc',... makro to cope with corpus-specific issues
# OUT:
#  eaf dict
#     .myTierName
#       .i: tierIndex (starting with 0)
#       .type: 'segment' (so far only segment supported!)
#       .removed: set()  ANNOTATION_IDs that are removed depending on embed
#       .items
#         .myItemIdx
#           .'label'
#           .'t_start' in sec
#           .'t_end'   in sec
#           .'ts_start' - timeSloptID
#           .'ts_end'
#           .'id' annot Id
def i_eaf(f,opt={}):
    opt = opt_default(opt,{'min_l':0, 'embed':'none'})
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
        eaf[tn] = {'type':'segment','i':tier_i,'items':{},'removed':set()}
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
                        xav = extract_xml_element(xaa,'ANNOTATION_VALUE')
                        lab = xav.text
                        if opt['embed'] == 'igc':
                            lab = re.sub('\s*\-\-+\s*',', ',lab)
                            #lab = re.sub('\s*\?\?+\s*','',lab)
                            #lab = re.sub('\(.+?\)','',lab)
                            lab = re.sub('\s*\?\?+\s*',' <usb> ',lab)
                            lab = re.sub('\(.+?\)',' <usb> ',lab)
                            lab = string_std(lab)
                            # remove | sep at start and end of text
                            lab = re.sub('^\s*\|','',lab)
                            lab = re.sub('\|\s*$','',lab)
                            if len(lab)==0:
                                eaf[tn]['removed'].add(aid)
                                continue
                        eaf[tn]['items'][i]={'label':lab,
                                             't_start':t_start,
                                             't_end':t_end,
                                             'ts_start':aa['TIME_SLOT_REF1'],
                                             'ts_end':aa['TIME_SLOT_REF2'],
                                             'id':aid}
                        i+=1
                        #print(t_start,t_end,lab)
                        #stopgo()
    return eaf

# returns eaf time stamp dict
# IN:
#   eaf xml root object
# OUT:
#   ts dict
#     timeStamp -> timeValue (onset already added, in sec)
def i_eaf_ts(root):
    ts = {}
    # time onset (in ms)
    ons = i_eaf_ons(root)
    tsx = extract_xml_element(root,'TIME_ORDER')
    for e in tsx:
        if e.tag == 'TIME_SLOT':
            a = e.attrib
            ts[a['TIME_SLOT_ID']] = (int(a['TIME_VALUE'])+ons)/1000
    return ts

# returns eaf time onset from header
# IN:
#   eaf xml root object
# OUT:
#   int onset (in ms)
def i_eaf_ons(root):
    hd = extract_xml_element(root,'HEADER')
    md = extract_xml_element(hd,'MEDIA_DESCRIPTOR')
    mda = md.attrib
    return int(mda['TIME_ORIGIN'])
  
# standardise strings, i.e. remove initial and final blanks
# and replace multiple by single blanks 
# IN:
#   s string
# OUT:
#   s standardized string
def string_std(s):
    s = re.sub('\s+',' ',s)
    s = re.sub('^\s+','',s)
    s = re.sub('\s+$','',s)
    return s

# empty directory by removing + regenerating
def purge_dir(d):
    if os.path.isdir(d):
        sh.rmtree(d)
        os.makedirs(d)

    return True


# remove nan from array
# IN:
#   x - 1-dim array
# OUT:
#   x without nan
def rm_nan(x):
    return x[~np.isnan(x)]

# linear interpolation over NaN
# IN:
#   y - np array 1-dim
#   r - 'i'|'e'
#       what to return if only NaNs contained:
#        <'i'> - input list
#        'e' - empty list
# OUT:
#   y with NaNs replaced by linint
#     (or input if y consists of NaNs only)
def linint_nan(y,r='i'):
    xi = find(y,'is','nan')

    # no NaN
    if len(xi)==0:
        return y

    # only NaN
    if len(xi)==len(y):
        if r=='i':
            return y
        return ea()

    # interpolate
    xp = find(y,'is','finite')
    yp = y[xp]
    yi = np.interp(xi,xp,yp)
    y[xi] = yi
    return y

# switches elements with idx i and j in list l
# IN:
#   l list
#   i, j indices
# OUT:
#   l updated
def list_switch(l,i,j):
    bi = cp.deepcopy(l[i])
    l[i] = cp.deepcopy(l[j])
    l[j] = bi
    return l

# executes copasul.py from other scripts
# IN:
#    c config filename (full path)
# OUT:
#    as specified in c
def copasul_call(c):
    d = os.path.dirname(os.path.realpath(__file__))
    cmd = os.path.join(d,'copasul.py')
    os.system("{} -c {}".format(cmd,c))

# extended not:
# returns True if key not in dict or dict->key==False
# IN:
#  d: dict
#  k: key
# OUT:
#  b: True|False
def ext_false(d,k):
    if k not in d:
        return False
    if d[k]:
        return True
    return False

# returns True if key in dict or dict->key==True
# IN:
#  d: dict
#  k: key
# OUT:
#  b: True|False
def ext_true(d,k):
    if (k in d) and d[k]:
        return True
    return False

# calls copasul in pipeline
# IN:
#   opt pipeline config
#          must contain ['fsys']['config'][myKey]
#          with location of copasul config file
#   myKey key in opt['fsys']['config'] to be adressed
# OUT:
#   -
def pipeline_copasul_call(opt,x):
    copasul_call(opt['fsys']['config'][x])

# OUT: list of default 10 item color circle + some additional ones
def colors():
    return ['blue', 'orange', 'green', 'red','purple',
            'brown', 'pink', 'gray', 'olive', 'cyan',
            'lime', 'darkgreen', 'maroon', 'salmon',
            'darkcyan','powderblue','lightgreen']

# sorts elements in list x according to their time points in list t
# IN:
#   x - list of elements to be sorted
#   t - list of their time stamps
# OUT:
#   x sorted
def timeSort(x,t):
    for i in range(0,len(x)-1):
        for j in range(i+1,len(x)):
            if t[i]>t[j]:
                b = x[j]
                x[j] = x[i]
                x[i] = b
    return x

# returns True if input string is punctuation mark
def is_punc(s):
    if re.search('^[<>,!\|\.\?:\-\(\);\"\']+$',s):
        return True
    return False

# task: 'expand'|'shorten'
#   from/to iso<->fullform
# so far for all languages supported by snowball stemmer
def lng_map(task='expand'):
    lm = {'nld':'dutch',
          'dan':'danish',
          'eng':'english',
          'fin':'finnish',
          'fra':'french',
          'deu':'german',
          'hun':'hungarian',
          'ita':'italian',
          'nor':'norwegian',
          'porter':'porter',
          'por':'portuguese',
          'ron':'romanian',
          'rus':'russian',
          'spa':'spanish',
          'swe':'swedish'}
    if task=='expand':
        return lm
    else:
        lmr = {}
        for x in lm:
            lmr[lm[x]]=x
        return lmr

# returns condition entropy H(y|x) for input lists x and y
# IN:
#  x
#  y
# OUT:
#  H(y|x)
def lists2condEntrop(x,y):
    c = lists2count(x,y)
    return count2condEntrop(c)

# IN:
#   c: output of lists2count()
#     e.g. c['joint'][x][y]
# OUT:
#   H(Y|X) (beware order!)
def count2condEntrop(c):
    h = 0
    for x in c['joint']:
        nx = c['margin'][x]
        if nx==0:
            continue
        pys = 0
        for y in c['joint'][x]:
            py = c['joint'][x][y]/nx
            if py==0:
                continue
            pys += (py*binlog(py))
        h -= ((nx/c['N'])*pys)
    return h

# returns redundancy given two input lists
# IN:
#  x
#  y
# OUT:
#  r = I(x;y)/(H(x)+H(y)
def lists2redun(x,y):
    ixy = count2mi(lists2count(x,y))
    hx = list2entrop(x)
    hy = list2entrop(y)
    return ixy/(hx+hy)

# returns mutual infor of 2 input lists
# IN:
#   x: list
#   y: list
# OUT:
#   mi: mutual info
def lists2mi(x,y):
    return count2mi(lists2count(x,y))

# returns dict with 2 layers for cooc counts in 2 input lists
# IN:
#   x
#   y (same lengths!)
# OUT:
#   c['joint'][x][y]=myCount f.a. x,y in <xs, ys>
#    ['margin'][x]=myCount
#    ['N']: len(x)
def lists2count(x,y):
    c, n = {'joint':{},'margin':{}}, 0
    for i in idx_a(len(x)):
        for z in [x[i],y[i]]:
            if z not in c['margin']:
                c['margin'][z]=0
        if x[i] not in c['joint']:
            c['joint'][x[i]]=dict()
        if y[i] not in c['joint'][x[i]]:
            c['joint'][x[i]][y[i]]=0
        c['joint'][x[i]][y[i]] += 1
        c['margin'][x[i]] += 1
        c['margin'][y[i]] += 1
    c['N'] = len(x)
    return c

# mutual info based on lists2count() object
# IN:
#   c: output of lists2count()
# OUT:
#   mi: mutual info
def count2mi(c):
    mi=0
    n = c['N']
    for x in c['joint']:
        for y in c['joint'][x]:
            pxy = c['joint'][x][y]/n
            px = c['margin'][x]/n
            py = c['margin'][y]/n
            if min(px,py,pxy)==0:
                continue
            mi += (pxy * binlog(pxy/(py*py)))
    return mi

# wrapper around Praat f0 extractor
# example call: wrapper_py/hgc.py
# needs opt of format config/hgc.json
# F0 extraction
# IN:
#   opt
# OUT:
#   f0 files
def make_f0(opt):
    pth = opt['fsys']['data']
    par = opt['param']['f0']

    # clean up f0 subdir
    sh.rmtree(pth['f0'])
    os.makedirs(pth['f0'])


    os.system("praat {} 0.01 {} {} {} {} wav f0".format(par['tool'],
                                                        par['param']['min'],
                                                        par['param']['max'],
                                                        pth['aud'],pth['f0']))
    return True

### figure init #################################

# init new figure with onclick->next, keypress->exit
# figsize can be customized
# IN:
#   fs tuple <()>
# OUT:
#   figure object
# OUT:
#   figureHandle
def newfig(fs=()):
    if len(fs)==0:
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
