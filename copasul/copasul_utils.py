from collections import Counter
import copy as cp
import csv
import datetime
import json
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import re
import shutil as sh
import scipy.cluster.vq as sc
import sklearn.preprocessing as sp
import scipy.stats as st
import subprocess
import sys
import xml.etree.ElementTree as et
import xml.sax.saxutils as xs


#########################################################
# collection of general purpose functions ###############
#########################################################


'''
 I/O functions
 input_wrapper()
 i_tg()       -> tg dict
 i_par()      -> interchange dict
 i_copa_xml() -> interchange dict
 i_lol()      -> 1- or 2dim list of strings
 i_seg_lab()  -> d.t [[on off]...]
                  .lab [label ...]
 i_keyVal() -> dict z: a;b -> z["a"]="b"
 i_numpy: calls np.loadtxt() returns np.array list of floats
          (1 col -> 1-dim array; else 1 sublist per row)
       'pandas_csv': csv file into dict colName -> colContent (using pandas)
 output_wrapper()
 o_tg()
 o_par()
 o_copa_xml()
 annotation processing
 par_mau2word()
   creates word segment tier from .par file MAU tier
 tg_mau2chunk()
   creates chunk tier from pauses in MAU
 tg_add()
   adds/replaces tier to TextGrid
 tg_tier()
   returns tier from TextGrid
 tg_tn()
   returns list TextGrid tier names
 tg_tierType()
   returns 'points' or 'intervals' (= key to access items in tier)
 tg_mrg()
   select tiers of >=1 TextGrid and create new TextGrid from these tiers
 format transformations
 tg_tab2tier():
   numpy array to TextGrid tier
 tg_tier2tab():
   TextGrid tier to numpy and label array
 tg2inter()
   TextGrid -> interchange format
 inter2tg()
   interchange format -> TextGrid
 tg2par()
   as tg2inter() + added header element with sample rate
 par2tg()
   as inter2tg(), but omitting the header item
 tab2copa_xml_tier()
   numpy array to copa_xml tier
 pipeline: add tiers to existing TextGrid (w/o replacement)
 myTg = myl.input_wrapper(myInputFile,'TextGrid')
 myAdd = myl.input_wrapper(myAddFile,'TextGrid')
 with tier replacement:
 opt = {'repl':True}
 without replacement
 opt = {'repl':False}
 for x in tg_tn(myAdd):
    myTg = tg_add(myTg,tg_tier(myAdd,x),opt)

 basic matrix op functions
 cmat(): 2-dim matrix concat, converts if needed all input to 2-dim arrays
 lol(): ensure always 2-dim array (needed e.g. for np.concatenate of 2-dim and
        1-sim array)
 push(): appending without dimensioning flattening (vs append)
 find(): simulation of matlab find functionality
 find_interval(): indices of values within range, pretty slow
 first_interval(): first index of row containint specific element
              IMPORTANT: ask whether idx is >=0 since -1 indicates not found
'''

def push(x, y, a=0):

    '''
    pushes 1 additional element y to array x (default: row-wise)
      if x is not empty, i.e. not []: yDim must be xDim-1, e.g.
          if x 1-dim: y must be scalar
          if x 2-dim: y must 1-dim
    if x is empty, i.e. [], the dimension of the output is yDim+1
    Differences to np.append:
      append flattens arrays if dimension of x,y differ, push does not
    REMARK: cmat() might be more appropriate if 2-dim is to be returned
    
    Args:
      x: (np.array) (can be empty)
      y: (np.array) (if x not empty, then one dimension less than x)
      a: (int) axis (0: push row, 1: push column)
    
    Returns:
      (np.array) [x y] concatenation
    '''

    if (listType(y) and len(y) == 0):
        return x
    if len(x) == 0:
        return np.array([y])
    return np.concatenate((x, [y]), axis=a)


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


def sec2idx(i, fs, ons=0.0):

    '''
    transforms seconds to numpy array indices (=samplesIdx-1)

    Args:
    i: (float) time in sec
    fs: (int) sampling rate
    ons: (float) time onset to be added in sec

    Returns:
    (int) index
    '''

    return np.round(i * fs + ons - 1).astype(int)


def sec2smp(i, fs, ons=0.0):

    '''
    transforms seconds to sample indices (arrayIdx+1)

    Args:
    i: (float) time in sec
    fs: (int) sampling rate
    ons: (float) time onset to be added in sec

    Returns:
    (int) sample index
    '''
    
    return np.round(i * fs + ons).astype(int)



def idx2sec(i, fs, ons=0):

    '''
    transforms numpy array indices (=samplesIdx-1) to seconds
    
    Args:
    i: (int) index
    fs: (int) sampling rate
    ons: (int) onset index

    Returns:
    s: (float) time in seconds

    '''
    
    return (i + 1 + ons) / fs


def smp2sec(i, fs, ons=0):

    '''
    transforms sample indices (arrayIdx+1) to seconds

    Args:
    i: (int) index
    fs: (int) sampling rate
    ons: (int) onset index

    Returns:
    s: (float) time in seconds
    '''

    return (i + ons) / fs


def seg_list(x):

    '''
    segment list into subsequent same element segments
    
    Args:
      x: (list) 1-dim
    
    Returns:
      y: (list) 2-dim

    example:
      [a, a, b, b, a, c, c] -> [[a, a], [b, b], [a], [c, c]]
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


def lists(typ='register', ret='list'):

    '''
    returns predefined lists or sets (e.g. to address keys in copa dict)

    Args:
    typ: (str)
    ret: (str) return type "set", or "list"

    Returns:
    (set or list) of strings
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

    return set([])


def find(x_in, op, v):

    '''
    similar to matlab "find" for 1-dim data
    
    Args:
     x_in (np.array) 1-dim
     op: (str) > , < , ==, is, isinfimum, issupremum
     v: (float, str) (incl. 'max'|'min'|'nan' etc. in combination with op 'is')
    
    Returns:
     ci (np.array) 1-dim index array
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


def mae(x, y=None):

    '''
    returns mean absolute error of vectors
    or of one vector and zeros (=mean abs dev)
    
    Args:
      x: (np.array)
      y: (np.array), zeros if not provided
    
    Returns:
      (float) man abs error
    '''

    if y is None:
        y = np.zeros(len(x))

    x = np.array(x)
    y = np.array(y)
        
    return np.mean(np.abs(x - y))


def rss(x, y=None):

    '''
    residual squared deviation
    
    Args:
      x: (np.array) data vector
      y: (np.array) prediction vector (e.g. fitted line)
    
    Returns:
      r: (float) residual squared deviation

    '''

    if y is None:
        y = np.zeros(len(x))

    x = np.asarray(x)
    y = np.asarray(y)

    return np.sum((x - y) ** 2)


def aic_ls(x, y, k=3):

    '''
    AIC information criterion for least squares fit
    for model comparison, i.e. without constant terms
    
    Args:
      x: (np.array) underlying data
      y: (np.array) predictions
      k: (ind) number of parameters (<3> for single linear fit)

    Returns:
      aic: (float) AIC value
    '''

    n = len(x)
    r = rss(x, y)
    if r == 0:
        return 2 * k
    aic = 2 * k + n * np.log(r)
    
    return aic


def robust_div(x, y):

    '''
    robust division x / y
    returns np.nan for 0-divisions
    '''
    
    if y == 0 or np.isnan(x) or np.isnan(y):
        return np.nan
    
    return x / y

def robust_corrcoef(x, y):

    '''
    robust wrapper around np.corrcoef
    
    Args:
    x, y: (np.arrays) of same length
    
    Returns:
    r: (float) Pearson correlation coef
    '''

    if len(x) <= 1:
        return 0.0
    elif np.min(x) == np.max(x) and np.min(y) == np.max(y):
        return 1.0
    elif np.min(x) == np.max(x) or np.min(y) == np.max(y):
        return 0.0

    r = np.corrcoef(x, y)
    return r[0, 1]

def mse(x, y=None):

    '''
    mean squared error of vectors x and y
    '''
    
    if y is None:
        y = np.zeros(len(x))

    x = np.array(x)
    y = np.array(y)
        
    return np.mean((x - y) ** 2)


def rmsd(x, y=None):

    '''
    returns RMSD of two vectors x and y
    
    Args:
     x: (np.array)
     y: (np.array) <zeros(len(x))>
    
    Returns:
     (float) root mean squared dev between x and y
    '''
    
    if y is None:
        y = np.zeros(len(x))

    x = np.array(x)
    y = np.array(y)
        
    return np.sqrt(np.mean((x - y) ** 2))


def find_interval(x, iv):

    '''
    returns indices of values in 1-dim x that are >= iv[0] and <= iv[1]
    
    Args:
      x: (np.array) 1-dim
      iv: (list) [on, off] specifying the interval boundaries
    
    Returns:
      xi: (np.array) of indices of elements within range of iv
    '''
    
    xi = np.where((x >= iv[0]) & (x <= iv[1]))[0]
    return xi


def first_interval(x, iv):

    '''
    returns row index of seg containing t (only first in case of multiple)
    
    Args:
     x: (float)
     iv: (2-dim np.array) of intervals [on offset]
    
    Returns:
     ri: (int) row index; -1 means no interval found

    '''
    
    ri = -1
    xi = np.where((iv[:, 0] <= x) & (iv[:, 1] >= x))[0]
    
    if len(xi) > 0:
        ri = xi[0]

    return int(ri)


def seq_windowing(win, rng, align='center'):

    '''
    vectorized version of windowing
    
    Args:
        win: (int) window length in samples
        rng: (list) [on, off] range of indices to be windowed
        align: (str) <center>|left|right alignment
    
    Returns:
      ww: (np.array) [[on off] ...]
    '''

    win = int(win)
    
    if align == "center":
        vecwin = np.vectorize(windowing)
    elif align == "right":
        vecwin = np.vectorize(windowing_rightAligned)
    elif align == "left":
        vecwin = np.vectorize(windowing_leftAligned)

    ii = np.arange(rng[0], rng[1])

    # returns tuple ([on ...], [off, ...])
    ww = vecwin(ii, {"win": win, "rng": rng})

    # [[on off], ...]
    ww = np.array(ww).T
    
    return ww


def windowing(i, o):

    '''
    window of length o["win"] on and offset around single index
    limited by range o["rng"]. vectorized version: seq_windowing()
    
    Args:
      i: (array) indices
      o: (dict)
        win: (int) window length in samples
        rng: (list) [on, off] range of indices to be windowed
    
    Returns:
      on, off (int-s) indices of window around i
    '''

    win = o["win"]
    rng = o["rng"]
    
    # half window
    wl = max([1, math.floor(win / 2)])
    on = max([rng[0], i - wl])
    off = min([i + wl, rng[1]])
    
    # extend window
    d = (2 * wl - 1) - (off - on)
    if d > 0:
        if on > rng[0]:
            on = max([rng[0], on - d])
        elif off < rng[1]:
            off = min([off + d, rng[1]])
            
    return on, off


def windowing_rightAligned(i, o):

    '''
    window around each sample so that it is at the right end (no look-ahead)

    Args:
      i: (np.array) indices
      o: (dict)
        win: (int) window length in samples
        rng: (list) [on, off] range of indices to be windowed
    
    Returns:
      on, off (int-s) indices of window around i

    '''
    
    win = o["win"]
    rng = o["rng"]
    on = max([rng[0], i - win])
    off = min([i, rng[1]])
    
    # extend window (left only)
    d = wl - (off - on)
    if d > 0:
        if on > rng[0]:
            on = max([rng[0], on - d])
            
    # relax 0, 0 case (zero length win)
    if off == on:
        off += 1

    return on, off


def windowing_leftAligned(i, o):

    '''
    window around each sample so that it is at the left end (no looking back)

    Args:
      i: (np.array) indices
      o: (dict)
        win: (int) window length in samples
        rng: (list) [on, off] range of indices to be windowed
    
    Returns:
      on, off (int-s) indices of window around i

    '''

    win = o["win"]
    rng = o["rng"]
    on = max([rng[0], i])
    off = min([i + win, rng[1]])
    
    # extend window (right only)
    d = win - (off - on)
    if d > 0:
        if off < rng[1]:
            off = min([rng[1], off + d])
            
    # relax -1, -1 case (zero length win)
    if on == off:
        on -= 1
        
    return on, off


def windowing_idx(i, s):

    '''
    as windowing(), but returning all indices from onset to offset

    Args:
      i: (int) current index
      s: (dict)
       .win window length
       .rng [on, off] range of indices to be windowed
    
    Returns:
      (np.array) [on:1:off] in window around i

    '''

    on, off = windowing(i, s)
    return np.arange(on, off, 1)



def intersect(a, b):

    '''
    returns intersection list of two 1-dim lists

    Args:
    a, b: lists
    
    Returns:
    (list) sorted intersection of a, b

    '''
    
    return sorted(set(a) & set(b))


def sorted_keys(x):

    '''
    returns sorted list keys
    
    Args:
      x: (dict)
    
    Returns:
      (list) of sorted keys
    '''

    return sorted(x.keys())


def numkeys(x):

    ''' for backward compatibility '''
    
    return sorted_keys(x)


def add_subdict(d, s):

    '''
    add key to empty subdict if not yet part of dict
    
    Args:
      d: (dict)
      s: (key)
    
    Returns:
      d: (dict) incl key pointing to empty subdict
    '''

    if s not in d:
        d[s] = {}
        
    return d


def file_collector(d, e=None):

    '''
    returns files incl full path as list (recursive dir walk)
    
    Args:
      d: (str, dict) directory name; or dict containing fields 'dir' and 'ext'
      e: (str), extension; not used if d is dict
    
    Returns:
      ff: (list) sorted list of full paths to files

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


def outl_idx(y, opt):

    '''
    returns outlier indices
    
    Args:
    y: (np.array) numeric array
    opt: (dict)
         'f' -> factor of min deviation
         'm' -> from 'mean', 'median' or 'fence'
             (mean: m +/- f*sd,
              median: med +/- f*iqr,
              fence: q1-f*iqr, q3+f*iqr)
         'zi' -> true|false - ignore zeros in m and outlier calculation
    
    Returns:
    io: (np.array) outlier indices
    '''

    if opt['zi'] == True:
        i = (y != 0 & np.isfinite(y)).nonzero()
    else:
        i = (np.isfinite(y)).nonzero()

    f = opt['f']

    if np.size(i) == 0:
        return np.array([])
    
    # getting lower and upper boundary lb, ub
    if opt['m'] == 'mean':

        # mean +/- f*sd
        m = np.mean(y.take(i))
        r = np.std(y.take(i))
        lb, ub = m - f * r, m + f * r
    else:
        
        m = np.median(y.take(i))
        q1, q3 = np.percentile(y.take(i), [25, 75])
        q1 = np.round(q1, 8)
        q3 = np.round(q3, 8)
        r = q3 - q1

        if opt['m'] == 'median':

            # median +/- f*iqr
            lb, ub = m - f * r, m + f * r
        else:

            # Tukey's fences: q1-f*iqr , q3+f*iqr
            lb, ub = q1 - f * r, q3 + f * r

    if opt['zi'] == False:
        io = (np.isfinite(y) & ((y > ub) | (y < lb))).nonzero()
    else:
        io = (np.isfinite(y) & ((y > 0) & ((y > ub) | (y < lb)))).nonzero()

    return io


def output_wrapper(v, f, typ, opt={'sep': ',', 'header': True}):

    '''
    output wrapper for typ 'pickle', 'TextGrid', 'csv' and
    'string', 'list' (1-dim)
    
    Args:
      v: (any)
            any for pickle; dict type for json, csv, csv_quote; TextGrid/list/string
      f: (str) file name
      typ: (str) 'pickle'|'TextGrid'|'json'|'list'|'list1line'|'string'|'csv'|'csv_quote'
      opt: (dict) used by some output types, e.g. 'sep', 'header' for csv
    
    File output
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
        
    elif re.search(r'(string|list|list1line)', typ):
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
        if re.search(r'\.csv$', f):
            pd.DataFrame(v).to_csv(f"{f}", na_rep='NA', index_label=False,
                                   index=False, sep=opt['sep'], header=opt['header'])
        else:
            pd.DataFrame(v).to_csv(f"{f}.csv", na_rep='NA', index_label=False,
                                   index=False, sep=opt['sep'], header=opt['header'])
            
    elif typ == 'csv_quote':
        if re.search(r'\.csv$', f):
            pd.DataFrame(d).to_csv(f"{f}", na_rep='NA', index_label=False, index=False,
                                   quoting=csv.QUOTE_NONNUMERIC, sep=opt['sep'],
                                   header=opt['header'])
        else:
            pd.DataFrame(d).to_csv(f"{f}.csv", na_rep='NA', index_label=False, index=False,
                                   quoting=csv.QUOTE_NONNUMERIC, sep=opt['sep'],
                                   header=opt['header'])


def o_tg(tg, fil):

    '''
    TextGrid output into file (content is appended if file exists)
    
    Args:
      tg: (dict) read by i_tg()
      f: (str) file name
    
    File output
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

    for i in sorted_keys(tg['item']):

        # subkey := intervals or points?
        if re.search(tg['item'][i]['class'], 'texttier', re.I):
            subkey = 'points'
        else:
            subkey = 'intervals'

        if tg['format'] == 'long':
            h.write(f"{idt}item [{i}]:\n")

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

        for j in sorted_keys(tg['item'][i][subkey]):
            if tg['format'] == 'long':
                h.write(f"{idt}{idt}{subkey} [{j}]:\n")
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
      (dict) TextGrid element name -> list of field names
    '''

    return {'head': ['xmin', 'xmax', 'size'],
            'item': ['class', 'name', 'xmin', 'xmax', 'size'],
            'points': ['time', 'mark'],
            'intervals': ['xmin', 'xmax', 'text']}


def tgv(v, a):

    '''
    rendering of TextGrid values
    
    Args:
      v: (str, float, int ...)
      a: (str) name of attribute
    
    Returns:
      s: (str, float, int ...) renderedValue
    '''

    if re.search(r'(xmin|xmax|time|size)', a):
        return v
    else:
        return f"\"{v}\""


def tg_tier(tg, tn):

    '''
    returns tier subdict from TextGrid
    
    Args:
      tg: (dict) by i_tg()
      tn: (str) name of tier
    
    Returns:
      t: (dict) tier (deepcopy)
    '''

    if tn not in tg['item_name']:
        return {}
    
    return cp.deepcopy(tg['item'][tg['item_name'][tn]])


def tg_tn(tg):

    '''
    returns list of TextGrid tier names
    
    Args:
      tg: (dict)
    
    Returns:
      tn: (list) sorted list of tiernames
    '''

    return sorted(list(tg['item_name'].keys()))


def tg_tierType(t):

    '''
    returns tier type
    
    Args:
      t: (dict) tg tier by tg_tier()
    
    Returns:
      typ: (str) 'points'|'intervals'|''
    '''

    for x in ['points', 'intervals']:
        if x in t:
            return x
        
    return ''


def tg_txtField(typ):

    '''
    returns text field name according to tier type
    
    Args:
      typ: (str) tier type returned by tg_tierType(myTier)
    
    Returns:
      (str) 'points'|<'text'>
    '''

    if typ == 'points':
        return 'mark'
    return 'text'



def tg_mau2chunk(tg, tn='MAU', cn='CHUNK', cl='c'):

    '''
    creates chunk tier (interpausal units) from MAU tier in TextGrid
    
    Args:
      tg: (dict) textgrid dict
      tn: (str) MAUS tier name
      cn: (str) CHUNK tier name 
      cl: (str) chunk label
    
    Returns:
      c: (dict) chunk tier

    REMARK:
      c can be added to tg by tg_add(tg, c)
    '''

    pau = '<p:>'
    
    # MAU tier
    t = tg_tier(tg, tn)
    
    # CHUNK tier
    k = 'intervals'
    c = {'size': 0, 'name': cn, k: {}}
    for x in ['xmin', 'xmax', 'class']:
        c[x] = t[x]

    # t onset, idx in chunk tier
    to, j = 0, 1
    kk = sorted_keys(t[k])

    for i in kk:
        t1 = np.round(t[k][i]['xmin'], 4)
        t2 = np.round(t[k][i]['xmax'], 4)
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



def tg_mrg(tgf, tiers, opt=None):

    '''
    merge tiers from >=1 TextGrid in tgf to new TextGrid
    
    Args:
      tgf: (string or list) of TextGrid files
      tiers: (string or list) of tier names (contained in one of these files)
      opt: (dict) output TextGrid specs
          'format': <'long'>|'short'
          'name': <'mrg'>
    
    Returns:
      tg_mrg: (dict) merged TextGrid

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
        for i in sorted_keys(d):
            if x not in d[i]['tn']:
                continue
            tier_x = tg_tier(d[i]['tg'], x)
            tg_mrg = tg_add(tg_mrg, tier_x, {'repl': True})
            break

    return tg_mrg


def nan_repl(x, mp=None):

    '''
    replaces NA... values in list according to mapping in map
    
    Args:
      x: (np.array) 1-dim
      mp: (dict) mapping (default NA, NaN, nan -> np.nan)
    
    Returns:
      x: (np.array) NANs, INFs etc replaced
    '''

    if mp is None:
        mp = {'NA': np.nan, 'NaN': np.nan, 'nan': np.nan}

    x = np.array(x)
    for z in mp:
        x[x == z] = mp[z]

    return x



def nan2mean(x, v="median"):

    '''
    replaces NaN and INF by v
    
    Args:
      x: np.array()
      v: value by which to replace, float or string "mean", "median"
    
    Returns:
      y: np.array() with replaced NaN
    '''

    inan = np.where(np.isnan(x))[0]
    iinf = np.where(~np.isfinite(x))[0]
    
    if max(len(inan), len(iinf)) == 0:
        return x

    # replacement value
    if type(v) is not str:
        m = v
    else:
        ifi = np.where(np.isfinite(x))[0]
        if v == "mean":
            m = np.mean(x[ifi])
        else:
            m = np.median(x[ifi])

    if len(inan) > 0:
        x[inan] = m
    if len(iinf) > 0:
        x[iinf] = m

    return x


def input_wrapper(f, typ, opt=None):

    '''
    input wrapper for
    'json', 'pickle', 'TextGrid', 'tab'/'lol', 'par', 'xml', 'l_txt', 'string'
    'lol_txt', 'seg_lab', 'list','csv', 'copa_csv', 'pandas_csv', 'key_val',
    'glove'
    
    Args:
       f: (str) file name
       typ: (str) type of input ('json', etc, see above)
       opt: (dict) additional options
         'sep': <None>, resp ',' separator
         'to_str': <False> (transform all entries in pandas dataframe (returned as dict) to strings)
    
    Returns:
       (any): containedVariable

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

       xml is restricted to copa_xml format
       *csv: recommended: 'pandas_csv', since 'csv' treats all values as strings
       and 'copa_csv' is more explicit than 'pandas_csv' in type definitions but
       might get deprecated.
       Current diffs between copa_csv and pandas_csv: idx columns as ci, fi etc
       and  numeric speakerIds will be treated as integer in pandas_csv but as
       strings in cop_csv.
    '''

    if opt is None:
        opt = {}
    
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
        return i_lol(f, sep=None, frm='1d')

    # 2 dim list
    if typ == 'lol_txt':
        return i_lol(f, sep=sep)

    # segments and labels
    if typ == 'seg_lab':
        return i_seg_lab(f)

    # keys and values
    if typ == 'key_val':
        return i_keyVal(f, opt["sep"])

    # json
    if typ == "json":
        with open(f, m) as h:
            return json.load(h)

    # pickle
    if typ == "pickle":
        with open(f, m) as h:
            return pickle.load(h)

    sys.exit(f"file type {typ} cannot be processed.")
        

def is_pau(s, lab=''):

    '''
    returns True if string s is empty or equal pause label lab
    
    Args:
     s: (str)
     lab: (str) pause label
    
    Returns:
     (boolean)
    '''

    if re.search(r'^\s*$', s) or s == lab:
        return True
    
    return False


def copa_categ_var(x):

    '''
    decides whether name belongs to categorical variable
    
    Args:
    x: (str)
    
    Returns:
    (boolean)
    '''

    if ((x in lists('factors', 'set')) or
        re.search(r'^(grp|lab|class|spk|tier)', x) or
        re.search(r'_(grp|lab|class|tier)$', x) or
        re.search(r'_(grp|lab|class|tier)_', x) or
            re.search(r'is_(init|fin)', x)):
        return True
    
    return False


def copa_reference_var(x):

    '''
    decides whether name belongs to reference variable (time, base value, etc)

    Args:
    x: (str)

    Returns:
    (boolean)
    '''

    if re.search(r"(t_on|t_off|bv|dur)", x):
        return True
    
    return False


def copa_opt_dynad(task, opt):

    '''
    Dynamically adjusts 'fsys' part on copa options based on input.
    Requires uniform config format as in wrapper_py/config/ids|hgc.json
    
    Args:
      task: (str) subfield in opt[fsys][config] pointing to copa opt json file
      opt: (dict) in which to find this subfield
    
    Returns:
      copa_opt: (dict)copasul options with adjusted fsys specs
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
       opt: (dict) copa options
    
    Returns:
       df: (dict) copa feature dict
    '''

    f = os.path.join(opt['fsys']['export']['dir'],
                     f"{opt['fsys']['export']['stm']}.summary.csv")
    
    return input_wrapper(f, "copa_csv")


def copa_unifChannel(d):

    '''
    replaces multiple channel rhy columns to single one that
    contains values of respective channel idx+1
    example row i:
      ci=0
      SYL_1_rate = 4
      SYL_2_rate = 0
     -> SYL_rate = 4
    only works if channel-tier relations are expressed by index=ci+1
    
    Args:
     d: (dict) features
    
    Returns:
     y: (dict) features with unified columns
    '''

    y, z = {}, {}
    for x in d:
        if re.search(r'[12]_', x):
            # key and subkey in z
            k = re.sub(r'_*[12]_', '_', x)
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


def i_keyVal(f, sep=None):

    '''
    returns dict z: z["a"]="b" from rows a;b
    rows with length <2 are skipped

    Args:
    f: (str) input table file name
    sep: (str) column separator

    Returns:
    z: (dict)
    '''

    z = dict()
    with open(f, encoding='utf-8') as h:
        for u in h:
            if sep is None:
                x = u.split()
            else:
                u = str_standard(u, True)
                x = u.split(sep)
            if len(x) < 2:
                continue
            z[x[0]] = x[1]
        h.close()
        
    return z


def i_seg_lab(f, sep=None):

    '''
    [on off label] rows converted to 2-dim np.array and label list
    
    Args:
      f: (str) file name
      sep: (str) column separator,
          regex need to be marked by 'r:sep'
          e.g.: 'r:\\t+' - split at tabs (double backslash needed!)
    
    Returns:
      d: (dict)
       .t (np.array) [[on off]...]
       .lab (list) [label ...]
    '''


    if type(sep) is str and re.search(r'^r:', sep):
        is_re = True
        sep = re.sub(r'^r:', '', sep)
    else:
        is_re = False
    d = {'t': [], 'lab': []}
    js = ' '
    
    with open(f, encoding='utf-8') as h:
        for z in h:

            # default or customized split
            if sep is None:
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
            d['t'].append(list(x[0:2]))
            
            if len(x) >= 3:
                d['lab'].append(js.join(x[2:len(x)]))
            else:
                d['lab'].append('')
        h.close()

    d['t'] = np.array(d['t'])
        
    return d


def counter(x):

    '''
    counting events from input list
    
    Args:
      x: (list)
    
    Returns:
      c: (dict)
       myType -> myCount
      n: (int) number of tokens in x
    '''

    return Counter(x), len(x)


def count2prob(c, n):

    '''
    converts counts to MLE probs
    
    Args:
       c: (Counter dict) from counter()
       n: (int) number of tokens from counter()
    
    Returns:
       p: (dict)
        myType -> myProb
    '''

    p = {}
    for x in c:
        p[x] = c[x] / n
        
    return p


def prob2entrop(p):

    '''
    probs to entropy from prob DICT
    
    Args:
       p: (dict) myType -> myProb from count2prob()
    
    Returns:
       h: (float) unigram entropy
    '''

    h = 0
    for x in p:
        if p[x] == 0:
            continue
        h -= (p[x] * binlog(p[x]))
        
    return h


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
      x: (list or set) of strings
      pm: (dict) unigram language model
    
    Returns:
      p: (dict)
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
      x: (str)
      pm: (dict) unigram probmod
    
    Returns:
      (float) probability of x
    '''

    if x in pm:
        return pm[x]
    
    return 0


def list2entrop(x):

    '''
    wrapper around counter() and count2prob(), prob2entrop()
    
    Args:
      x: (list) of strings
    
    Returns:
      h: (float) unigram entropy
    '''

    c, n = counter(x)
    p = count2prob(c, n)
    h = prob2entrop(p)

    return h


def uniq(x):

    '''
    returns sorted + unique element list
    
    Args:
      x: (list)
    
    Returns:
      x: (list) unique
    '''
    
    return sorted(list(set(x)))


def cellwise(f, x):

    '''
    apply scalar input functions to 1d or 2d arrays
    
    Args:
      f: (function handle)
      x: (np.array)

    Return:
      x: (np.array) processed by f
    '''

    if len(x) == 0:
        return x
    
    vf = np.vectorize(f)

    # 1d input
    if np.asarray(x).ndim == 1:
        return vf(x)

    # 2d input
    return np.apply_along_axis(vf, 1, x)


def lol(f, opt=None):

    '''
    converts input to 2-dim array
    
    Args:
      f: (str) fileName or list
      opt: (dict)
          'colvec' (boolean) enforce to return column vector
                (for input files with 1 row or 1 column
                 FALSE returns [[...]], and TRUE returns [[.][.]...])
    
    Returns:
      x: (np.array) 2-dim
    '''
    
    opt = opt_default(opt, {'colvec': False})

    if (type(f) is str and (not is_file(f))):
        sys.exit(f, ": file does not exist.")

    if type(f) is str:
        x = np.loadtxt(f)
    else:
        x = np.array(f)

    # 2-dim
    if x.ndim == 1:
        x = x.reshape((-1, x.size))

    # transpose to column vector
    if opt['colvec'] and len(x) == 1:
        x = np.transpose(x)

    return x


def ncol(x):

    '''
    returns number of columns
    1 if array is one dimensional
    
    Args:
      x: (np.array)
    
    Returns:
      (int) number of columns
    '''

    if np.ndim(x) == 1:
        return 1
    
    return len(x[0, :])


def df(f, col):

    '''
    outputs a data frame assigning each column a title
    
    Args:
       f: (str) file name
       col: (list) column names
    
    Returns:
       df: (pd.DataFrame) with content of f
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
      x: (str) full path
    
    Returns:
      d: (str) directory
      s: (str) file stem
      e: (str) extension
    '''

    dd = os.path.split(x)
    d = dd[0]
    s = os.path.splitext(os.path.basename(dd[1]))
    e = s[1]
    e = re.sub(r'\.', '', e)
    
    return d, s[0], e


def stm(f):

    '''
    returns file name stem
    
    Args:
      f: (str) full path
    
    Returns:
      s: (str) stem
    '''

    s = os.path.splitext(os.path.basename(f))[0]
    return s


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
      x: (np.array) normalized
    '''

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
    normalizes scalar to range; opt.min|max set to opt.rng
    supports minmax only

    Args:
    x: (np.array)
    opt: (dict) with specs mtd, min, max, rng
    '''

    if opt['mtd'] == 'minmax':
        mi = opt['min']
        ma = opt['max']
        r = opt['rng']
        
        if ma > mi:
            # nrm [0, 1]
            x = (x - mi)/(ma - mi)

            # nrm [rng[0], rng[1]]
            x = r[0] + x * (r[1] - r[0])
            
    return x


def wav_int2float(s):

    '''
    maps integers from -32768 to 32767 to interval [-1 1]
    
    Args:
    s: (np.array) signal read by scipy.io.wavfile.read()

    Returns:
    s: (np.array) normalized to be within range -1, 1
    '''

    s = s / 32768
    s[s < -1] = -1
    s[s > 1] = 1
    
    return s


def nrm_zero_set(t, t0, rng):

    '''
    normalisation of T to range specified in vector RANGE
    
    Args:
    t: (np.array) of values to be time-normalized
    t0: (float) value in t which will become 0
    rng: (list) min and max value of normalized values (interval must include 0)

    Returns:
    tn: (np.array) normalized values

    '''
    
    if len(t) == 0:
        return t
    
    # t halves
    t1 = t[t <= t0]
    t2 = t[t > t0]

    if len(t1) == 0 or len(t2) == 0:
        return nrm_vec(t, {'mtd': 'minmax', 'rng': rng})

    # corresponding ranges
    r1 = [rng[0], 0]
    r2 = [rng[1] / len(t2), rng[1]]

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

    Args:
    l: (int) length
    sts: (int) stepsite

    Returns:
    (np.arange)
    '''
    
    return np.arange(0, l, sts)


def idx_seg(on, off, sts=1):

    '''
    returns index array between on and off (both included)

    Args:
    l: (int) length
    sts: (int) stepsite

    Returns:
    (np.arange)
    '''

    return np.arange(on, off+1, sts)



def idx(l):

    '''
    returns index iterable of list L
    '''
    
    return range(len(l))



def ndim(x):

    '''
    returns dimension of numpy array
    
    Args:
      x: (np.array)
    
    Returns:
      (int) for number of dims
    '''

    return len(x.shape)


def tg_tier2tab(t, opt=None):

    '''
    transforms TextGrid tier to 2 arrays
    point -> 1 dim + lab
    interval -> 2 dim (one row per segment) + lab
    
    Args:
      t: (dict) tg tier (by tg_tier())
      opt: (dict)
          .skip <""> regular expression for labels of items to be skipped
                if empty, only empty items will be skipped
    
    Returns:
      x: (np.array) 1- or 2-dim array of time stamps
      lab: (list) corresponding labels

    REMARK:
      empty intervals are skipped
    '''

    opt = opt_default(opt, {"skip": ""})
    if len(opt["skip"]) > 0:
        do_skip = True
    else:
        do_skip = False

    x, lab = [], []
    
    if 'intervals' in t:
        for i in sorted_keys(t['intervals']):
            z = t['intervals'][i]
            if len(z['text']) == 0:
                continue
            if do_skip and re.search(opt["skip"], z["text"]):
                continue

            x.append([z['xmin'], z['xmax']])
            lab.append(z['text'])
    else:
        for i in sorted_keys(t['points']):
            z = t['points'][i]
            if do_skip and re.search(opt["skip"], z["mark"]):
                continue
            x.append(z['time'])
            lab.append(z['mark'])

    x = np.array(x)
            
    return x, lab


def tg_tab2tier(t, lab, specs):

    '''
    transforms table to TextGrid tier
    
    Args:
       t: (np.array) 1- or 2-dim array with time info
       lab: (list) of labels
       specs: (dict)
            ['class'] <'IntervalTier' for 2-dim, 'TextTier' for 1-dim>
            ['name']
            ['xmin'] <0>
            ['xmax'] <max tab>
            ['size'] - will be determined automatically
            ['lab_pau'] - <''>
    
    Returns:
       tt: (dict) tg tier (see i_tg() subdict below myItemIdx)
    
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
    
    if nd == 1:
        # point tier content
        for i in idx_a(len(t)):

            # point tier content might be read as [[x],[x],[x],...] or [x,x,x,...]
            if of_list_type(t[i]):
                z = t[i, 0]
            else:
                z = t[i]
            tt['points'][i + 1] = {'time': z, 'mark': lab[i]}
        tt['size'] = len(t)
    else:
        # interval tier content
        j = 1
        
        # initial pause
        if t[0, 0] > tt['xmin']:
            tt['intervals'][j] = {'xmin': tt['xmin'],
                                  'xmax': t[0, 0], 'text': lp}
            j += 1
        for i in idx_a(len(t)):

            # pause insertions
            if ((j - 1 in tt['intervals']) and
                    t[i, 0] > tt['intervals'][j - 1]['xmax']):
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
            j += 1  # uniform 1 subtraction for size

        # size
        tt['size'] = j - 1
        
    return tt



def tg_add(tg, tier, opt={'repl': True}):

    '''
    add tier to TextGrid
    
    Args:
      tg: (dict) from i_tg(); can be empty dict
      tier: (dict) subdict to be added:
          same dict form as in i_tg() output, below 'myItemIdx'
      opt: (dict)
         ['repl'] - replace tier of same name
    
    Returns:
      tg: (dict) updated

    REMARK:
      if generated from scratch head xmin and xmax are taken over from
      the tier - which might need to be corrected afterwards!
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
        ii = sorted_keys(tg['item'])
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
       an: (dict) annotation e.g. by i_par() or i_copa_xml()
    
    Returns:
       tg: (dict) TextGrid

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
            for j in sorted_keys(an[x]['items']):
                y = an[x]['items'][j]
                tt[j + 1] = {'time': y['t'],
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
            for i in sorted_keys(an[x]['items']):
                y = an[x]['items'][i]

                # pause insertions
                if ((j - 1 in tt) and
                    y['t_start'] > tt[j-1]['xmax']):
                    tt[j] = {'xmin': tt[j-1]['xmax'],
                             'xmax': y['t_start'], 'text': ''}
                    j += 1
                tt[j] = {'xmin': y['t_start'],
                         'xmax': y['t_end'], 'text': y['label']}
                tg['item'][ii]['xmax'] = tt[j]['xmax']
                j += 1

            # size
            tg['item'][ii]['size'] = j - 1

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
            j = max(tg['item'][ii]['intervals']) + 1
            xm = tg['item'][ii]['intervals'][j - 1]['xmax']
            tg['item'][ii]['intervals'][j] = {'text': '', 'xmin': xm,
                                              'xmax': tg['head']['xmax']}
        tg['item'][ii]['xmax'] = tg['head']['xmax']
        
    return tg


def par2tg(par_in):

    '''
    as inter2tg() but omitting header item
    
    Args:
      par: (dict) from i_par()
    
    Returns:
      tg: (dict) as with i_tg()
    '''

    par = cp.deepcopy(par_in)
    del par['header']
    
    return inter2tg(par)


def tg_item_keys(t):

    '''
    returns item-related subkeys: 'intervals', 'text' or 'points', 'mark'
    
    Args:
      t: (dict) tier
    
    Returns:
      (str) key1
      (str) key2
    '''

    if 'intervals' in t:
        return 'intervals', 'text'
    
    return 'points', 'mark'


def tg2par(tg, fs):

    '''
    wrapper around tg2inter() + adding 'header' item / 'class' key
    
    Args:
     tg: (dict) read by tg_in
     fs: (int) sample rate
    
    Returns:
     par: (dict) of partitur (copa-xml format)

    REMARK:
       information loss! Output cannot contain word index references!
       Thus MAU is BPF class 2 in this case, without 'i' field.
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
     par: (dict) BAS partitur content
     tn: (str) tier name
    
    Returns:
     c: (int) tier class
    '''

    t = par[tn]['type']
    if t == 'null':
        return 1
    
    n = sorted_keys(par[tn]['items'])
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
      fs: (int) sample rate
    
    Returns:
      h: (dict) header
    '''

    return {'type': 'header', 'class': 0, 'items': {'SAM': fs}}


def tg2inter(tg, opt=None):

    '''
    transforms textgrid dict to interchange format
     same as i_par(), i_copa_xml() output
    
    Args:
      tg: (dict) from i_tg
      opt: (dict)
        snap: (boolean) if True, also empty-label intervals are kept
    
    Returns:
      an (dict):
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
        else:
            # empty tier
            continue
        
        for i in sorted_keys(t[k]):
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
      ff: (str) file name
    
    Returns:
      tg: (dict)
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
     ff: (str) file name
    Out:
     tg: (dict) see i_tg()
    '''

    tg = {'name': ff, 'format': 'short', 'head': {},
          'item_name': {}, 'item': {}, 'type': 'TextGrid'}
    (key, fld, skip, state, nf) = ('head', 'xmin', True, 'head', tg_nf())
    idx = {'item': 0, 'points': 0, 'intervals': 0}
    
    with open(ff, encoding='utf-8') as f:
        for z in f:
            z = re.sub(r'\s*\n$', '', z)
            if re.search(r'object\s*class', z, re.I):
                fld = nf[state]['#']
                skip = False
                continue
            else:
                if ((skip == True) or re.search(r'^\s*$', z) or
                        re.search('<exists>', z)):
                    continue
            if re.search(r'(interval|text)tier', z, re.I):
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
                elif re.search(r'(points|intervals)', state):
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
     ff: (str) file name
    
    Returns:
     tg: (dict) see i_tg()
    '''

    tg = {'name': ff, 'format': 'long', 'head': {},
          'item_name': {}, 'item': {}}
    (key, skip) = ('head', True)
    idx = {'item': 0, 'points': 0, 'intervals': 0}
    
    with open(ff, encoding='utf-8') as f:
        for z in f:
            z = re.sub(r'\s*\n$', '', z)
            if re.search(r'object\s*class', z, re.I):
                skip = False
                continue
            else:
                if ((skip == True) or re.search(r'^\s*$', z) or
                        re.search('<exists>', z)):
                    continue
            if re.search(r'item\s*\[\s*\]:?', z, re.I):
                key = 'item'
            elif re.search(r'(item|points|intervals)\s*\[(\d+)\]\s*:?', z, re.I):
                m = re.search(
                    r'(?P<typ>(item|points|intervals))\s*\[(?P<idx>\d+)\]\s*:?', z)
                i_type = m.group('typ').lower()
                idx[i_type] = int(m.group('idx'))
                if i_type == 'item':
                    idx['points'] = 0
                    idx['intervals'] = 0
            elif re.search(r'([^\s+]+)\s*=\s*\"?(.*)', z):
                m = re.search(r'(?P<fld>[^\s+]+)\s*=\s*\"?(?P<val>.*)', z)
                (fld, val) = (m.group('fld').lower(), m.group('val'))
                fld = re.sub('number', 'time', fld)
                val = re.sub(r'[\"\s]+$', '', val)
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
    next TextGrid item dict

    Returns:
    (dict) mapping each item to the one following in TextGrid

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
      ff: (str) TextGrid file name
    
    Returns:
      s (str) 'short'|'long'
    '''

    with open(ff, encoding='utf-8') as f:
        for z in f:
            if re.search(r'^\s*<exists>', z):
                f.close
                return 'short'
            elif re.search(r'xmin\s*=', z):
                f.close
                return 'long'
    return 'long'


def opt_default(c, d):

    '''
    recursively adds default fields of dict d to dict c
       if not yet specified in c
    
    Args:
     c: (dict) some dict
     d: (dict) dict with default values
    
    Returns:
     c: (dict) merged dict (defaults added to c)
    '''

    if c is None:
        c = {}
    
    for x in d:
        if x not in c:
            c[x] = d[x]
        elif type(c[x]) is dict:
            c[x] = opt_default(c[x], d[x])
            
    return c


def halx(x, l):

    '''
    length adjustment of list x to length l
    
    Args:
      x: (list or np.array)
      l: (int) required length
    
    Returns:
      x (list) adjusted

     if x is longer than l, x[0: l] is returned
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
    length adjustment of 2 lists x, y to shorter list
    
    Args:
      x: (list)
      y: (list)
    
    Returns:
      x,y shortened to same length if needed
    '''

    if len(x) > len(y):
        x = x[0:len(y)]
    elif len(y) > len(x):
        y = y[0:len(x)]
        
    return x, y





def check_var(c):

    '''
    diverse variable checks and respective reactions
    
    Args:
     c: (dict)
      ['var'] - any variable or variable container
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
                sys.exit(f"ERROR! Clustering of {dom} contours requires stylization step.\n" \
                         f"Set navigate.do_styl_{dom} to true.")

        # styl - dom = gnl|glob|loc|bnd|rhy...
        elif s['step'] == 'styl':
            dom = s['dom']

            # copa['data'][ii][i] available?
            # -> first file+channelIdx (since in file/channel-wise augmentation steps does not start from
            #       copa is initialized for single varying ii and i)
            ii, i, err = check_var_numkeys(copa)
            if err:
                sys.exit(f"ERROR! {dom} feature extraction requires preprocessing step.\n" \
                         "Set navigate.do_preproc to true.")

            # preproc fields available?
            if check_var_copa_preproc(copa, ii, i):
                sys.exit(f"ERROR! {dom} feature extraction requires preprocessing step.\n" \
                         "Set navigate.do_preproc to true.")

            # domain field initialization required? (default True)
            if (('req' not in s) or (s['req'] == True)):
                if check_var_copa_dom(copa, dom, ii, i):
                    sys.exit(f"ERROR! {dom} feature extraction requires preprocessing step.\n" \
                             "Set navigate.do_preproc to true.")

            # dependencies on other subdicts
            if 'dep' in s:
                for x in s['dep']:
                    if check_var_copa_dom(copa, x, ii, i):
                        sys.exit(f"ERROR! {dom} feature extraction requires {x} features.\n" \
                                 f"Set navigate.do_{x} to true.")

            # ideosyncrasies
            if re.search(r'^rhy', dom):
                if ('rate' not in copa['data'][ii][i]):
                    sys.exit(f"ERROR! {dom} feature extraction requires an update of the preprocessing step.\n" \
                             "Set navigate.do_preproc to true")
            if dom == 'bnd':
                if ((copa['config']['styl']['bnd']['residual']) and
                        ('r' not in copa['data'][0][0]['f0'])):
                    sys.exit(f"ERROR! {dom} feature extraction based on f0 residuals requires a previous " \
                             "global contour stylization so that the register can be subtracted from the " \
                             "f0 contour.\nSet navigate.do_styl_glob to true, or set styl.bnd.residual to false")


def check_var_copa_preproc(copa, ii, i):

    '''
    check blocks called by check_var()
    preproc fields given?
    returns True in case of violation

    Args:
    copa: (dict)
    ii: (int) file index
    i: (int) channel index

    Returns:
    (boolean)

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
      copa: (dict)
    
    Returns:
      i: (int) lowest numeric key in copa['data']
      j: (int) lowest numeric key in copa['data'][i]
      err: (boolean) True if there is no copa['data'][i][j], else False
    '''

    if type(copa['data']) is not dict:
        return -1, -1, True
    
    a = sorted_keys(copa['data'])
    if len(a) == 0 or (type(copa['data'][a[0]]) is not dict):
        return -1, -1, True

    b = sorted_keys(copa['data'][a[0]])
    if len(b) == 0 or (type(copa['data'][a[0]][b[0]]) is not dict):
        return -1, -1, True

    return a[0], b[0], False



def check_var_copa_dom(copa, dom, ii, i):

    '''
    check whether domain subdict is initialized

    Args:
    copa: (dict)
    dom: (str) domain
    ii: (int) file index
    i: (int) channel index

    Returns:
    (boolean) True if initialized
    
    '''

    if dom not in copa['data'][ii][i]:
        return True
    
    return False


def sig_preproc(y, opt=None):

    '''
    basic signal preprocessing function
    
    Args:
      y: (np.array) signal vector
      opt: (dict)
         ['rm_dc']: centralize
    '''

    dflt = {'rm_dc': True}
    opt = opt_default(opt, dflt)

    # remove DC
    if opt['rm_dc'] == True:
        y = y - np.mean(y)

    return y


def fsys_stm(opt, s):

    '''
    concatenates directory and stem of fsys-subdirectory
    
    Args:
      opt: (dict) with subdict ['fsys'][s][dir|stm]
      s: (str) subdict name (e.g. 'out', 'export' etc)

    Returns:
      fo: (str) full path name
    '''

    fo = os.path.join(opt['fsys'][s]['dir'],
                      opt['fsys'][s]['stm'])
    
    return fo


def copa_yseg(copa, dom, ii, i, j, t=[], y=[]):

    '''
    returns f0 segment from vector according to copa time specs
    
    Args:
      copa: (dict)
      dom: (str) 'glob'|'loc'
      ii: (int) file index
      i: (int) channel index
      j: (int) segment index
      t: (np.array) time vector of channel i
      y: (np.array) f0 vector of channel i
    
    Returns:
      ys: (np.array) f0-segment in file ii, channel ii, segment j
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
      an: (dict) generated by i_copa_xml()
      f: (str) file name
    
    outputs xml to f
    '''

    # identation
    i1 = "  "
    i2 = f"{i1}{i1}"
    i3 = f"{i2}{i1}"
    i4 = f"{i2}{i2}"
    i5 = f"{i3}{i2}"
    
    # subfields for tier type
    fld = {'event': ['label', 't'], 'segment': ['label', 't_start', 't_end']}

    # output
    h = open(f, mode='w', encoding='utf-8')
    h.write(f"{xml_hd()}\n<annotation>\n{i1}<tiers>\n")

    # over tiers
    for x in sorted(an.keys()):
        h.write(f"{i2}<tier>\n")
        h.write(f"{i3}<name>{x}</name>\n{i3}<items>\n")
        
        # over items
        for i in sorted_keys(an[x]['items']):
            h.write("{}<item>\n".format(i4))
            for y in fld[an[x]['type']]:
                z = an[x]['items'][i][y]
                if y == 'label':
                    z = xs.escape(z)
                h.write(f"{i5}<{y}>{z}</{y}>\n")
            h.write(f"{i4}</item>\n")
        h.write(f"{i3}</items>\n{i2}</tier>\n")
    h.write(f"{i1}</tiers>\n</annotation>\n")
    h.close()

    return


def xml_hd():

    '''
    returns xml header

    Returns:
    (str) header

    '''

    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"


def i_list(f):

    '''
    returns file content as 1-dim list of strings (one element per row)
    
    Args:
      f: (str) file name
    
    Returns:
      l: (list)
    '''

    with open(f, encoding='utf-8') as h:
        l = [x.strip() for x in h]
        h.close()
    return l


def i_lol(f, sep=None, frm='2d'):

    '''
    reads any text file as 1-dim or 2-dim list (of strings)
        
    Args:
      s: (str) file name
      sep: (str) separator
          regex need to be marked by 'r:sep'
          e.g.: 'r:\\t+' - split at tabs (double backslash needed!)
      frm: (str) '1d'|'2d'
    
    Returns:
      d: (list) 2-dim x (sublists: rows splitted at whitespace)
         or 1-dim (rows concat to 1 list)
    '''

    if type(sep) is str and re.search(r'^r:', sep):
        is_re = True
        sep = re.sub(r'^r:', '', sep)
    else:
        is_re = False
        
    d = []
    with open(f, encoding='utf-8') as h:
        for z in h:
            # default or customized split
            if sep is None:
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
      x: (str) some string
      a: (boolean) if True, also replace all whitespace(-sequences)
         by single blank
    
    Returns:
      x: (str) standardized string
    '''

    x = re.sub(r'^\s+', '', x)
    x = re.sub(r'\s+$', '', x)
    
    if a:
        x = re.sub(r'\s+', ' ', x)

    return x


def o_par(par, f):

    '''
    outputs BAS partiture file
    
    Args:
      par: (dict) in i_copa_xml format (read by i_par)
      f: (str) output file name
    
    Partitur File output
      Tier class examples:
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
         [f"SAM: {fs}"],
         ["LBD:"]]

    for tn in sorted_keys(par):
        if tn == 'header':
            continue
        tier = cp.deepcopy(par[tn])
        c = tier['class']
        
        # add to par output list
        for i in sorted_keys(tier['items']):
            a = o_par_add(tier, tn, i, c, fs)

            # unsnap for segments
            o = o_par_unsnap(o, a, c)
            o.append(a)

    # -> 1dim list
    for i in idx(o):
        o[i] = ' '.join(map(str, o[i]))

    # output
    output_wrapper(o, f, 'list')
    
    return



def o_par_unsnap(o, a, c):

    '''
    for segment tiers reduces last time offset in o by 1 sample
    if equal to onset in a
    
    Args:
      o: (list) partitur output so far
      a: (list) new list to add
      c: (int) tier class
    
    Returns:
      o: (list) updated
    '''

    # no MAU segment tier
    if c != 4 or a[0] != 'MAU:':
        return o
    
    # diff tiers
    if o[-1][0] != a[0]:
        return o
    t = o[-1][1]
    dur = o[-1][2]
    if t + dur != a[1] - 1:
        o[-1][2] = a[1] - 1 - t
        
    return o


def o_par_add(tier, tn, i, c, fs):

    '''
    adds row to par output file
    
    Args:
      tier: (dict) tier subdict from par
      tn: (str) current tier name
      i: (int) current item index in tier
      c: (int) tier class 1-5
      fs: (int) sample rate

    Returns:
      a: (list) row to be added to partitur output

    '''

    x = tier['items'][i]
    s = f"{tn}:"
    
    if c == 1:
        # no time reference
        a = [s, x['i'], x['label']]
    elif c == 2 or c == 4:
        # segment
        t = int(x['t_start'] * fs)
        dur = int((x['t_end'] - x['t_start']) * fs)
        if c == 2:
            a = [s, t, dur, x['label']]
        else:
            a = [s, t, dur, x['i'], x['label']]
    elif c == 3 or c == 5:
        # event
        t = int(x['t'] * fs)
        if c == 3:
            a = [s, t, x['label']]
        else:
            a = [s, t, x['i'], x['label']]
            
    return a


def i_par(f, opt=None):

    '''
    read BAS partiture file
    
    Args:
      s: (str) file name
      opt: (dict)
          snap: <False>
            if True off- and onset of subsequent intervals in MAU are set equal
            (of use for praat-TextGrid conversion so that no empty segments
             are outputted)
          header: <True>
          fs: sample rate, needed if no header provided, i.e. header=False
    
    Returns: 
    par: (dict)
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
    
    output is same as i_copa_xml() + 'class' key for tier class and
    'i' for word idx

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
        elif headOff == False or (not re.search(r':$', z[0])):
            continue
        tn = re.sub(r':$', '', z[0])
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
                par[tn]['items'][ii[tn]]['t_start'] = \
                            par[tn]['items'][ii[tn] - 1]['t_end']
            else:
                par[tn]['items'][ii[tn]]['t_start'] = int(z[1]) / fs
            par[tn]['items'][ii[tn]]['t_end'] = \
                            (int(z[1]) + int(z[2]) - 1) / fs
        elif tc[tn] == 3:
            par[tn]['items'][ii[tn]]['label'] = js.join(z[2:len(z)])
            par[tn]['items'][ii[tn]]['t'] = int(z[1]) / fs
        elif tc[tn] == 4:
            par[tn]['items'][ii[tn]]['label'] = js.join(z[4:len(z)])
            if opt['snap'] and ii[tn] > 0:
                par[tn]['items'][ii[tn]]['t_start'] = \
                            par[tn]['items'][ii[tn] - 1]['t_end']
            else:
                par[tn]['items'][ii[tn]]['t_start'] = int(z[1]) / fs
            par[tn]['items'][ii[tn]]['t_end'] = \
                            (int(z[1]) + int(z[2]) - 1) / fs
            par[tn]['items'][ii[tn]]['i'] = z[3]
        elif tc[tn] == 5:
            par[tn]['items'][ii[tn]]['label'] = js.join(z[3:len(z)])
            par[tn]['items'][ii[tn]]['t'] = int(z[1]) / fs
            par[tn]['items'][ii[tn]]['i'] = z[2]
        ii[tn] += 1

    # add header subdict
    par['header'] = par_header(fs)

    return par


def tab2copa_xml_tier(x, lab=None):

    '''
    transforms 1- or 2dim table to [myTierName] subdict
    
    Args:
      x: (np.array) 1- or 2-dim array with time info
      lab: (list) corresp label; if empty uniform label 'x' is assigned
    
    Returns:
      y dict
         ['type']: event|segment
         ['items'][itemIdx]['label']
                           ['t'] or ['t_start']
                                    ['t_end']
    call:
      myAnnot[myTierName] = tab2copa_xml_tier(myTab, myLab)
    '''

    if ncol(x) == 2:
        typ = 'segment'
    else:
        typ = 'event'
    tier = {'type': typ, 'items': {}}

    if lab is None:
        lab = []

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
    read copa xml of type:
    <annotation><tiers><tier><name>
                             <items><item><label>
                                          <t_start>
                                          <t_end>
                                          <t>
    
    Args:
      f: (str) input file name
    
    Returns:
    annot: (dict) tier
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
                'label': extract_xml_element(item, 'label', 'text')
            }
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
      myTree: (xml.etree.ElementTree (sub-)object)
      elementName: (str)
      ret: (str) 'element'|'string'
    
    Returns:
      (object or string)
    '''

    for e in myTree:
        if e.tag == elementName:
            if ret == 'element':
                return e
            else:
                return e.text
            
    return None


def myLog(o, task, msg=''):

    '''
    log file opening/closing/writing
    
    Args:
      o: (dict) that needs to contain log file name in o['fsys']['log']
      task: (str) 'open'|'write'|'print'|'close'
      msg: (str) message <''>
    
    Returns:
      for task 'open': o (dict) with new field o['log']
                       containing file handle
      else: (boolean) True
    '''

    if task == 'open':
        h = open(o['fsys']['log'], 'a')
        h.write(f"\n## {isotime()}\n")
        o['log'] = h
        return o
    elif task == 'write':
        o['log'].write(f"{msg}\n")       
    elif task == 'print':
        o['log'].write(f"{msg}\n")
        print(msg)
    else:
        o['log'].close()
        
    return True


def cntr_classif(c, d, opt=None):

    '''
    nearest centroid classification
    
    Args:
      c: (dict)
         [myClassKey] -> myCentroidVector
      d: (np.array) feature matrix (ncol == ncol(c[myClassKey])
      opt: (dict)
         ['wgt']  -> feature weights (ncol == ncol(c[myClassKey]))
         ['attract'] -> class attractiveness (> 0)
                     [myClassKey] -> attractiveness <1>.
                     Distance d to class is multiplied derived from
                     d = 1 - attract / sum(attract)
    
    Returns:
      a: (list) with class IDs
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
            cdist[x] = 1 - opt['attract'][x] / attract_sum
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
    returns weighted mean
    
    Args:
      x: (np.array) vector
      w: (np.array) weight vector of same length
    
    Returns:
      y: (np.array) weighted mean of x
    '''
    
    return np.sum(w * x) / np.sum(w)



def feature_weights_unsup(x, mtd='corr'):

    '''
    estimates feature weights for unsupervised learning
    
    Args:
      x: (np.array) m x n feature matrix, columns-variables, rows-observations
      mtd: (str) currently only 'corr' supported
    
    Returns:
      w: (np.array) 1 x n weight vector
    
    Method 'corr':
      - for 'informed clustering' with uniform feature trend, e.g. all
         variables are expected to have high values for cluster A and
         low values for cluster B
      - wgt(feat):
           if corr(feat,rowMeans) <= 0: 0
           if corr(feat,rowMeans) > 0: its corr
      wgts are normalized to w / sum(w)
    '''

    # row means
    m = np.mean(x, axis=1)

    # correlations
    w = []
    for i in range(ncol(x)):
        r = np.corrcoef(x[:, i], m)
        r = np.nan_to_num(r)
        w.append(max(r[0, 1], 0))

    w = np.array(w)
        
    # unusable: uniform weight
    if np.sum(w) == 0:
        return np.ones(len(m))

    # normalize
    return w / np.sum(w)



def dist_eucl(x, y, w=[]):

    '''
    weighted euclidean distance between x and y
    
    Args:
      x: (np.array)
      y: (np.array)
      w: weight array <[]>
    
    Returns:
      d: (float) Euclidean distance
    '''
    
    if len(w) == 0:
        w = np.ones(len(x))
        
    q = x - y

    return np.sqrt((w * q * q).sum())


def calc_delta(x):

    '''
    calculates delta values x[i, :] - x[i-1, :] for values in x
    
    Args:
     x: feature matrix (rows: time, columns: features)
    
    Returns:
     dx: delta matrix
    
    REMARK:
     first delta is calculated by x[0, :] - x[1, :]
    '''
    
    if len(x) <= 1:
        return x
    
    dx0 = np.array([x[0, :] - x[1, :]])
    dx = np.diff(x, axis=0)
    dx = np.row_stack((dx0, dx))
    
    return dx


def cmat(x, y, a=0):

    '''
    table merge
    always returns 2-dim nparray
    input can be 1- or 2-dim
    
    Args:
      x: (np.array)
      y: (np.array)
      a: (int) axis  (0: append rows, 1: append columns)
    
    Returns:
      (np.array): [x y]
    '''
    
    if len(x) == 0 and len(y) == 0:
        return np.array([])
    elif len(x) == 0:
        return lol(y)
    elif len(y) == 0:
        return lol(x)

    return np.concatenate((lol(x), lol(y)), a)


def non_empty_val(d, x):

    '''
    returns True if
      - x is key in dict d and d[x] is non-empty list or string

    Args:
    d: (dict)
    x: (str) keyname

    '''

    if (x not in d) or len(d[x]) == 0:
        return False
    
    return True


def ea(n=1):

    '''
    return empty np.array(s)
    
    Args:
      n: number (up to 5)
    
    Returns:
      np.array([]), ...
    '''

    if n == 1:
        return np.array([])
    if n == 2:
        return np.array([]), np.array([])
    if n == 3:
        return np.array([]), np.array([]), np.array([])
    if n == 4:
        return np.array([]), np.array([]), np.array([]), np.array([])
    if n == 5:
        return np.array([]), np.array([]), np.array([]), \
            np.array([]), np.array([])

    sys.exit("max 5 arrays")


def eai(n=1):

    '''
    return empty np.array(s) of type int
    
    Args:
      n: (int) number (up to 5)
    
    Returns:
      np.array([]), ...
    '''

    if n == 1:
        return np.array([]).astype(int)
    if n == 2:
        return np.array([]).astype(int), np.array([]).astype(int)
    if n == 3:
        return np.array([]).astype(int), np.array([]).astype(int), \
            np.array([]).astype(int)
    if n == 4:
        return np.array([]).astype(int), np.array([]).astype(int), \
            np.array([]).astype(int), np.array([]).astype(int)
    if n == 5:
        return np.array([]).astype(int), np.array([]).astype(int), \
            np.array([]).astype(int), np.array([]).astype(int), \
            np.array([]).astype(int)
    
    sys.exit("max 5 arrays")


def robust_split(x, p):

    '''
    robustly splits array in data above '>' and below '<=' value
    so that always two non-empty index vectors or scalars are returned
    (except of empty list input)
    
    Args:
     x: (np.array)
     p: (float) split value
    
    Returns:
     i0, i1: (np.array-s or int-s) of items in x <= and > p
    '''
    
    if len(x) == 1:
        return 0, 0
    
    if len(x) == 2:
        return np.argmin(x), np.argmax(x)

    else:
        for px in [p, np.mean(x)]:
            i0 = np.where(x <= px)[0]
            i1 = np.where(x > px)[0]
            if min(len(i0), len(i1)) > 0:
                return i0, i1
            
    i = np.arange(0, len(x))
    
    return i, i


def is_type(x):

    '''
    return variable type str|int|float|list (latter incl np.array)
    fallback 'unk'
    
    Args:
      x: (any)
    
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


def rob0(x):

    '''
    robust 0 for division, log, etc
    '''
    
    if x == 0.0:
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
    returns str as list
    
    Args:
     x: (str or list)
    
    Returns:
     x: (list) e.g. [x]

    '''
    
    if type(x) is list:
        return x

    return [x]


def two_dim_array(x):

    '''
    tramsforms 1-dim to 2-dim array
    
    Args:
      x: (np.array) 1-dim
    
    Returns:
      y: (np.array) 2-dim
    '''

    return x.reshape((-1, x.size))


def webservice_output(cmd, opt):

    '''
    collect output from BAS webservices g2p or MAUS
    write into file opt['out'], resp opt['o']
    
    Args:
      cmd: (str) service call string
      opt: (dict)
        fatal: (boolean), if True, exit if unsuccesful
        local: (boolean)
        o/out: (str) outPutFile
    
    Returns:
      (boolean) True if successful
    '''


    opt = opt_default(opt, {'fatal': False})

    print(cmd)
    ans = subprocess.check_output(cmd, shell=True,
                                  universal_newlines=True)
    print(ans)

    # local call -> done
    if opt['local']:
        return

    # parse webservice answer
    m = re.search(r'<downloadLink>(?P<res>(.+?))<\/downloadLink>', ans)

    if m is None:
        if opt['fatal']:
            sys.exit(f"error: {ans}")
        else:
            print(f"error: {ans}")
            return False

    res = m.group('res')
    if res is None:
        if opt['fatal']:
            sys.exit(f"error: {ans}")
        else:
            print(f"error: {ans}")
            return False

    # unifying across maus and g2p
    if 'out' in opt:
        o = opt['out']
    else:
        o = opt['o']

    os.system(f"wget {res} -O {o}")

    return True


def rm_nan(x):

    '''
    remove nan-s from array
    
    Args:
      x: (np.array) 1-dim array
    
    Returns:
      x: (np.array) without nan elements
    '''

    return x[~np.isnan(x)]


def copasul_call(c, as_sudo=False):

    '''
    executes copasul.py from other scripts
    
    Args:
       c: (str) config filename (full path)
       as_sudo: (boolean) if True, sudo-call
    
    '''

    d = os.path.dirname(os.path.realpath(__file__))
    cmd = os.path.join(d, 'copasul.py')
    if as_sudo:
        cmd = f"sudo {cmd}"

    os.system(f"{cmd} -c {c}")


def as_sudo(opt):

    '''
    returns True if opt dict contains as_sudo key with value True

    Args:
    opt: (dict)

    Returns:
    (boolean)

    '''

    if ext_true(opt, 'as_sudo'):
        return True

    return False


def ext_false(d, k):

    '''
    extended False: returns True if (k not in d) OR (d[k] is False)
    
    Args:
     d: (dict)
     k: (str) key name
    
    Returns:
     (boolean)
    '''

    if ((k not in d) or (not d[k])):
        
        return True
    return False


def ext_true(d, k):

    '''
    extended True: returns True if (k in d) AND (d[k] is True)
    
    Args:
     d: (dict)
     k: (str) key name
    
    Returns:
     (boolean)
    '''

    if (k in d) and d[k]:
        return True
    
    return False

    
def count2condEntrop(c):

    '''
    
    Args:
      c: (dict) output of lists2count()
        e.g. c['joint'][x][y]
    
    Returns:
      h: (float) H(Y|X) (order matters)
    '''

    h = 0.0
    for x in c['joint']:
        nx = c['margin'][x]
        if nx == 0:
            continue
        pys = 0.0
        
        for y in c['joint'][x]:
            py = c['joint'][x][y] / nx
            if py == 0:
                continue
            pys += (py * binlog(py))
        h -= ((nx / c['N']) * pys)
        
    return h


def lists2count(x, y):

    '''
    returns dict for cooc counts for 2 input lists
    
    Args:
      x: (list) length n
      y: (list) length n
    
    Returns:
      c: (dict)
        c['joint'][x][y]=myCount f.a. x,y in <xs, ys>
         ['margin_x'][x]=myCount
         ['margin_y'][y]=myCount
         ['N']: len(x)
    '''

    c = {'joint': {}, 'margin_x': {}, 'margin_y': {}}
    n = 0
    
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
      what: (str) 'x','y','x|y','y|x' which prob to calculate
      c: (dict) returned from lists2count()
      x: (str) key name
      y: (str) key name
    
    Returns:
      p: (float) prob
    '''

    if what == 'x':
        return c['margin_x'][x] / c['N']
    if what == 'y':
        return c['margin_y'][y] / c['N']
    if what == 'y|x':
        try:
            return c['joint'][x][y] / c['margin_x'][x]
        except:
            return 0.0
        
    if what == 'x|y':
        try:
            return c['joint'][x][y] / c['margin_y'][y]
        except:
            return 0.0
        
    if what == '-x':
        return (c['N'] - c['margin_x'][x]) / c['N']
    
    if what == '-y':
        return (c['N'] - c['margin_y'][y]) / c['N']
    
    if what == 'y|-x':
        try:
            c_joint = c['margin_y'][y] - c['joint'][x][y]
        except:
            c_joint = c['margin_y'][y]
        c_hist = c['N'] - c['margin_x'][x]
        return c_joint / c_hist
    
    if what == 'x|-y':
        try:
            c_joint = c['margin_x'][x] - c['joint'][x][y]
        except:
            c_joint = c['margin_x'][x]
        c_hist = c['N'] - c['margin_y'][y]
        return c_joint / c_hist
    
    return np.nan


def count2mi(c):

    '''
    returns mutual info based on lists2count() dict
    
    Args:
      c: (dict) returned by lists2count()
    
    Returns:
      mi: (float) mutual information
    '''

    mi = 0
    n = c['N']
    for x in c['joint']:
        for y in c['joint'][x]:
            pxy = c['joint'][x][y] / n
            px = c['margin'][x] / n
            py = c['margin'][y] / n
            if min(px, py, pxy) == 0:
                continue
            mi += (pxy * binlog(pxy / (py * py)))
            
    return mi


def make_pulse(opt):

    '''
    wrapper around Praat pulse extractor
    config analogously to make_f0()
    
    Args:
      opt: (dict)
    
    output into pulse files
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


def make_f0(opt):

    '''
    wrapper around Praat f0 extractor
    
    Args:
      opt: (dict)
    
    output into f0 files
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


def newfig(fs=None):

    '''
    init new figure with onclick->next, keypress->exit
    figsize can be customized
    
    Args:
      fs: (tuple) figure size
    
    Returns:
      fig (handle) figure
    
    '''

    if fs is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=fs)
        
    cid1 = fig.canvas.mpl_connect('button_press_event', fig_next)
    cid2 = fig.canvas.mpl_connect('key_press_event', fig_key)

    return fig


def myPlot(xin=[], yin=[], opt=None):

    '''
    plots x against y(s)
    
    Args:
      x: (np.array or dict) of x values
      y: (np.array or dict) of y values
      opt: (dict)
         bw: boolean; False; black-white
         nrm_x: boolean; False; minmax normalize all x values
         nrm_y: boolean; False; minmax normalize all y values
         ls: dict; {}; linespecs (e.g. '-k'), keys as in y
         lw: dict; {}; linewidths, keys as in y
            ls and lw for y as dict input
         legend_order; []; order of keys in x and y to appear in legend
         legend_lab; {}; mapping of keys to labels in dict
    
    shows plot

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

    # legend
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

    
def myCmd(cmd, opt):

    '''
    returns command string +/- sudo
    
    Args:
      cmd: (str) name of command
      opt: (dict) that can contain key as_sudo with boolean value
    
    Returns:
      o: (sudo) command name +/- sudo prefix
    '''

    if as_sudo(opt):
        return f"sudo {cmd}"
    
    return (cmd)


def is_file(x):

    '''
    returns True if x is file, else False

    Args:
    x: (str)

    Returns:
    (boolean)
    '''

    if os.path.isfile(x):
        return True
    return False


def is_dir(x):

    '''
    returns True if x is dir, else False

    Args:
    x: (str)

    Returns:
    (boolean)
    '''

    if os.path.isdir(x):
        return True
    
    return False


