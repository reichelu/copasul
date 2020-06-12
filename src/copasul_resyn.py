
# author: Uwe Reichel, Budapest, 2016

import mylib as myl
import numpy as np
import os
import copy as cp
import sys
import re

# adds resynthesized f0 contour
# (in Hz; for non-AG segments no resyn values provided, thus set to 0)
# IN:
#   copa
# OUT:
#   +['data'][myFileIdx][myChannelIdx]['f0']['resyn']
def resyn(copa):
    # files
    for ii in myl.numkeys(copa['data']):
        # channels
        for i in myl.numkeys(copa['data'][ii]):
            copa = resyn_channel(copa,ii,i)
    return copa

# called by resyn for single file/channel
def resyn_channel(copa,ii,i):
    opt = copa['config']
    c = copa['data'][ii][i]
    # time
    t = c['f0']['t']
    # base value (in Hz) for ST->Hz transform
    if opt['preproc']['st']: bv = c['f0']['bv']
    # register rep ('bl'|'ml'|'tl'|'rng'|'none')
    reg = opt['styl']['register']
    c['f0']['resyn'] = np.zeros(len(t))
    # glob segs
    for j in myl.numkeys(c['glob']):
        gs = c['glob'][j]
        # time in glob seg
        ttg = np.linspace(gs['t'][0],gs['t'][1],len(gs['decl']['tn']))
        # register line in glob seg
        if re.search('(bl|ml|tl)$',reg):
            y_reg = gs['decl'][reg]['y']
        elif reg=='rng':
            y_reg={'bl': gs['decl']['bl'],
                   'tl': gs['decl']['tl']}
        else:
            y_reg = np.zeros(len(gs['decl'][reg]['y']))
        # loc segs
        for k in c['glob'][j]['ri']:
            # time on|off / f0 in locseg
            tl = c['loc'][k]['t'][0:2]
            yl = cp.deepcopy(c['loc'][k]['acc']['y'])
            # indices of values in c['f0']['resyn'] to be replaced
            yi = myl.find_interval(t,tl)
            # indices in globseg to add register
            gi = myl.find_interval(ttg,tl)
            # +register
            yl = resyn_add_register(yl,y_reg,gi)
            # -> Hz transform
            if opt['preproc']['st']:
                yl = 2**(yl/12)*bv;
            yi,yl = myl.hal(yi,yl)
            c['f0']['resyn'][yi]=yl
    return copa

# add register to local f0 contour
# IN:
#   y   - local f0 contour
#   reg - either vector to add or dict with 'bl' and 'tl' for range de-norm
#   i   - idx in reg corresponding to y
# OUT:
#   y+register
def resyn_add_register(y,reg,i):
    # level de-norm
    if type(reg) is not dict:
        r = reg[i]
        y,r = myl.hal(y,r)
        return y+r
    # range de-norm
    bl = reg['bl'][i]
    tl = reg['tl'][i]
    y,bl = myl.hal(y,bl)
    y,tl = myl.hal(y,tl)
    z = np.asarray([])
    for u in range(len(bl)):
        z = myl.push(z,bl[u]+y[u]*(tl[u]-bl[u]))
    return z


