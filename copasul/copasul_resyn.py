import copy as cp
import numpy as np
import re

import copasul.copasul_utils as utils


def resyn(copa):

    '''
    adds resynthesized f0 contour
    (in Hz; for non-AG segments no resyn values provided, thus set to 0)
    
    Args:
      copa: (dict)
    
    Returns:
      +['data'][myFileIdx][myChannelIdx]['f0']['resyn']
    '''

    # files
    for ii in utils.sorted_keys(copa['data']):
        # channels
        for i in utils.sorted_keys(copa['data'][ii]):
            copa = resyn_channel(copa, ii, i)
    return copa


def resyn_channel(copa, ii, i):

    '''
    called by resyn for single file/channel
    '''

    opt = copa['config']
    c = copa['data'][ii][i]
    # time
    t = c['f0']['t']
    # base value (in Hz) for ST->Hz transform
    if opt['preproc']['st']:
        bv = c['f0']['bv']
    # register rep ('bl'|'ml'|'tl'|'rng'|'none')
    reg = opt['styl']['register']
    c['f0']['resyn'] = np.zeros(len(t))
    # glob segs
    for j in utils.sorted_keys(c['glob']):
        gs = c['glob'][j]
        # time in glob seg
        ttg = np.linspace(gs['t'][0], gs['t'][1], len(gs['decl']['tn']))
        # register line in glob seg
        if re.search(r'(bl|ml|tl)$', reg):
            y_reg = gs['decl'][reg]['y']
        elif reg == 'rng':
            y_reg = {'bl': gs['decl']['bl'],
                     'tl': gs['decl']['tl']}
        else:
            y_reg = np.zeros(len(gs['decl'][reg]['y']))
        # loc segs
        for k in c['glob'][j]['ri']:
            # time on|off / f0 in locseg
            tl = c['loc'][k]['t'][0:2]
            yl = cp.deepcopy(c['loc'][k]['acc']['y'])
            # indices of values in c['f0']['resyn'] to be replaced
            yi = utils.find_interval(t, tl)
            # indices in globseg to add register
            gi = utils.find_interval(ttg, tl)
            # +register
            yl = resyn_add_register(yl, y_reg, gi)
            # -> Hz transform
            if opt['preproc']['st']:
                yl = 2 ** (yl / 12) * bv
            yi, yl = utils.hal(yi, yl)
            c['f0']['resyn'][yi] = yl
    return copa


def resyn_add_register(y, reg, i):

    '''
    add register to local f0 contour
    
    Args:
      y: (np.array) local f0 contour
      reg: (np.array or dict with values for 'bl' and 'tl') for range de-norm
      i: (index) idx in reg corresponding to y
    
    Returns:
      y+register
    '''

    # level de-norm
    if type(reg) is not dict:
        r = reg[i]
        y, r = utils.hal(y, r)
        return y + r
    
    # range de-norm
    bl = reg['bl'][i]
    tl = reg['tl'][i]
    y, bl = utils.hal(y, bl)
    y, tl = utils.hal(y, tl)
    z = np.asarray([])
    for u in range(len(bl)):
        z = utils.push(z, bl[u] + y[u] * (tl[u] - bl[u]))
    return z

