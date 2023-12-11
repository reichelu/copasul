import copy as cp
import numpy as np
import os
import pandas as pd
import re
import scipy as si
import sys
from tqdm import tqdm

import copasul.copasul_sigproc as cosp
import copasul.copasul_utils as utils


def pp_main(copa, f_log_in=None):

    '''
    main preproc bracket

    adds:
    .myFileIdx
       .file_stem
       .ii  - input idx (from lists in fsys, starting with 1)
       .preproc
         .glob -> [[on off]...]
         .loc  -> [[on off center]...]
         .f0   -> [[time f0]...]
         .bv   -> f0BaseValue
    
    Args:
      copa: (dict)
      f_log_in: logFileHandle
    
    Returns: 
      copa (dict)

      (ii=fileIdx, i=channelIdx, j=segmentIdx,
       k=myTierNameIndex, l=myBoundaryIndex)
      +['config']
      +['data'][ii][i]['fsys'][indicesInConfig[Fsys]Lists]
                          ['f0'|'aud'|'glob'|'loc'|'bnd']['dir'|'stm'|'ext'|'typ']
                                                   ['tier*']
                          ['f0']['t'|'y'|'bv']
                          ['glob'][j]['t'|'to'|'ri'|'lab'|'tier']
                          ['loc'][j]['t'|'to'|'ri'|'lab_ag'|'lab_acc'|'tier_ag'|'tier_acc']
                          ['bnd'][k][l]['t'|'to'|'lab'|'tier']
                          ['gnl_f0'][k][l]['t'|'to'|'lab'|'tier']
                          ['gnl_en'][k][l]['t'|'to'|'lab'|'tier']
                          ['rhy_f0'][k][l]['t'|'to'|'lab'|'tier']
                          ['rhy_en'][k][l]['t'|'to'|'lab'|'tier']
    '''

    global f_log
    f_log = f_log_in

    myLog("DOING: preprocessing ...")

    # detach config
    opt = cp.deepcopy(copa['config'])

    # ff['f0'|'aud'|'annot'|'pulse'] list of full/path/files
    ff = pp_file_collector(opt)
    
    # over files
    for ii in tqdm(range(len(ff['f0'])), desc='preprocessing'):
        
        copa['data'][ii] = {}

        # f0 and annotation file content
        f0_dat = utils.input_wrapper(ff['f0'][ii], opt['fsys']['f0']['typ'])
        annot_dat = utils.input_wrapper(
            ff['annot'][ii], opt['fsys']['annot']['typ'])
        
        # over channels
        for i in range(opt['fsys']['nc']):
            myLog(f"\tfile {utils.stm(ff['f0'][ii])}, channel {i+1}")
            copa['data'][ii][i] = {}
            copa = pp_channel(copa, opt, ii, i, f0_dat, annot_dat, ff, f_log)

    # f0 semitone conversion by grouping factor
    copa = pp_f0_st_wrapper(copa)

    return copa


def pp_f0_st_wrapper(copa):

    '''
    f0 semitone conversion by grouping factor (usually speaker)
    for f0 base value

    Args:
      copa
    
    Returns:
      copa with converted f0 values in ['data'][myFi][myCi]['f0']['y']

    REMARKS:
      groupingValues not differentiated across groupingVariables of different
      channels
      e.g. base_prct_grp.1 = 'spk1' (e.g. ='abc')
           base_prct_grp.2 = 'spk2' (e.g. ='abc')
        -> 1 base value is calculated for 'abc' and not 2 for spk1_abc and
           spk2_abc, respectively
    '''

    opt = copa['config']
    
    # do nothing
    if 'base_prct_grp' not in opt['preproc']:
        return copa

    # 1. collect f0 for each grouping level
    # over file/channel idx
    # fg[myGrpLevel] = concatenatedF0
    fg = {}
    
    # lci[myChannelIdx] = myGroupingVar
    lci = copa['config']['preproc']['base_prct_grp']
    for ii in utils.sorted_keys(copa['data']):
        for i in utils.sorted_keys(copa['data'][ii]):
            fg = pp_f0_grp_fg(fg, copa, ii, i, lci)

    # 2. base value for each grouping level
    # bv[myGrpLevel] = myBaseValue
    bv = pp_f0_grp_bv(fg, opt)
    
    # 3. group-wise semitone conversion
    for ii in utils.sorted_keys(copa['data']):
        for i in utils.sorted_keys(copa['data'][ii]):
            copa = pp_f0_grp_st(copa, ii, i, bv, lci, opt)

    return copa


def pp_f0_grp_fg(fg, copa, ii, i, lci):

    '''
    towards grp-wise semitone conversion: fg update
    
    Args:
      fg: (dict) fg[myGrpLevel] = concatenatedF0
      copa
      ii: (int) fileIdx
      i: (int) channelIdx
      lci: (dict) copa['config']['base_prct_grp']
    
    Returns:
      fg: updated
    '''

    z = copa['data'][ii][i]

    # channel-related grouping factor level
    x = z['grp'][lci[i]]
    if x not in fg:
        fg[x] = np.array([])
    fg[x] = np.append(fg[x], z['f0']['y'])
    return fg


def pp_f0_grp_bv(fg, opt):

    '''
    calculate base value for each grouping level
    
    Args:
      fg: (dict) fg[myGrpLevel] = concatenatedF0
      opt: (dict)
    
    Returns:
      bv: (dict)
         bv[myGrpLevel] = myBaseValue
    '''

    # base prct
    b = opt['preproc']['base_prct']
    bv = {}
    for x in fg:
        yi = np.where(fg[x] > 0)[0]
        cbv, b = pp_bv(fg[x][yi], opt)
        bv[x] = cbv

    return bv



def pp_f0_grp_st(copa, ii, i, bv, lci, opt):

    '''
    grp-wise semitone conversion
    
    Args:
      copa
      ii: (int) fileIdx
      i: (int) channelIdx
      bv: (dict) bv[myGrpLevel]=myBaseValue
      lci: (dict) copa['config']['base_prct_grp']
      opt: (dict)
    
    Returns:
      copa with st-transformed f0
    '''
    
    # grp value
    gv = copa['data'][ii][i]['grp'][lci[i]]
    y = copa['data'][ii][i]['f0']['y']
    yr, z = pp_semton(y, opt, bv[gv])

    copa['data'][ii][i]['f0']['y'] = yr
    copa['data'][ii][i]['f0']['bv'] = bv[gv]

    return copa


def pp_channel(copa, opt, ii, i, f0_dat, annot_dat, ff, f_log_in=None):

    '''
    file x channel-wise filling of copa['data']
    
    Args:
      copa
      opt: (dict) copa['config']
      ii: (int) fileIdx
      i: (int) channelIdx
      f0_dat: (np.array) f0 file content
      annot_dat: (various) annotation file content
      ff: (dict) file system dict by pp_file_collector()
      f_log_in: (handle) log file handle, for calls from augmentation
                 environment (without pp_main())
    
    Returns:
      copa
      +['data'][ii][i]['fsys'][indicesInConfig[Fsys]Lists]
                         ['f0'|'aud'|'glob'|'loc'|'bnd']['dir'|'stm'|'ext'|'typ']
                                                        ['tier']
                      ['f0']['t'|'y'|'bv']
                      ['glob'][j]['t'|'to'|'ri'|'lab'|'tier']
                      ['loc'][j]['t'|'to'|'ri'|'lab_ag'|'lab_acc'|'tier_ag'|'tier_acc']
                      ['bnd'][k][l]['t'|'to'|'lab'|'tier']
                      ['gnl_f0'][k][l]['t'|'to'|'lab'|'tier']
                      ['gnl_en'][k][l]['t'|'to'|'lab'|'tier']
                      ['rhy_f0'][k][l]['t'|'to'|'lab'|'tier']
                      ['fyn_en'][k][l]['t'|'to'|'lab'|'tier']

    for one input file's channel i
    '''

    global f_log
    f_log = f_log_in

    # fsys subdict
    # ['i'] - file idx
    # ['f0|aud|annot|glob|loc|bnd|...']['stm|...']
    copa['data'][ii][i]['fsys'] = pp_fsys(opt['fsys'], ff, ii, i)

    # grouping
    copa = pp_grp(copa, ii, i)

    # F0 (1) ########################
    # if embedded in augmentation f0 preproc already done
    if 'skip_f0' in opt:
        t = copa['data'][ii][i]['f0']['t']
        y = copa['data'][ii][i]['f0']['y']
        f0 = np.concatenate((t[:, None], y[:, None]), axis=1)
        f0_ut = f0
    else:
        f0, f0_ut = pp_read_f0(f0_dat, opt['fsys']['f0'], i)
        
    # chunk ########################
    # list of tiernames for channel i (only one element for chunk)
    tn = pp_tiernames(opt['fsys'], 'chunk', 'tier', i)
    if len(tn) > 0:
        stm = copa['data'][ii][i]['fsys']['chunk']['stm']
        chunk, chunk_ut, lab_chunk = pp_read(
            annot_dat, opt['fsys']['chunk'], tn[0], stm, 'chunk')
    else:
        chunk = np.array([])
        
    # no chunks -> single chunk per file
    if len(chunk) == 0:
        chunk = np.array([[f0[0, 0], f0[-1, 0]]])
        chunk_ut = np.array([[f0_ut[0, 0], f0_ut[-1, 0]]])
        lab_chunk = opt['fsys']['label']['chunk']

    # glob #########################
    tn = pp_tiernames(opt['fsys'], 'glob', 'tier', i)
    if len(tn) > 0:
        stm = copa['data'][ii][i]['fsys']['glob']['stm']
        glb, glb_ut, lab_glb = pp_read(
            annot_dat, opt['fsys']['chunk'], tn[0], stm, 'glob')

    else:
        glb = np.array([])

    # point -> segment tier
    if len(glb) > 0 and np.size(glb, 1) == 1:
        glb, glb_ut, lab_glb = pp_point2segment(
            glb, glb_ut, lab_glb, chunk, chunk_ut, opt['fsys']['chunk'])

    # no glob segs -> use chunks
    if len(glb) == 0:
        glb = cp.deepcopy(chunk)
        glb_ut = cp.deepcopy(chunk_ut)
        lab_glb = cp.deepcopy(lab_chunk)

    # loc ##########################
    tn_loc = set()
    tn_acc = pp_tiernames(opt['fsys'], 'loc', 'tier_acc', i)
    tn_ag = pp_tiernames(opt['fsys'], 'loc', 'tier_ag', i)
    stm = copa['data'][ii][i]['fsys']['loc']['stm']
    if len(tn_ag) > 0:
        loc_ag, loc_ag_ut, lab_loc_ag = pp_read(
            annot_dat, opt['fsys']['chunk'], tn_ag[0], stm, 'loc')
        tn_loc.add(tn_ag[0])
    else:
        loc_ag, loc_ag_ut, lab_loc_ag = pp_read_empty()
    if len(tn_acc) > 0:
        loc_acc, loc_acc_ut, lab_loc_acc = pp_read(
            annot_dat, opt['fsys']['chunk'], tn_acc[0], stm, 'loc')
        tn_loc.add(tn_acc[0])
    else:
        loc_acc, loc_acc_ut, lab_loc_acc = pp_read_empty()
    loc, loc_ut = np.array([]), np.array([])

    # [[on off center]...]
    if (len(loc_ag) > 0 and len(loc_acc) > 0):

        # assigning corresponding ag and acc items
        loc, loc_ut, lab_ag, lab_acc = pp_loc_merge(loc_ag_ut, lab_loc_ag,
                                                    loc_acc_ut, lab_loc_acc,
                                                    opt['preproc'])
    # [[on off]...]
    elif len(loc_ag) > 0:
        loc = loc_ag
        loc_ut = loc_ag_ut
        lab_ag = lab_loc_ag
        lab_acc = []
        
    # [[center]...]
    elif len(loc_acc) > 0:
        loc = loc_acc
        loc_ut = loc_acc_ut
        lab_ag = []
        lab_acc = lab_loc_acc

    # no loc segs
    if len(loc) == 0:
        lab_ag = []
        lab_acc = []

    # F0 (2) ################################
    # preproc + filling copa.f0 #############
    if 'skip_f0' not in opt:
        f0, t, y, bv = pp_f0_preproc(f0, glb[-1][1], opt)
        copa['data'][ii][i]['f0'] = {'t': t, 'y': y, 'bv': bv}
    else:
        # horiz extrapolation only to sync f0 and glob
        # for embedding in augment
        f0 = pp_zp(f0, glb[-1][1], opt, True)
        copa['data'][ii][i]['f0'] = {'t': f0[:, 0], 'y': f0[:, 1]}
        
    # error?
    if np.max(y) == 0:
        myLog(f"ERROR! {ff['f0'][ii]} contains only zeros " \
              "that will cause trouble later on.\n" \
              "Please remove f0, audio and annotation file from data " \
              "and re-start the analysis.", True)

    # sync onsets of glob and loc to f0 #####
    if len(glb) > 0:
        glb[0, 0] = np.max([glb[0, 0], f0[0, 0]])
        glb_ut[0, 0] = np.max([glb_ut[0, 0], f0[0, 0]])

    if len(loc) > 0:
        loc[0, 0] = np.max([loc[0, 0], f0[0, 0]])
        loc_ut[0, 0] = np.max([loc_ut[0, 0], f0[0, 0]])

    if len(chunk) > 0:
        chunk[0, 0] = np.max([chunk[0, 0], f0[0, 0]])
        chunk_ut[0, 0] = np.max([chunk_ut[0, 0], f0[0, 0]])

    # for warnings
    fstm = copa['data'][ii][i]['fsys']['annot']['stm']

    # copa.chunk ############################
    copa['data'][ii][i]['chunk'] = {}
    jj, bad_j, good_j = 0, np.asarray([]).astype(
        int), np.asarray([]).astype(int)
    for j in range(len(chunk)):
        if too_short('chunk', chunk[j,], fstm):
            bad_j = np.append(bad_j, j)
            continue
        good_j = np.append(good_j, j)
        copa['data'][ii][i]['chunk'][jj] = {}
        copa['data'][ii][i]['chunk'][jj]['t'] = chunk[j,]
        copa['data'][ii][i]['chunk'][jj]['to'] = chunk_ut[j,]
        if len(lab_chunk) > 0:
            copa['data'][ii][i]['chunk'][jj]['lab'] = lab_chunk[j]
        else:
            copa['data'][ii][i]['chunk'][jj]['lab'] = ''
        jj += 1

    if len(bad_j) > 0:
        chunk = chunk[good_j,]
        chunk_ut = chunk_ut[good_j,]

    # copa.glob ############################
    copa['data'][ii][i]['glob'] = {}
    jj, bad_j, good_j = 0, np.asarray([]).astype(
        int), np.asarray([]).astype(int)
    for j in range(len(glb)):
        if too_short('glob', glb[j,], fstm):
            bad_j = np.append(bad_j, j)
            continue
        good_j = np.append(good_j, j)
        copa['data'][ii][i]['glob'][jj] = {}
        copa['data'][ii][i]['glob'][jj]['t'] = glb[j,]
        copa['data'][ii][i]['glob'][jj]['to'] = glb_ut[j,]
        copa['data'][ii][i]['glob'][jj]['ri'] = np.array([]).astype(int)
        if len(lab_glb) > 0:
            copa['data'][ii][i]['glob'][jj]['lab'] = lab_glb[j]
        else:
            copa['data'][ii][i]['glob'][jj]['lab'] = ''
        jj += 1
    if len(bad_j) > 0:
        glb = glb[good_j,]
        glb_ut = glb_ut[good_j,]

    # within-chunk position of glb
    rci = pp_apply_along_axis(pp_link, 1, glb, chunk)
    for j in utils.idx(glb):
        is_init, is_fin = pp_initFin(rci, j)
        copa['data'][ii][i]['glob'][j]['is_init_chunk'] = is_init
        copa['data'][ii][i]['glob'][j]['is_fin_chunk'] = is_fin
        copa['data'][ii][i]['glob'][j]['tier'] = tn[0]

        
    # copa.loc #############################
    copa['data'][ii][i]['loc'] = {}
    jj, bad_j, good_j = 0, [], []

    # [center...] to sync gnl feats if required by opt['preproc']['loc_sync']
    loc_t = []

    # row-wise application: uniformly 3 time stamps: [[on off center]...]
    #   - midpoint in case no accent is given
    #   - symmetric window around accent in case no AG is given
    loc = pp_apply_along_axis(pp_loc, 1, loc, opt)

    # link each idx in loc.t to idx in glob.t
    # index in ri: index of locseg; value in ri: index of globseg
    ri = pp_apply_along_axis(pp_link, 1, loc, glb)

    # ... same for loc.t to chunk.t
    rci = pp_apply_along_axis(pp_link, 1, loc, chunk)

    # over segments [[on off center] ...]
    for j in utils.idx(loc):

        # no parenting global segment -> skip
        if ri[j] < 0:
            bad_j.append(j)
            continue

        # strict layer loc limit (not crossing glb boundaries)
        locSl = pp_slayp(loc[j, :], glb[ri[j], :])

        # [on off] of normalization window (for gnl features)
        # not crossing globseg, not smaller than locSl[1:2]
        loc_tn = pp_loc_w_nrm(loc[j, :], glb[ri[j], :], opt)
        if too_short('loc', locSl, fstm):
            bad_j.append(j)
            continue
        good_j.append(j)
        copa['data'][ii][i]['loc'][jj] = {}
        copa['data'][ii][i]['loc'][jj]['ri'] = ri[j]

        # position of local segment in global one and in chunk
        # 'is_fin', 'is_init', both 'yes' or 'no'
        is_init, is_fin = pp_initFin(ri, j)

        # same for within chunk position
        is_init_chunk, is_fin_chunk = pp_initFin(rci, j)
        copa['data'][ii][i]['loc'][jj]['is_init'] = is_init
        copa['data'][ii][i]['loc'][jj]['is_fin'] = is_fin
        copa['data'][ii][i]['loc'][jj]['is_init_chunk'] = is_init_chunk
        copa['data'][ii][i]['loc'][jj]['is_fin_chunk'] = is_fin_chunk
        if len(tn_ag) > 0:
            copa['data'][ii][i]['loc'][jj]['tier_ag'] = tn_ag[0]
        else:
            copa['data'][ii][i]['loc'][jj]['tier_ag'] = ''
        if len(tn_acc) > 0:
            copa['data'][ii][i]['loc'][jj]['tier_acc'] = tn_acc[0]
        else:
            copa['data'][ii][i]['loc'][jj]['tier_acc'] = ''

        # labels
        if len(lab_ag) > 0:
            copa['data'][ii][i]['loc'][jj]['lab_ag'] = lab_ag[j]
        else:
            copa['data'][ii][i]['loc'][jj]['lab_ag'] = ''
        if len(lab_acc) > 0:
            copa['data'][ii][i]['loc'][jj]['lab_acc'] = lab_acc[j]
        else:
            copa['data'][ii][i]['loc'][jj]['lab_acc'] = ''

        # time
        copa['data'][ii][i]['loc'][jj]['t'] = locSl
        copa['data'][ii][i]['loc'][jj]['to'] = loc_ut[j, :]
        copa['data'][ii][i]['loc'][jj]['tn'] = loc_tn

        loc_t.append(locSl[2])
        if (ri[j] > -1):
            copa['data'][ii][i]['glob'][ri[j]]['ri'] = np.concatenate(
                (copa['data'][ii][i]['glob'][ri[j]]['ri'], [jj]), axis=0)
        jj += 1
        
    loc_t = np.array(loc_t)
    if len(bad_j) > 0:
        loc = loc[good_j,]
        loc_ut = loc_ut[good_j,]

    ### bnd, gnl_*, rhy_* input #############################
    # additional tier index layer, since features can be derived
    # from several tiers
    # copa['data'][ii][i][bnd|gnl_*|rhy_*][tierIdx][segmentIdx]
    # (as opposed to chunk, glob, loc)
    # keys: tierNameIdx in opt, values: t, ot, lab, tier
    # in accent augment embedding feature sets will be time-synced
    # to loc segments (see copasul_augment.aug_prep_copy())
    if 'loc_sync' not in opt['preproc']:
        doSync = False
    else:
        doSync = opt['preproc']['loc_sync']
        
    # over feature set (bnd etc)
    for ft in utils.lists('bgd'):
        if ft not in opt['fsys']:
            continue

        # r becomes copa['data'][ii][i][ft]
        r = {}  
        # tier idx
        k = 0
        lab_pau = opt['fsys'][ft]['lab_pau']

        # over tiers for channel i
        for tn in pp_tiernames(opt['fsys'], ft, 'tier', i):
            # pp_tiernames overgeneralizes in some contexts -> skip
            #   non-existent names
            if not pp_tier_in_annot(tn, annot_dat, opt['fsys'][ft]['typ']):
                continue
            tx, to, lab = pp_read(annot_dat, opt['fsys'][ft], tn, '', 'bdg')

            # time to intervals (analysis + norm windows)
            # t_nrm: local normalization window limited by chunk boundaries
            # t_trend: windows from chunk onset to boundary, and
            #              from boundary to chunk offset
            t, t_nrm, t_trend = pp_t2i(tx, ft, opt, chunk)
            r[k] = {}
            jj, bad_j, good_j = 0, [], []

            # for sync, use each loc interval only once
            blocked_i = {}

            # over segment index
            for a in utils.idx(lab):
                # skip too short segments until sync with locseg is required
                # that will be checked right below
                if (too_short(tn, t[a, :], fstm) and
                        ((not doSync) or (tn not in tn_loc))):
                    bad_j.append(a)
                    continue
                if (doSync and (tn in tn_loc)):
                    mfi = utils.find_interval(loc_t, t[a, :])
                    if len(mfi) == 0:
                        bad_j.append(a)
                        continue

                    # all mfi indices already consumed?
                    all_consumed = True
                    for ij in range(len(mfi)):
                        if mfi[ij] not in blocked_i:
                            ijz = ij
                            all_consumed = False
                            blocked_i[mfi[ij]] = True
                        if not all_consumed:
                            break
                    if all_consumed:
                        bad_j.append(a)
                        continue

                good_j.append(a)
                r[k][jj] = {'tier': tn, 't': t[a, :], 'to': to[a, :],
                            'tn': t_nrm[a, :], 'tt': t_trend[a, :],
                            'lab': lab[a]}
                jj += 1
                
            if len(bad_j) > 0:
                t = t[good_j,]
                tx = tx[good_j,]

            # position in glob and chunk segments
            # links to parent segments (see above loc, or glb)
            ri = pp_apply_along_axis(pp_link, 1, tx, glb)
            rci = pp_apply_along_axis(pp_link, 1, tx, chunk)

            # over segment idx
            for j in utils.idx(tx):
                is_init, is_fin = pp_initFin(ri, j)
                is_init_chunk, is_fin_chunk = pp_initFin(rci, j)
                r[k][j]['is_init'] = is_init
                r[k][j]['is_fin'] = is_fin
                r[k][j]['is_init_chunk'] = is_init_chunk
                r[k][j]['is_fin_chunk'] = is_fin_chunk

            # rates of tier_rate entries for each segment
            #   (all rate tiers of same channel as tn)
            if re.search(r'^rhy_', ft):

                # ...['tier_rate'] -> npArray of time values (1- or 2-dim)
                tt = pp_tier_time_to_tab(
                    annot_dat, opt['fsys'][ft]['tier_rate'], i, lab_pau)

                # add rate[myTier] = myRate
                #     ri[myTier] = list of reference idx
                # segment index
                #   (too short segments already removed from t)
                for a in range(len(t)):
                    r[k][a]['rate'] = {}
                    r[k][a]['ri'] = {}

                    # over rate tiers
                    for b in tt:
                        rate, ri = pp_rate_in_interval(tt[b], r[k][a]['t'])
                        r[k][a]['rate'][b] = rate
                        r[k][a]['ri'][b] = ri
            k += 1

        copa['data'][ii][i][ft] = r
        if ft == 'rhy_f0':
            copa['data'][ii][i]['rate'] = pp_f_rate(annot_dat, opt, ft, i)
            
    return copa


def pp_initFin(ri, j):

    '''
    checks for a segment/time stamp X whether in which position it
    is within the parent segment Y (+/- inital, +/- final)
    
    Args:
      ri: (list) of reverse indices (index in list: index of X in its tier,
                                   value: index of Y in parent tier)
      j: (int) index of current X
    
    Returns:
      is_init: (str) 'yes'|'no' X is in initial position in Y
      is_fin: (str) 'yes'|'no' X is in final position in Y
    '''

    # does not belong to any parent segment
    if ri[j] < 0:
        return 'no', 'no'

    # initial?
    if j == 0 or ri[j-1] != ri[j]:
        is_init = 'yes'
    else:
        is_init = 'no'

    # final?
    if j == len(ri) - 1 or ri[j+1] != ri[j]:
        is_fin = 'yes'
    else:
        is_fin = 'no'

    return is_init, is_fin


def pp_point2segment(pnt, pnt_ut, pnt_lab, chunk, chunk_ut, opt):

    '''
    transforms glob point into segment tier
      - points are considered to be right segment boundaries
      - segments do not cross chunk boundaries
      - pause labeled points are skipped
    
    Args:
      pnt: (np.array) [[timePoint]...] of global segment right boundaries
      pnt_ut: (np.array) [[timePoint]...] same with orig time
      pnt_lab: (np.array) [label ...] f.a. timePoint
      chunk: (np.array) [[on off] ...] of chunks not to be crossed
      chunk_ut: (np.array) [[on off] ...] same with orig time
      opt: (dict) with key 'lab_pau'
    
    Returns:
      seg: (np.array) [[on off]...] from pnt
      seg_ut: (np.array) [[on off]...] from pnt_ut
      seg_lab: (list) [lab ...] from pnt_lab
    '''


    # output
    seg, seg_ut, seg_lab = [], [], []

    # phrase onset, current chunk idx
    t_on, t_on_ut, c_on = chunk[0, 0], chunk_ut[0, 0], 0

    for i in utils.idx_a(len(pnt)):

        # pause -> only shift onset
        if pp_is_pau(pnt_lab[i], opt['lab_pau']):
            t_on, t_on_ut = pnt[i, 0], pnt_ut[i, 0]
            c_on = utils.first_interval(t_on, chunk)
            continue

        # current offset
        t_off, t_off_ut = pnt[i, 0], pnt_ut[i, 0]

        # if needed right shift onset to chunk start
        c_off = utils.first_interval(t_off, chunk)

        if min(c_on, c_off) > -1 and c_off > c_on:
            t_on, t_on_ut = chunk[c_off, 0], chunk_ut[c_off, 0]

        # update output
        seg.append([t_on, t_off])
        seg_ut.append([t_on_ut, t_off_ut])
        seg_lab.append(pnt_lab[i])

        # update time stamps
        t_on = t_off
        t_on_ut = t_off_ut
        c_on = c_off

    seg = np.array(seg)
    seg_ut = np.array(seg_ut)
    
    return seg, seg_ut, seg_lab


def pp_loc_w_nrm(loc, glb, opt):

    '''
    normalization window for local segment
    - centered on locseg center
    - limited by parenting glb segment
    - not smaller than loc segment
    
    Args:
      loc: (np.array) current local segment [on off center]
      glb: (np.array) current global segment [on off]
      opt: (dict) copa['config']
    
    Returns:
      tn: (np.array) normalization window [on off]
    '''

    # special window for loc?
    if (('loc' in opt['preproc']) and ('nrm_win' in opt['preproc']['loc'])):
        w = opt['preproc']['loc']['nrm_win'] / 2
    else:
        w = opt['preproc']['nrm_win'] / 2
    c = loc[2]
    tn = np.asarray([c - w, c + w])
    tn[0] = np.max([np.min([tn[0], loc[0]]), glb[0]])
    tn[1] = np.min([np.max([tn[1], loc[1]]), glb[1]])
    tn = np.round(tn, 2)
    
    return tn


def too_short(typ, seg, fstm):

    '''
    signals and log-warns if segment is too short
    
    Args:
      type of segment 'chunk|glob|...'
      seg row ([on off] or [on off center])
      fileStem for log warning
    
    Returns:
      True|False if too short
      warning message in log file
    '''

    if ((seg[1] <= seg[0]) or
            (len(seg) > 2 and (seg[2] < seg[0] or seg[2] > seg[1]))):
        myLog(f"WARNING! {fstm}: {typ} segment too short: {seg[0]} {seg[1]}. " \
              "Segment skipped.")
        return True
    return False


def rm_too_short(typ, dat, fstm):
    d = dat[:, 1] - dat[:, 0]
    bad_i = np.where(d <= 0)[0]
    if len(bad_i) > 0:
        good_i = np.where(d > 0)[0]
        myLog(f"WARNING! {fstm}: file contains too short {typ} segments, " \
              "which were removed.")
        dat = dat[good_i,]
    return dat


def pp_f0_preproc(f0, t_max, opt):

    '''
    F0 preprocessing:
      - zero padding
      - outlier identification
      - interpolation over voiceless segments and outliers
      - smoothing
      - semtione transform
    
    Args:
      f0: (np.array) [[t f0]...]
      t_max: (float) max time to which contour is needed
      opt: (dict) copa['config']
    
    Returns:
      f0: (np.array) [[t_complete zero-padded_f0]...]
      t: (np.array) time vector
      y: (np.array) preprocessed f0 vector
      bv: (float) f0 base value (Hz)
    '''

    # zero padding
    f0 = pp_zp(f0, t_max, opt)

    # detach time and f0
    t, y = f0[:, 0], f0[:, 1]

    # do nothing with zero-only segments
    # (-> will be reported as error later on)
    if np.max(y) == 0:
        return y, t, y, 1

    # setting outlier to 0
    y = cosp.pp_outl(y, opt['preproc']['out'])

    # interpolation over 0
    y = cosp.pp_interp(y, opt['preproc']['interp'])

    # smoothing
    if 'smooth' in opt['preproc']:
        y = cosp.pp_smooth(y, opt['preproc']['smooth'])
        
    # <0 -> 0
    y[y < 0] = 0

    # semitone transform, base ref value (in Hz)
    #     later by calculating the base value over a grp factor (e.g. spk)
    if 'base_prct_grp' in opt['preproc']:
        bv = -1
    else:
        y, bv = pp_semton(y, opt)
        
    return f0, t, y, bv


def pp_loc_merge(ag, lab_ag, acc, lab_acc, opt):

    '''
    merging AG segment and ACC event tiers to n x 3 array [[on off center]...]
    opt['preproc']['loc_align']='skip': only keeping AGs and ACC for which exactly
                                        1 ACC is within AG
                                'left': if >1 acc in ag keeping first one
                                'right': if >1 acc in ag keeping last one
    
    Args:
      ag: (np.array) nx2 [[on off]...] of AGs
      lab_ag: (list) of AG labels
      acc: (np.array) mx1 [[timeStamp]...]
      lab_acc: (list) of ACC labels
      opt: (dict) opt['preproc']
    
    Returns:
      d: (np.array) ox3 [[on off center]...] ox3, %.2f trunc times
      d_ut: (np.array) same not trunc'd
      lag: (list) of AG labels
      lacc: (list) of ACC labels
    '''

    d, lag, lacc = [], [], []
    
    for i in range(len(ag)):
        j = utils.find_interval(acc, ag[i, :])
        jj = -1
        if len(j) == 1:
            jj = j[0]
        elif len(j) > 1 and opt['loc_align'] != 'skip':
            if opt['loc_align'] == 'left':
                jj = j[0]
            elif opt['loc_align'] == 'right':
                jj = j[-1]
        if jj < 0:
            continue

        # update acc[jj] -> acc[jj][0]
        if len(acc[jj]) == 0:
            a = np.mean([ag[i,]])
        else:
            a = acc[jj][0]
        d.append([ag[i, 0], ag[i, 1], a])
        lag.append(lab_ag[i])
        lacc.append(lab_acc[jj])

    d = np.array(d)
    d_rnd = np.round(d, 2)
    
    return d_rnd, d, lag, lacc


def pp_grp(copa, ii, i):

    '''
    grouping values from filename
    
    Args:
      copa: (dict)
      ii: (int) fileIdx
      i: (int) channelIdx
    
    Returns:
      copa + ['data'][ii][i]['grp'][myVar] = myVal
    '''

    copa['data'][ii][i]['grp'] = {}

    # grouping options
    opt = copa['config']['fsys']['grp']
    if len(opt['lab']) > 0:
        myStm = copa['data'][ii][i]['fsys'][opt['src']]['stm']
        g = re.split(opt['sep'], myStm)
        for j in utils.idx_a(len(g)):
            if j >= len(opt['lab']):
                myLog(f"ERROR! {myStm} cannot be split into grouping values",
                      True)
            lab = opt['lab'][j]
            if len(lab) == 0:
                continue
            copa['data'][ii][i]['grp'][lab] = g[j]
    return copa


def pp_apply_along_axis(fun, dim, var, opt):

    '''
    robust wrapper around np.apply_along_axis()
    '''

    if len(var) > 0:
        return np.apply_along_axis(fun, dim, var, opt)
    return []


def pp_f_rate(tg, opt, ft, i):

    '''
    file level rates
    
    Args:
      tg: (dict) annot file dict
      opt: (dict) from opt['fsys'][myFeatSet]
      ft: (str) featureSetName
      i: (int) channelIdx
    
    Returns:
      rate: (dict) for rate of myRateTier events/intervals in file
    '''

    fsys = opt['fsys'][ft]
    lab_pau = fsys['lab_pau']
    rate = {}
    
    if 'tier_rate' not in fsys:
        return rate

    # tier names for resp channel
    tn_opt = {'ignore_sylbnd': True}

    # over tier_rate names
    for rt in pp_tiernames(opt['fsys'], ft, 'tier_rate', i, tn_opt):
        if rt in rate:
            continue

        # workaround since pp_tiernames() also outputs tier names not in TextGrid
        if rt not in tg['item_name']:
            continue

        if rt not in tg['item_name']:

            # if called by augmentation
            if 'sloppy' in opt:
                myLog(f"WARNING! Annotation file does not (yet) contain tier {rt} " \
                      f"which is required by the tier_rate element for feature set {ft}. " \
                      "Might be added by augmentation. If not this missing tier will " \
                      "result in an error later on.")
                continue
            else:
                myLog(f"ERROR! Annotation file does not (yet) contain tier {rt} " \
                      f"which is required by the tier_rate element for feature set {ft}.",
                      True)
                
        t = tg['item'][tg['item_name'][rt]]

        # file duration
        l = tg['head']['xmax'] - tg['head']['xmin']
        if 'intervals' in t:
            sk = 'intervals'
            fld = 'text'
        else:
            sk = 'points'
            fld = 'mark'

        # empty tier
        if sk not in t:
            rate[rt] = 0
        else:
            n = 0
            for k in utils.sorted_keys(t[sk]):
                if pp_is_pau(t[sk][k][fld], lab_pau):
                    continue
                n += 1
            rate[rt] = n / l

    return rate


def pp_rate_in_interval(t, bnd):

    '''
    gives interval or event rate within on and offset in bnd
    
    Args:
      t: (np.array) 1-dim for events, 2-dim for intervals
      bnd: (list) [on off]
    
    Returns:
      r: (float) rate
      ri: (np.array) indices of contained segments
    '''

    l = bnd[1] - bnd[0]

    if utils.ndim(t) == 1:
        # point tier
        i = utils.find_interval(t, bnd)
        n = len(i)    
    else:
        # interval tier
        i = np.where((t[:, 0] < bnd[1]) & (t[:, 1] > bnd[0]))[0]
        n = len(i)
        
        # partial intervals within bnd
        if n > 0:
            if t[0, 0] < bnd[0]:
                n -= (1 - ((t[i[0], 1] - bnd[0]) / (t[i[0], 1] - t[i[0], 0])))
            if t[-1, 1] > bnd[1]:
                n -= (1 - ((bnd[1] - t[i[-1], 0]) / (t[i[-1], 1] - t[i[-1], 0])))
        n = np.max([n, 0])
        
    return n / l, i


def pp_tier_time_to_tab(tg, rts, ci, lab_pau):

    '''
    returns time info of tiers as table
    
    Args:
      f: (dict) textGrid content
      rts: (list) of tiernames to be processed
      ci: (int) channelIdx
      lab_pau: (str) pause label
    
    Returns:
      tt (dict)
        'myTier': (np.array) of time stamps, resp. on- offsets

    REMARKS:
      as in pp_read() pause intervals are skipped, thus output of both functions is in sync
    '''

    tt = {}

    # over tier names for which event rate is to be determined
    for rt in rts:
        x = rt
        if rt not in tg['item_name']:
            crt = f"{rt}_{int(ci+1)}"
            if crt not in tg['item_name']:
                myLog(f"WARNING! Tier {rt} does not exist. " \
                      "Cannot determine event rates for this tier.")
                continue
            else:
                x = crt
                
        tt[x] = []
        t = tg['item'][tg['item_name'][x]]
        if 'intervals' in t:
            for i in utils.sorted_keys(t['intervals']):
                if pp_is_pau(t['intervals'][i]['text'], lab_pau):
                    continue
                tt[x].append([t['intervals'][i]['xmin'], t['intervals'][i]['xmax']])
        elif 'points' in t:
            for i in utils.sorted_keys(t['points']):
                tt[x].append(t['points'][i]['time'])
                
        tt[x] = np.round(tt[x], 2)
        
    return tt


def pp_tiernames(fsys, fld, typ, ci, tn_opt={}):

    '''
    returns list of tiernames from fsys of certain TYP in certain channel idx
    
    Args:
      fsys: (dict) subdict (=copa['config']['fsys'] or
                     copa['config']['fsys']['augment'])
      fld: (str) subfield: 'chunk', 'syl', 'glob', 'loc', 'rhy_*', etc
      typ: (str) tierType: 'tier', 'tier_ag', 'tier_acc', 'tier_rate', 'tier_out_stm' ...
      ci: (int) channelIdx
      tn_opt: (dict) <{}> caller-customized options
         'ignore_sylbnd' TRUE
    
    Returns:
      tn: (list) list of tier names for channel ci in fsys[fld][typ]

    Remarks:
       - tn elements are either given in fsys as full names or as stems
          (for the latter: e.g. *['tier_out_stm']*, ['chunk']['tier'])
       - In the latter case the full name 'myTierName_myChannelIdx' has been
         added to the fsys['channel'] keys in copasul_analysis and
         is added as full name to tn
       - tn DOES NOT check, whether its elements are contained in annotation file
    '''

    tn = []
    ci = int(ci)
    if ((fld not in fsys) or (typ not in fsys[fld])):
        return tn

    # over tiernames
    xx = cp.deepcopy(fsys[fld][typ])
    if type(xx) is not list:
        xx = [xx]
    for x in xx:

        # append channel idx for tiers generated in augmentation step
        # add bnd infix for syllable augmentation
        xc = f"{x}_{ci+1}"
        if 'ignore_sylbnd' not in tn_opt.keys():
            xbc = f"{x}_bnd_{ci+1}"
            yy = [x, xc, xbc]
        else:
            yy = [x, xc]
            
        for y in yy:
            if ((y in fsys['channel']) and (fsys['channel'][y] == ci)):
                tn.append(y)

    if (fld == 'glob' and typ == 'tier' and 'syl' in fsys and
        'tier_out_stm' in fsys['syl']):
        add = f"{fsys['syl']['tier_out_stm']}_bnd_{ci+1}"
        if add not in tn:
            tn.append(add)
    elif (fld == 'loc' and typ == 'tier_acc' and 'syl' in fsys and
          'tier_out_stm' in fsys['syl']):
        add = f"{fsys['syl']['tier_out_stm']}_{ci+1}"
        if add not in tn:
            tn.append(add)
            
    return tn


def pp_file_collector(opt):

    '''
    returns input file lists
    
    Args:
      opt: (dict)
    
    Returns:
      ff: (dict)
        ['f0'|'aud'|'annot'|'pulse'] list of full/path/files
                             only defined for those keys,
                             where files are available
    
    checks for list lengths
    '''

    ff = {}
    
    for x in utils.lists('afa'):
        if x not in opt['fsys']:
            continue
        f = utils.file_collector(opt['fsys'][x]['dir'],
                                 opt['fsys'][x]['ext'])
        if len(f) > 0 or x == 'annot':
            ff[x] = f
            
    # length check
    # file lists must have length 0 or equal length
    # at least one list must have length > 0
    # annotation files can be generated from scratch
    #    in this case stems of f0 (or aud) files are taken over
    for xy in ['f0', 'aud', 'annot']:
        if xy not in ff:
            myLog(f"ERROR! No {xy} files found!", True)
    l = max(len(ff['f0']), len(ff['aud']), len(ff['annot']))

    if l == 0:
        myLog("ERROR! Neither signal nor annotation files found!", True)

    for x in utils.lists('afa'):
        if x not in ff:
            continue
        if ((len(ff[x]) > 0) and (len(ff[x]) != l)):
            myLog(
                "ERROR! Numbers of f0/annotation/audio/pulse files must be 0 or equal!", True)

    if len(ff['annot']) == 0:
        if ((not opt['fsys']['annot']) or (not opt['fsys']['annot']['dir']) or
            (not opt['fsys']['annot']['ext']) or (not opt['fsys']['annot']['typ'])):
            myLog("ERROR! Directory, type, and extension must be specified for " \
                  "annotation files generated from scratch!", True)
        if len(ff['f0']) > 0:
            gg = ff['f0']
        else:
            gg = ff['aud']

        for i in range(len(gg)):
            f = os.path.join(opt['fsys']['annot']['dir'],
                             f"{utils.stm(gg[i])}.{opt['fsys']['annot']['ext']}")
            ff['annot'].append(f)

    return ff


def pp_t2i(b, typ, opt, t_chunks, untrunc=False):

    '''
    bnd data: time stamps to adjacent intervals
    gnl_*|rhy_* data: time stamps to intervals centered on time stamp
    chunk constraint: interval boundaries are limited by chunk boundaries if any
    normalizing time windows
    bb becomes copa...[myFeatSet]['t']
    bb_nrm becomes copa...[myFeatSet]['tn']
    bb_trend becomes copa...[myFeatSet]['tt']
    
    Args:
      b: (np.array) 2-dim time stamp [[x],[x],...] or interval array
      typ: (str) 'bnd'|'gnl_*'|'rhy_*'
      opt: (dict) copa['config']
      t_chunks: (np.array) mx2 array: [start end] of interpausal chunk segments
              (at least file [start end])
      untrunc: (boolean)
          if True, returns original (not truncated) time values,
          if False: round2
    
    Returns:
      bb: (np.array) nx2 array: [[start end]...]
              GNL_*, RHY_*: analysis window
                 segment input: same as input
                 time stamp input: window centered on time stamp (length, see below)
              BND: adjacent segments
                 segment input: same as input
                 time stamp input: segment between two time stamps (starting from chunk onset)
      bb_nrm: (np.array) nx2 array
              GNL_*, RHY_*: [[start end]...] normalization windows
                 segment input: centered on segment midpoint
                                minimum length: input segment
                 time stamp input: centered on time stamp
              BND: uniform boundary styl window independent of length of adjacent segments
                 segment input: [[start segmentOFFSET (segmentONSET) end] ...]
                 time stamp input: [[start timeStamp end] ...]
      bb_trend: (np.array) nx3 array for trend window pairs in current chunk
                 probably relevant for BND only
              GNL_*, RHY_*:
                 segment input: [[chunkOnset segmentMIDPOINT chunkOffset] ...]
                 time stamp input : [[chunkOnset timeStamp chunkOffset] ...]
              BND:
                 segment input:
                      non-chunk-final: [[chunkOnset segmentOFFSET segmentONSET chunkOffset] ...]
                      chunk-final: [[chunk[I]Onset segmentOFFSET segmentONSET chunk[I+1]Offset] ...]
                          for ['cross_chunk']=False, chunk-final is same as non-chunk-final with
                          segmentOFFSET=segmentONSET
                 time stamp input : [[chunkOnset timeStamp chunkOffset] ...]

    REMARKS:
      - all windows (analysis, norm, trend) are limited by chunk boundaries
      - analysis window length: opt['preproc']['point_win']
      - norm window length: opt['preproc']['nrm_win'] for GNL_*, RHY_*
                            opt['bnd']['win'] for BND
      - BND: - segmentOFFSET and segmentONSET refer to subsequent segments
               -- for non-chunk-final segments: always in same chunk
               -- for chunk-final segments: if ['cross_chunk'] is False than
                  segmentOFFSET is set to segmentONSET.
                  If ['cross_chunk'] is True, then segmentONSET refers to
                  the initial segment of the next chunk and Offset refers to the next chunk
               -- for non-final segments OR for ['cross_chunk']=True pauses between the
                  segments are not considered for F0 stylization and pause length is an interpretable
                  boundary feature. For final segments AND ['cross_chunk']=False always zero pause
                  length is measured, so that it is not interpretable (since ONSET==OFFSET)
                  -> for segments always recommended to set ['cross_chunk']=TRUE
                  -> for time stamps depending on the meaning of the markers (if refering to labelled
                     boundaries, then TRUE is recommended; if referring to other events restricted
                     to a chunk only, then FALSE)
             - analogously chunk-final time stamps are processed dep on the ['cross_chunk'] value
             - for time stamps no difference between final- and non-final position
    a: analysis window
    n: normalization window
    t: trend window
    x: event
    GNL_*, RHY_*
    segment input: [... [...]_a ...]_n
    event input:   [... [.x.]_a ...]_n
    BND
    segment input: [... [...]_a ...]_n
                     [             ]
    event input: [...|...]_a x [...|...]_a
                     [             ]_n
    '''

    bb, bb_nrm, bb_trend = [], [], []

    # column number
    #   1: time stamps, 2: intervals
    nc = b.shape[1]

    # window half lengths
    # wl - analysis -> bb
    # wl_nrm - normalization -> bb_nrm
    # for bnd: longer norm window in ['styl']['bnd']['win'] is taken
    if ((typ in opt['preproc']) and ('point_win' in opt['preproc'][typ])):
        wl = opt['preproc'][typ]['point_win'] / 2
    else:
        wl = opt['preproc']['point_win'] / 2
        
    if typ == 'bnd':
        wl_nrm = opt['styl']['bnd']['win'] / 2
    else:
        if ((typ in opt['preproc']) and ('nrm_win' in opt['preproc'][typ])):
            wl_nrm = opt['preproc'][typ]['nrm_win'] / 2
        else:
            wl_nrm = opt['preproc']['nrm_win'] / 2

    if typ == 'bnd':
        # feature set: bnd
        
        if nc == 1:
            
            # time stamp input
            # first onset
            o = t_chunks[0, 0]
            for i in range(len(b)):

                # time point: current time stamp
                c = b[i, 0]

                # analysis window
                # chunk idx of onset and current time stamp
                ci1 = utils.first_interval(o, t_chunks)
                ci2 = utils.first_interval(c, t_chunks)

                # next segments chunk to get offset in case of wanted chunk-crossing
                # for trend windows
                if i+1 < len(b) and opt['styl']['bnd']['cross_chunk']:
                    ci3 = utils.first_interval(b[i+1, 0], t_chunks)
                else:
                    ci3 = ci2
                    
                if (ci1 == ci2 or ci2 < 0 or opt['styl']['bnd']['cross_chunk']):
                    # same chunk or chunk-crossing wanted -> adjacent
                    bb.append([o, c])
                else:
                    # different chunks -> onset is chunk onset of current time stamp
                    bb.append([t_chunks[ci2, 0], c])
                    
                # nrm window
                ww = pp_limit_win(c, wl_nrm, t_chunks[ci2, :])
                bb_nrm.append([ww[0], c, ww[1]])
                
                # trend window
                bb_trend.append([t_chunks[ci2, 0], c, t_chunks[ci3, 1]])

                # update onset
                o = c

            # last segment: current time stamp to chunk offset
            ci = utils.first_interval(o, t_chunks)
            if ci < 0:
                ci = len(t_chunks) - 1
            if o < t_chunks[ci, 1]:
                bb.append([o, t_chunks[ci, 1]])
        else:
            # segment input
            bb = b
            
            for i in range(len(b)):
                
                # time point: segment offset
                c = b[i, 1]

                # its chunk idx
                ci1 = utils.first_interval(c, t_chunks)

                # its chunk limitations
                r = pp_chunk_limit(c, t_chunks)

                # next segment
                if i+1 < len(b):
                    # next segments onset
                    c2 = b[i+1, 0]

                    # next segment's chunk
                    ci2 = utils.first_interval(c2, t_chunks)

                    # range-offset and next segment's onset for trend window
                    r2t = r[1]
                    c2t = c2

                    # crossing chunk boundaries
                    # -> adjust segmentOnset c2t and chunkOffset r2t for trend window
                    if ci2 > ci1:
                        if opt['styl']['bnd']['cross_chunk']:
                            r2t = t_chunks[ci2, 1]
                        else:
                            c2t = c

                        # for norm window
                        c2 = c
                else:
                    c2 = c
                    c2t = c
                    r2t = r[1]
                    
                # nrm window: limit to chunk boundaries
                ww = pp_limit_win(c, wl_nrm, r)
                if c2 != c:
                    vv = pp_limit_win(c2, wl_nrm, r)
                    bb_nrm.append([ww[0], c, c2, vv[1]])
                else:
                    bb_nrm.append([ww[0], c, c2, ww[1]])

                # trend window
                bb_trend.append([r[0], c, c2t, r2t])

    else:
        # feature sets: gnl, rhy

        # segment input
        if nc > 1:
            bb = b
            
        # nrm windows
        for i in range(len(b)):
            
            # center (same for time stamps and segments)
            c = np.mean(b[i, :])
            
            # chunk bnd limits
            r = pp_chunk_limit(c, t_chunks)

            # event input
            if nc == 1:
                
                # analysis window
                bb.append(list(pp_limit_win(c, wl, r)))
    
            # nrm interval
            oo = pp_limit_win(c, wl_nrm, r)

            # set minimal length to analysis window
            on = min([bb[i][0], oo[0]])
            off = max([bb[i][1], oo[1]])

            # nrm window
            bb_nrm.append([on, off])
            
            # trend window
            bb_trend.append([r[0], c, r[1]])

    bb = np.array(bb)
    bb_nrm = np.array(bb_nrm)
    bb_trend = np.array(bb_trend)
            
    if untrunc == False:
        bb = np.round(bb, 2)
        bb_nrm = np.round(bb_nrm, 2)
        bb_trend = np.round(bb_trend, 2)
        
    return bb, bb_nrm, bb_trend


def pp_limit_win(c, w, r):

    '''
    limits window of HALF length w centered on time stamp c to range r
    
    Args:
      c: (float) timeStamp
      w: (float) window half length
      r: (list) limitating range
    
    Returns:
      s: (np.array) [on off]
    '''
    
    on = np.max([r[0], c - w])
    off = np.min([r[1], c + w])
    return np.array([on, off])


def pp_chunk_limit(c, t_chunks):

    '''
    returns [on off] of chunk in which time stamp is located
    
    Args:
      c: (float) time stamp
      t_chunks: (np.array) [[on off]...] of chunks

    Returns:
      r: (np.array) [on, off] of current chunk
    '''

    ci = utils.first_interval(c, t_chunks)
    if ci < 0:
        # fallback: file boundaries
        r = [t_chunks[0, 0], t_chunks[-1, 1]]
    else:
        # current chunk boundaries
        r = t_chunks[ci, :]
    return r


def pp_fsys(fsys, ff, ii, i):

    '''
    add file-system info to dict at file-level
    
    Args:
      fsys: (dict) config['fsys']
      ff: (dict) ['f0'|'aud'|'annot'] -> list of files
      ii: (int) fileIdx
      i: (int) channelIdx
    
    Returns:
      fs: (dict) spec
        [i] - fileIdx
        ['f0'|'aud'|'glob'|'loc'|...]
              [stm|dir|typ|tier*|lab*] stem|path|mime|tierNames|pauseEtcLabel
    
    REMARK:
      tierNames only for respective channel i
    '''

    fs = {'i': ii}

    # 'f0'|'aud'|'augment'|'pulse'|'glob'|'loc'...
    for x in utils.lists('facafa'):

        # skip 'pulse' etc if not available
        if x not in fsys:
            continue

        # 'f0'|'aud'|'annot'|'pulse' or featSet keys
        if x in ff:
            fs[x] = {'stm': utils.stm(ff[x][ii])}
        else:
            fs[x] = {'stm': utils.stm(ff['annot'][ii])}

        for y in fsys[x]:
            if y == 'dir':
                if x in ff:
                    fs[x][y] = os.path.dirname(ff[x][ii])
                else:
                    fs[x][y] = os.path.dirname(ff['annot'][ii])
            else:
                fs[x][y] = fsys[x][y]

    return fs


def pp_slayp(loc, glb):

    '''
    keep strict layer principle: limit loc bounds to glob segments
    
    Args:
      loc: (np.array) 1x3 row from locseg array [on off center]
      glb: (np.array) 1x2 row spanning loc row from globseg array
    
    Returns:
      loc: (np.array) 1x3 row limited to bounds of glob seg
    '''

    loc[0] = np.max([loc[0], glb[0]])
    if loc[1] > loc[0]:
        loc[1] = np.min([loc[1], glb[1]])
    else:
        loc[1] = glb[1]
        
    loc[2] = np.min([loc[1], loc[2]])
    loc[2] = np.max([loc[0], loc[2]])

    return loc


def pp_link(x, y):

    '''
    row linking from loc to globseg

    Args:
      x: [np.array] row in loc|glob etc (identified by its length;
                            loc: len 3, other len 1 or 2)
      y: [np.array] glb 2-dim array
    
    Returns:
      i: (int) rowIdx in glb
         (-1 if no linking possible)

    REMARK: not yet strict layer constrained fulfilled, thus
         robust assignment
    '''

    if len(y) == 0:
        return -1
    
    if len(x) > 2:
        i = np.where((y[:, 0] <= x[2]) & (y[:, 1] >= x[2]))[0]
    else:
        m = np.mean(x)
        i = np.where((y[:, 0] <= m) & (y[:, 1] >= m))[0]

    if len(i) == 0:
        i = -1
    else:
        i = i[0]

    return int(i)


def pp_tier_in_annot(tn, an, typ, ci=0):

    '''
    checks whether tier or tier_myChannelIdx is contained in annotation
    
    Args:
      tn: (str) tiername
      an: (dict) annotation dict
      typ: (str) 'xml'|'TextGrid'
      ci: (int) channel idx
    
    Returns:
      (boolean)
    '''

    ci = int(ci)
    
    tc = f"{tn}_{ci+1}"
    if (typ == 'xml' and
        (tn in an or tc in an)):
        return True
    
    if ((typ == 'TextGrid') and ('item_name' in an) and
        ((tn in an['item_name']) or (tc in an['item_name']))):
        return True

    return False


def pp_tier_in_annot_strict(tn, an, typ):

    '''
    checks whether tier is contained in annotation (not checking for
    tier_myChannelIdx as opposed to pp_tier_in_annot())
    
    Args:
      tn: (str) tiername
      an: (dict) annotation dict
      typ: (str) 'xml'|'TextGrid'
    
    Returns:
      True|False
    '''

    if (typ == 'xml' and (tn in an)):
        return True
    
    if ((typ == 'TextGrid') and ('item_name' in an) and
            (tn in an['item_name'])):
        return True

    return False


def pp_tier_class(tn, annot, typ):

    '''
    returns class of tier
    
    Args:
      tn: (str) tierName
      annot: (dict) annot dict
      typ: (str) annot type
    
    Returns: 'segment'|'event'|None

    Remark:
      TextGrid types 'intervals','points' matched segment/event
    '''

    if typ == 'xml':
        if tn in annot:
            return annot[tn]['type']
    elif typ == 'TextGrid':
        if tn in annot['item_name']:
            if 'intervals' in annot['item'][annot['item_name'][tn]]:
                return 'segment'
            return 'event'

    return None


def pp_read(an, opt, tn='', fn='', call=''):

    '''
    reads data from table or annotation file
    
    Args:
      dat: (np.array, dict) table or xml or TextGridContent
      opt: (dict)
          opt['fsys'][myDomain], relevant sub-keys: 'lab_pau', 'typ' in {'tab', 'xml', 'TextGrid'}
          opt['fsys']['augment'][myDomain]
               (relevant subdicts copied there in copasul_analysis:opt_init())
      tn: (str) tierName to select content (only relevant for xml and TextGrid)
      fn: (str) fileName for error messages
      call: (str) 'glob'|'loc' etc for customized settings
          (e.g. pauses are not skipped for glob point tier input)
    
    Returns:
      d: (np.array) 2-d array [[on off]...] or [[timeStamp] ...] values truncated as %.2f
      d_ut: (np.array) same as d with original untruncated time values
      lab: (list) of labels (empty for 'tab' input)
    
    REMARK:
      for TextGrid interval tier input, pause labelled segments are skipped
    '''

    lab = []

    # tab input
    if opt['typ'] == 'tab':
        d = an

    # xml input
    elif opt['typ'] == 'xml':
        if not pp_tier_in_annot(tn, an, opt['typ']):
            myLog(f"ERROR! {fn}: does not contain tier {tn}")
        d = []

        # selected tier
        t = an[tn]

        # 'segment' or 'event'
        tt = t['type']
        for i in utils.sorted_keys(t['items']):
            lab.append(t['items'][i]['label'])
            if tt == 'segment':
                d.append([float(t['items'][i]['t_start']),
                          float(t['items'][i]['t_end'])])
            else:
                d.append(float(t['items'][i]['t']))

    # TextGrid input
    elif opt['typ'] == 'TextGrid':
        if not pp_tier_in_annot(tn, an, opt['typ']):
            myLog(f"ERROR! {fn}: does not contain tier {tn}")
        d = []
        
        # selected tier
        t = an['item'][an['item_name'][tn]]

        # 'interals'/'text' or 'points'/'mark'
        if 'intervals' in t:
            kk = 'intervals'
            kt = 'text'
        else:
            kk = 'points'
            kt = 'mark'

        # skip empty tier
        if kk not in t:
            return d, d, lab

        for i in utils.sorted_keys(t[kk]):
            if pp_is_pau(t[kk][i][kt], opt['lab_pau']):

                # keep pauses for glob point tier input since used
                # for later transformation into segment tiers
                if not (kk == 'points' and call == 'glob'):
                    continue

            lab.append(t[kk][i][kt])
            if kk == 'intervals':
                d.append([float(t[kk][i]['xmin']),
                          float(t[kk][i]['xmax'])])
            else:
                d.append([float(t[kk][i]['time'])])

    # Warnings
    if len(d) == 0:
        if opt['typ'] == 'tab':
            myLog(f"WARNING! {fn}: empty table\n")
        else:
            myLog(f"WARNING! {fn}: no labelled segments contained in tier {tn}. " \
            "Replacing by default domain\n")

    if type(d) is list:
        d = np.array(d)
    d_rnd = np.round(d, 2)
            
    return d_rnd, d, lab


def pp_read_f0(f0_dat, opt, i):

    '''
    wrapper around pp_read for f0 input
      - extract channel i
      - unique on time stamps
    
    Args:
      f0_dat: (np.array) f0 table (1st col: time, 2-end column: channels)
      opt: (dict) opt['fsys']['f0']
      i: (int) channelIdx
    
    Returns:
      f0: (np.array) [[time f0InChannelI]...]
      f0_ut: (np.array) same without %.2f trunc values
    '''

    f0, f0_ut, dummy_lab_f0 = pp_read(f0_dat, opt)

    # extract channel from f0 [t f0FromChannelI]
    # i+1 since first column is time
    f0 = f0[:, [0, i+1]]
    f0_ut = f0_ut[:, [0, i+1]]

    # correct for praat rounding errors
    f0 = pp_t_uniq(f0)

    return f0, f0_ut


def pp_is_pau(lab, lab_pau):

    '''
    returns TRUE if label indicates pause (length 0, or pattern match
    with string lab_pau). Else FALSE.
    
    Args:
      lab: (str) label
      lab_pau: (str) pause-pattern
    
    Returns:
      (bool)
    '''

    if lab_pau == '' and len(lab) == 0:
        return True

    p = re.compile(lab_pau)
    if len(lab) == 0 or p.search(lab):
        return True

    return False


def pp_t_uniq(x):

    '''
    correct Praat time stamps might be non-unique and might contain gaps
    
    Args:
    x: (np.array)
    
    Returns:
    x: (np.array) fixed

    '''

    sts = 0.01
    
    # 1. unique
    t, i = np.unique(x[:, 0], return_index=True)
    x = np.column_stack((t, x[i, 1:]))
    
    # 2. missing time values?
    tc = np.arange(t[0], t[-1] + sts, sts)

    if len(t) == len(tc):
        return x
    else:
        d = set(tc) - set(t)
        if len(d) == 0:
            return x
        
    # add [missingTime 0] rows
    d = np.array(sorted(d))
    z = np.zeros(len(d))
    add = np.column_stack((d, z))
    x = np.concatenate((x, add), axis=0)

    # move new rows to correct time positions
    t, i = np.unique(x[:, 0], return_index=True)
    x = np.column_stack((t, x[i, 1:]))
    
    return x


def pp_loc(x, opt):

    '''
    local segments

    transform
      [[center]...] | [[on off]...]
    to
      [[on off center]...]

    Args:
    x: (np.array) 1 row; interval or even time stamps
    opt: (dict)

    Returns:
    x (np.array) [on, off, center]

    '''

    # special point win for loc?
    if (('loc' in opt['preproc']) and 'point_win' in opt['preproc']['loc']):
        wl = opt['preproc']['loc']['point_win'] / 2
    else:
        wl = opt['preproc']['point_win'] / 2
        
    if len(x) == 1:
        # center to [on, off, center]
        x = [max(0, x[0] - wl), x[0] + wl, x[0]]
    elif len(x) == 2:
        # intervall to [on, off, center]
        x = [x[0], x[1], np.mean(x)]

    x = np.round(x, 2)
        
    return x


def pp_semton(y, opt, bv=-1.0):

    '''
    semitone conversion

    Args:
    y: (np.array) of f0 values
    opt: (dict)
    bv: (float) base value

    Returns:
    y: (np.array) converted values
    bv: (float) f0 base value
    '''

    # transform positive values only
    yi = np.where(y > 0)[0]

    # base value
    if opt['preproc']['base_prct'] > 0 and bv < 0:
        bv, b = pp_bv(y[yi], opt)
    elif bv > 0:
        b = max(bv, 1)
    else:
        bv, b = 0, 1

    if opt['preproc']['st'] == 1:
        # ST conversion relative to BV
        y[yi] = 12 * np.log2(y[yi] / b)
    else:
        # BV subtraction on Hz scale
        y = y - bv
        
    return y, bv


def pp_bv(yp, opt):

    '''
    calculate base and semitone conversion reference value
    
    Args:
      yp: (np.array) f0 values (> 0 only!, see pp_semton())
      opt: (dict)

    Returns:
      bv: (float) base value in Hz
      b: (float) semtion conversion reference value (corrected bv)
    '''

    px = np.percentile(yp, opt['preproc']['base_prct'])
    px = np.round(px, 8)
    ypr = np.round(yp, 8)
    bv = np.median(ypr[ypr <= px])
    b = max(bv, 1)
    
    return bv, b


def pp_zp(f0, t_max, opt, extrap=False):

    '''
    zero padding
    
    Args:
      f0: (np.array) [[t f0]...]
      t_max: (float) max time value for which f0 contour is needed
      opt: (dict) copa['config']
      extrap: (boolean) if set then horizontal extrapolation instead of zero pad
    
    Returns:
      f0: (np.array) [[t f0]...] with zero (or horizontal) padding
                      - left to first sampled value (at sts),
                      - right to t_max (in sec)
    '''

    # stepsize
    sts = 1 / opt['fs']

    if extrap:
        zpl, zpr = f0[0, 1], f0[-1, 1]
    else:
        zpl, zpr = 0, 0

    if 0 < f0[0, 0]:
        prf = np.arange(0, f0[0, 0], sts)
    else:
        prf = np.array([])

    if f0[-1, 0] < t_max:
        sfx = np.arange(f0[-1, 0] + sts, t_max + sts, sts)
    else:
        sfx = np.array([])

    if len(prf) > 0:
        zz = zpl * np.ones(len(prf))
        prf = np.concatenate(([prf], [zz]), axis=0).T
        f0 = np.append(prf, f0, axis=0)
    if len(sfx) > 0:
        zz = zpr * np.ones(len(sfx))
        sfx = np.concatenate(([sfx], [zz]), axis=0).T
        f0 = np.append(f0, sfx, axis=0)

    return f0


def diagnosis(copa, h_log):

    '''
    print copa init/preproc diagnosis
    warnings to logfile

    Args:
    copa: (dict)
    h_log: (handle)

    Returns:
    ec: (int) error code

    '''

    h_log.write('# Diagnosis\n')
    
    # error code
    ec = 0

    c = copa['data']
    for ii in utils.sorted_keys(c):
        for i in utils.sorted_keys(c):
            ec = diagnosis_seg(c, ii, i, 'glob', ec, h_log)
            ec = diagnosis_seg(c, ii, i, 'loc', ec, h_log)

    if ec == 2:
        pp_log('Too many errors! Exit.', True)
    if ec == 0:
        pp_log("Everything seems to be ok!\n")
        
    return ec


def myLog(msg, e=False):

    '''
    log file output (if filehandle), else terminal output
    
    Args:
      msg: (str) message string
      e: (boolean) if True,  do exit
    '''

    global f_log
    try:
        f_log
    except:
        f_log = None
    if f_log is None:
        if e:
            sys.exit(msg)
        else:
            print(msg)
    else:
        f_log.write(f"{msg}\n")
        if e:
            f_log.close()
            sys.exit(msg)


def diagnosis_config(opt, ec, h_log):

    '''
    checks config syntax
    '''

    for x in ['f0', 'glob', 'loc', 'bnd', 'gnl_f0', 'gnl_en', 'rhy_f0', 'rhy_en']:
        if x not in opt['fsys']:
            if x == 'f0':
                h_log.write(
                    "ERROR! config.fsys does not contain f0 file field.\n")
                ec = 2
            continue
        for y in ['dir', 'ext', 'typ']:
            if y not in opt['fsys'][x]:
                h_log.write(
                    f"ERROR! config.fsys.{x} does not contain {y} field.\n")
                ec = 2
                continue

    # bnd, gnl_* lol
    for x in utils.lists('bgd'):
        ti = []
        tp = []
        ps = []
        if 'tier' in opt['fsys'][x]:
            ti = opt['fsys'][x]['tier']
        if 'lab_pau' in opt['fsys'][x]:
            ps = opt['fsys'][x]['lab_pau']
    return ec



def diagnosis_seg(c, ii, i, dom, ec, h_log):

    '''
    checks initialized copa subdicts glob, loc
    outputs warnings/errors to log file
    returns error code (0=ok, 1=erroneous, 2=fatal)
    '''

    # min seg length
    min_l = 0.03
    f = "{}/{}.{}".format(c[ii][i]['fsys'][dom]['dir'],
                          c[ii][i]['fsys'][dom]['stm'],
                          c[ii][i]['fsys'][dom]['ext'])
    for j in utils.sorted_keys(c[ii][i][dom]):
        # segment not linked
        if (('ri' not in c[ii][i][dom][j]) or
            ((type(c[ii][i][dom][j]['ri']) is list) and
             len(c[ii][i][dom][j]['ri']) == 0) or
                c[ii][i][dom][j]['ri'] == ''):
            ec = 1
            if dom == 'glob':
                h_log.write(f"WARNING! {f}:interval {c[ii][i][dom][j]['to'][0]} " \
                            f"{c[ii][i][dom][j]['to'][1]}:global segment does not " \
                            "dominate any local segment\n")
            else:
                h_log.write(f"WARNING! {f}:interval {c[ii][i][dom][j]['to'][0]} " \
                            f"{c[ii][i][dom][j]['to'][1]}:local segment is not dominated " \
                            "by any gobal segment\n")
        # segment too short
        if c[ii][i][dom][j]['t'][1]-c[ii][i][dom][j]['t'][0] < min_l:
            h_log.write(f"ERROR! {f}:interval {c[ii][i][dom][j]['to'][0]} " \
                        f"{c[ii][i][dom][j]['to'][1]}:{dom} segment too short!\n")
            ec = 2

    # locseg center (3rd col) not within [on off] (1st, 2nd col)
    if (dom == 'loc' and len(c[ii][i][dom][0]['t']) == 3):
        for j in utils.sorted_keys(c[ii][i][dom]):
            if ((c[ii][i][dom][j]['t'][2] <= c[ii][i][dom][j]['t'][0]) or
                    (c[ii][i][dom][j]['t'][2] >= c[ii][i][dom][j]['t'][1])):
                h_log.write(f"WARNING! {f}:interval {c[ii][i][dom][j]['to'][0]} " \
                            f"{c[ii][i][dom][j]['to'][1]}:local segments center " \
                            "not within its intervals. Set to midpoint\n")
                cmn = (c[ii][i][dom][j]['t'][0] + c[ii][i][dom][j]['t'][1]) / 2
                c[ii][i][dom][j]['t'][2] = np.round(cmn, 2)

    return ec



def pp_read_empty():

    '''
    returns empty time arrays and label lists (analogously to pp_read())
    '''

    return np.array([]), np.array([]), []

