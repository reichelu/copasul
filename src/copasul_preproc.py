
# author: Uwe Reichel, Budapest, 2016

import os
import mylib as myl
import pandas as pd
import numpy as np
import scipy as si
import sigFunc as sif
import sys
import re
import copy as cp

### main ######################################

# adds:
#    .myFileIdx
#       .file_stem
#       .ii  - input idx (from lists in fsys, starting with 1)
#       .preproc
#         .glob -> [[on off]...]
#         .loc  -> [[on off center]...]
#         .f0   -> [[time f0]...]
#         .bv   -> f0BaseValue

# main preproc bracket
# IN:
#   copa
#   (logFileHandle)
# OUT: (ii=fileIdx, i=channelIdx, j=segmentIdx,
#         k=myTierNameIndex, l=myBoundaryIndex)
#   +['config']
#   +['data'][ii][i]['fsys'][indicesInConfig[Fsys]Lists]
#                       ['f0'|'aud'|'glob'|'loc'|'bnd']['dir'|'stm'|'ext'|'typ']
#                                                ['tier*']
#                       ['f0']['t'|'y'|'bv']
#                       ['glob'][j]['t'|'to'|'ri'|'lab'|'tier']
#                       ['loc'][j]['t'|'to'|'ri'|'lab_ag'|'lab_acc'|'tier_ag'|'tier_acc']
#                       ['bnd'][k][l]['t'|'to'|'lab'|'tier']
#                       ['gnl_f0'][k][l]['t'|'to'|'lab'|'tier']
#                       ['gnl_en'][k][l]['t'|'to'|'lab'|'tier']
#                       ['rhy_f0'][k][l]['t'|'to'|'lab'|'tier']
#                       ['rhy_en'][k][l]['t'|'to'|'lab'|'tier']
def pp_main(copa,f_log_in=''):
    global f_log
    f_log = f_log_in

    myLog("DOING: preprocessing ...")
    
    # detach config
    opt = cp.deepcopy(copa['config'])
    # ff['f0'|'aud'|'annot'|'pulse'] list of full/path/files
    ff = pp_file_collector(opt)
    
    # over files
    for ii in range(len(ff['f0'])):

        #print(ff['annot'][ii]) #!c

        copa['data'][ii]={}
        # f0 and annotation file content
        f0_dat = myl.input_wrapper(ff['f0'][ii],opt['fsys']['f0']['typ'])
        annot_dat = myl.input_wrapper(ff['annot'][ii],opt['fsys']['annot']['typ'])
        # over channels
        for i in range(opt['fsys']['nc']):

            myLog("\tfile {}, channel {}".format(myl.stm(ff['f0'][ii]), i+1))
            
            copa['data'][ii][i]={}
            copa = pp_channel(copa,opt,ii,i,f0_dat,annot_dat,ff,f_log)

    # f0 semitone conversion by grouping factor
    copa = pp_f0_st_wrapper(copa)

    return copa


# f0 semitone conversion by grouping factor
# IN:
#   copa
# OUT:
#   copa with converted f0 values in ['data'][myFi][myCi]['f0']['y']
# REMARKS:
#   groupingValues not differentiated across groupingVariables of different
#   channels
#   e.g. base_prct_grp.1 = 'spk1' (e.g. ='abc')
#        base_prct_grp.2 = 'spk2' (e.g. ='abc')
#     -> 1 base value is calculated for 'abc' and not 2 for spk1_abc and
#        spk2_abc, respectively
def pp_f0_st_wrapper(copa):

    opt = copa['config']

    ## do nothing
    # ('base_prct_grp' is deleted in copasul_init.py if not
    #  needed, incl. the case that base_prct==0)
    if 'base_prct_grp' not in opt['preproc']:
        return copa
    
    ## 1. collect f0 for each grouping level
    # over file/channel idx
    # fg[myGrpLevel] = concatenatedF0
    fg = {}
    # lci[myChannelIdx] = myGroupingVar
    lci = copa['config']['preproc']['base_prct_grp']
    for ii in myl.numkeys(copa['data']):
        for i in myl.numkeys(copa['data'][ii]):
            fg = pp_f0_grp_fg(fg,copa,ii,i,lci)

    ## 2. base value for each grouping level
    #bv[myGrpLevel] = myBaseValue
    bv = pp_f0_grp_bv(fg,opt)

    ## 3. group-wise semitone conversion
    for ii in myl.numkeys(copa['data']):
        for i in myl.numkeys(copa['data'][ii]):
            copa = pp_f0_grp_st(copa,ii,i,bv,lci,opt)

    return copa
    
# towards grp-wise semitone conversion: fg update
# IN:
#   fg: fg[myGrpLevel] = concatenatedF0
#   copa
#   ii: fileIdx
#   i: channelIdx
#   lci: copa['config']['base_prct_grp']
# OUT:
#   fg: updated
def pp_f0_grp_fg(fg,copa,ii,i,lci):
    z = copa['data'][ii][i]
    # channel-related grouping factor level
    x = z['grp'][lci[i]]
    if x not in fg:
        fg[x] = myl.ea()
    fg[x] = np.append(fg[x],z['f0']['y'])
    return fg

# calculate base value for each grouping level
# IN:
#   fg: fg[myGrpLevel] = concatenatedF0
#   opt
# OUT:
#   bv[myGrpLevel] = myBaseValue
def pp_f0_grp_bv(fg,opt):
    # base prct
    b = opt['preproc']['base_prct']
    bv = {}
    for x in fg:
        yi = myl.find(fg[x],'>',0)
        cbv, b = pp_bv(fg[x][yi],opt)
        bv[x] = cbv
        
    return bv

# grp-wise semitone conversion
# IN:
#   copa
#   ii: fileIdx
#   i: channelIdx
#   bv: bv[myGrpLevel]=myBaseValue
#   lci: copa['config']['base_prct_grp']
#   opt
# OUT:
#   copa with st-transformed f0
def pp_f0_grp_st(copa,ii,i,bv,lci,opt):
    # grp value
    gv = copa['data'][ii][i]['grp'][lci[i]]
    y = copa['data'][ii][i]['f0']['y']
    yr, z = pp_semton(y,opt,bv[gv])
    copa['data'][ii][i]['f0']['y'] = yr
    copa['data'][ii][i]['f0']['bv'] = bv[gv]
    return copa


# standalone to modify only grouping fields in copa dict
# (via pp_main all subsequent processing would be overwritten)
# hacky call from copasul.py; opt['navigate']['do_export']
# for cases where json config did contain faulty fsys.grp settings
def pp_grp_wrapper(copa):
    for ii in myl.numkeys(copa['data']):
        for i in myl.numkeys(copa['data'][ii]):
            copa = pp_grp(copa,ii,i)
    return copa


# file x channel-wise filling of copa['data']
# IN:
#   copa
#   opt: copa['config']
#   ii - fileIdx
#   i - channelIdx
#   f0_dat - f0 file content
#   annot_dat - annotation file content
#   ff - file system dict by pp_file_collector()
#   f_log_in - log file handle, for calls from augmentation
#              environment (without pp_main())
# OUT:
#   copa
#   +['data'][ii][i]['fsys'][indicesInConfig[Fsys]Lists]
#                      ['f0'|'aud'|'glob'|'loc'|'bnd']['dir'|'stm'|'ext'|'typ']
#                                                     ['tier']
#                   ['f0']['t'|'y'|'bv']
#                   ['glob'][j]['t'|'to'|'ri'|'lab'|'tier']
#                   ['loc'][j]['t'|'to'|'ri'|'lab_ag'|'lab_acc'|'tier_ag'|'tier_acc']
#                   ['bnd'][k][l]['t'|'to'|'lab'|'tier']
#                   ['gnl_f0'][k][l]['t'|'to'|'lab'|'tier']
#                   ['gnl_en'][k][l]['t'|'to'|'lab'|'tier']
#                   ['rhy_f0'][k][l]['t'|'to'|'lab'|'tier']
#                   ['fyn_en'][k][l]['t'|'to'|'lab'|'tier']
# for one input file's channel i
def pp_channel(copa,opt,ii,i,f0_dat,annot_dat,ff,f_log_in=''):
    global f_log
    f_log = f_log_in

    ## fsys subdict
    # ['i'] - file idx
    # ['f0|aud|annot|glob|loc|bnd|...']['stm|...']
    copa['data'][ii][i]['fsys'] = pp_fsys(opt['fsys'],ff,ii,i)

    ## grouping
    copa = pp_grp(copa,ii,i)

    ## F0 (1) ########################
    # if embedded in augmentation f0 preproc already done
    if 'skip_f0' in opt:
        t = copa['data'][ii][i]['f0']['t']
        y = copa['data'][ii][i]['f0']['y']
        f0 = np.concatenate((t[:,None],y[:,None]),axis=1)
        f0_ut = f0
    else:
        f0, f0_ut = pp_read_f0(f0_dat,opt['fsys']['f0'],i)

    ## chunk ########################
    # list of tiernames for channel i (only one element for chunk)
    tn = pp_tiernames(opt['fsys'],'chunk','tier',i)
    if len(tn)>0:
        stm = copa['data'][ii][i]['fsys']['chunk']['stm']
        chunk, chunk_ut, lab_chunk = pp_read(annot_dat,opt['fsys']['chunk'],tn[0],stm,'chunk')
    else:
        chunk = myl.ea()
    # no chunks -> file
    if len(chunk)==0:
        chunk = np.asarray([[f0[0,0],f0[-1,0]]])
        chunk_ut = np.asarray([[f0_ut[0,0],f0_ut[-1,0]]])
        lab_chunk = opt['fsys']['label']['chunk']
        
    ## glob #########################
    tn = pp_tiernames(opt['fsys'],'glob','tier',i)
    if len(tn)>0:
        stm = copa['data'][ii][i]['fsys']['glob']['stm']
        glb, glb_ut, lab_glb = pp_read(annot_dat,opt['fsys']['chunk'],tn[0],stm,'glob')
        
    else:
        glb = myl.ea()

    # point -> segment tier
    if len(glb)>0 and np.size(glb,1)==1:
        glb, glb_ut, lab_glb = pp_point2segment(glb,glb_ut,lab_glb,chunk,chunk_ut,opt['fsys']['chunk'])
        
    # no glob segs -> chunks
    if len(glb)==0:
        glb=cp.deepcopy(chunk)
        glb_ut=cp.deepcopy(chunk_ut)
        lab_glb=cp.deepcopy(lab_chunk)        
        
    ## loc ##########################
    tn_loc = set()
    tn_acc = pp_tiernames(opt['fsys'],'loc','tier_acc',i)
    tn_ag = pp_tiernames(opt['fsys'],'loc','tier_ag',i)
    stm = copa['data'][ii][i]['fsys']['loc']['stm']
    if len(tn_ag)>0:
        loc_ag, loc_ag_ut, lab_loc_ag = pp_read(annot_dat,opt['fsys']['chunk'],tn_ag[0],stm,'loc')
        tn_loc.add(tn_ag[0])
    else:
        loc_ag, loc_ag_ut, lab_loc_ag = pp_read_empty()
    if len(tn_acc)>0:
        loc_acc, loc_acc_ut, lab_loc_acc = pp_read(annot_dat,opt['fsys']['chunk'],tn_acc[0],stm,'loc')
        tn_loc.add(tn_acc[0])
    else:
        loc_acc, loc_acc_ut, lab_loc_acc = pp_read_empty()
    loc, loc_ut = myl.ea(2)
    
    # [[on off center]...]
    if (len(loc_ag)>0 and len(loc_acc)>0):
        
        # assigning corresponding ag and acc items
        loc,loc_ut,lab_ag,lab_acc = pp_loc_merge(loc_ag_ut,lab_loc_ag,
                                                 loc_acc_ut,lab_loc_acc,
                                                 opt['preproc'])
    # [[on off]...]
    elif len(loc_ag)>0:
        loc = loc_ag
        loc_ut = loc_ag_ut
        lab_ag = lab_loc_ag
        lab_acc = []
    # [[center]...]
    elif len(loc_acc)>0:
        loc = loc_acc
        loc_ut = loc_acc_ut
        lab_ag = []
        lab_acc = lab_loc_acc
        
    # no loc segs
    if len(loc)==0:
        lab_ag = []
        lab_acc = []
        
    ## F0 (2) ################################
    ## preproc + filling copa.f0 #############
    if 'skip_f0' not in opt:
        f0, t, y, bv = pp_f0_preproc(f0,glb[-1][1],opt)
        copa['data'][ii][i]['f0'] = {'t':t, 'y':y, 'bv':bv}
    else:
        # horiz extrapolation only to sync f0 and glob
        # for embedding in augment
        f0 = pp_zp(f0,glb[-1][1],opt,True)
        copa['data'][ii][i]['f0'] = {'t':f0[:,0], 'y':f0[:,1]}
        
    ## error?
    if np.max(y)==0:
         myLog("ERROR! {} contains only zeros that will cause trouble later on.\nPlease remove f0, audio and annotation file from data and re-start the analysis.".format(ff['f0'][ii]),True)

    ## sync onsets of glob and loc to f0 #####
    if len(glb)>0:
        glb[0,0] = np.max([glb[0,0],f0[0,0]])
        glb_ut[0,0] = np.max([glb_ut[0,0],f0[0,0]])
    if len(loc)>0:
        loc[0,0] = np.max([loc[0,0],f0[0,0]])
        loc_ut[0,0] = np.max([loc_ut[0,0],f0[0,0]])
    if len(chunk)>0:
        chunk[0,0] = np.max([chunk[0,0],f0[0,0]])
        chunk_ut[0,0] = np.max([chunk_ut[0,0],f0[0,0]])
        
    # for warnings
    fstm = copa['data'][ii][i]['fsys']['annot']['stm']

    ## copa.chunk ############################
    copa['data'][ii][i]['chunk'] = {}
    jj, bad_j, good_j = 0, np.asarray([]).astype(int), np.asarray([]).astype(int)
    for j in range(len(chunk)):
        if too_short('chunk',chunk[j,],fstm):
            bad_j = np.append(bad_j,j)
            continue
        good_j = np.append(good_j,j)
        copa['data'][ii][i]['chunk'][jj] = {}
        copa['data'][ii][i]['chunk'][jj]['t'] = chunk[j,]
        copa['data'][ii][i]['chunk'][jj]['to'] = chunk_ut[j,]
        if len(lab_chunk)>0:
            copa['data'][ii][i]['chunk'][jj]['lab'] = lab_chunk[j]
        else:
            copa['data'][ii][i]['chunk'][jj]['lab'] = ''
        jj+=1
    if len(bad_j)>0:
        chunk = chunk[good_j,]
        chunk_ut = chunk_ut[good_j,]
    ## copa.glob ############################
    copa['data'][ii][i]['glob'] = {}
    jj, bad_j, good_j = 0, np.asarray([]).astype(int), np.asarray([]).astype(int)
    for j in range(len(glb)):
        if too_short('glob',glb[j,],fstm):
            bad_j = np.append(bad_j,j)
            continue
        good_j = np.append(good_j,j)
        copa['data'][ii][i]['glob'][jj] = {}
        copa['data'][ii][i]['glob'][jj]['t'] = glb[j,]
        copa['data'][ii][i]['glob'][jj]['to'] = glb_ut[j,]
        copa['data'][ii][i]['glob'][jj]['ri'] = np.array([]).astype(int)
        if len(lab_glb)>0:
            copa['data'][ii][i]['glob'][jj]['lab'] = lab_glb[j]
        else:
            copa['data'][ii][i]['glob'][jj]['lab'] = ''
        jj+=1
    if len(bad_j)>0:
        glb = glb[good_j,]
        glb_ut = glb_ut[good_j,]

    # within-chunk position of glb
    rci = pp_apply_along_axis(pp_link,1,glb,chunk)
    for j in myl.idx(glb):
        is_init, is_fin = pp_initFin(rci,j)
        copa['data'][ii][i]['glob'][j]['is_init_chunk'] = is_init
        copa['data'][ii][i]['glob'][j]['is_fin_chunk'] = is_fin
        copa['data'][ii][i]['glob'][j]['tier']=tn[0]
        
    ## copa.loc #############################
    copa['data'][ii][i]['loc'] = {}
    jj, bad_j, good_j = 0, np.asarray([]).astype(int), np.asarray([]).astype(int)
    # [center...] to sync gnl feats if required by opt['preproc']['loc_sync']
    loc_t = myl.ea()
    # row-wise application: uniformly 3 time stamps: [[on off center]...]
    #   - midpoint in case no accent is given
    #   - symmetric window around accent in case no AG is given
    loc = pp_apply_along_axis(pp_loc,1,loc,opt)
    # link each idx in loc.t to idx in glob.t
    # index in ri: index of locseg; value in ri: index of globseg
    ri = pp_apply_along_axis(pp_link,1,loc,glb)
    # ... same for loc.t to chunk.t
    rci = pp_apply_along_axis(pp_link,1,loc,chunk)
    # over segments [[on off center] ...]
    for j in myl.idx(loc):
        # no parenting global segment -> skip
        if ri[j] < 0:
            bad_j = np.append(bad_j,j)
            continue
        # strict layer loc limit (not crossing glb boundaries)
        locSl = pp_slayp(loc[j,:],glb[ri[j],:])
        # [on off] of normalization window (for gnl features)
        # not crossing globseg, not smaller than locSl[1:2]
        loc_tn = pp_loc_w_nrm(loc[j,:],glb[ri[j],:],opt)
        if too_short('loc',locSl,fstm):
            bad_j = np.append(bad_j,j)
            continue
        good_j = np.append(good_j,j)
        copa['data'][ii][i]['loc'][jj] = {}
        copa['data'][ii][i]['loc'][jj]['ri'] = ri[j]
        #### position of local segment in global one and in chunk
        # 'is_fin', 'is_init', both 'yes' or 'no'
        is_init, is_fin = pp_initFin(ri,j)
        #### same for within chunk position
        is_init_chunk, is_fin_chunk = pp_initFin(rci,j)
        copa['data'][ii][i]['loc'][jj]['is_init'] = is_init
        copa['data'][ii][i]['loc'][jj]['is_fin'] = is_fin
        copa['data'][ii][i]['loc'][jj]['is_init_chunk'] = is_init_chunk
        copa['data'][ii][i]['loc'][jj]['is_fin_chunk'] = is_fin_chunk
        if len(tn_ag)>0:
            copa['data'][ii][i]['loc'][jj]['tier_ag'] = tn_ag[0]
        else:
            copa['data'][ii][i]['loc'][jj]['tier_ag'] = ''
        if len(tn_acc)>0:
            copa['data'][ii][i]['loc'][jj]['tier_acc'] = tn_acc[0]
        else:
            copa['data'][ii][i]['loc'][jj]['tier_acc'] = ''
            
        #### labels
        if len(lab_ag)>0:
            copa['data'][ii][i]['loc'][jj]['lab_ag'] = lab_ag[j]
        else:
            copa['data'][ii][i]['loc'][jj]['lab_ag'] = ''
        if len(lab_acc)>0:
            copa['data'][ii][i]['loc'][jj]['lab_acc'] = lab_acc[j]
        else:
            copa['data'][ii][i]['loc'][jj]['lab_acc'] = ''
        copa['data'][ii][i]['loc'][jj]['t'] = locSl
        copa['data'][ii][i]['loc'][jj]['to'] = loc_ut[j,:]
        copa['data'][ii][i]['loc'][jj]['tn'] = loc_tn

        loc_t = np.append(loc_t,locSl[2])
        if (ri[j]>-1):
            copa['data'][ii][i]['glob'][ri[j]]['ri'] = np.concatenate((copa['data'][ii][i]['glob'][ri[j]]['ri'],[jj]),axis=0)
        jj+=1
    if len(bad_j)>0:
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
    for ft in myl.lists('bgd'):
        if ft not in opt['fsys']:
            continue
        r = {} # becomes copa['data'][ii][i][ft]
        # tier idx
        k=0
        lab_pau = opt['fsys'][ft]['lab_pau']
        ## over tiers for channel i
        for tn in pp_tiernames(opt['fsys'],ft,'tier',i):
            # pp_tiernames overgeneralizes in some contexts -> skip
            #   non-existent names
            if not pp_tier_in_annot(tn,annot_dat,opt['fsys'][ft]['typ']):
                continue
            tx, to, lab = pp_read(annot_dat,opt['fsys'][ft],tn,'','bdg')

            # time to intervals (analysis + norm windows)
            # t_nrm: local normalization window limited by chunk boundaries
            # t_trend: windows from chunk onset to boundary, and
            #              from boundary to chunk offset
            t, t_nrm, t_trend = pp_t2i(tx,ft,opt,chunk)
            r[k] = {}
            jj, bad_j, good_j = 0, np.asarray([]).astype(int), np.asarray([]).astype(int)
            # for sync, use each loc interval only once
            blocked_i = {}
            ## over segment index
            for a in myl.idx(lab):
                # skip too short segments until sync with locseg is required
                # that will be checked right below
                if (too_short(tn,t[a,:],fstm) and
                    ((not doSync) or (tn not in tn_loc))):
                    bad_j = np.append(bad_j,a)
                    continue
                if (doSync and (tn in tn_loc)):
                    mfi = myl.find_interval(loc_t,t[a,:])
                    if len(mfi) == 0:
                        bad_j = np.append(bad_j,a)
                        continue
                    # all mfi indices already consumed?
                    all_consumed=True
                    for ij in range(len(mfi)):
                        if mfi[ij] not in blocked_i:
                            ijz=ij
                            all_consumed=False
                            blocked_i[mfi[ij]]=True
                        if not all_consumed:
                            break
                    if all_consumed:
                        bad_j = np.append(bad_j,a)
                        continue
                good_j = np.append(good_j,a)                
                r[k][jj] = {'tier':tn,'t':t[a,:],'to':to[a,:],
                           'tn':t_nrm[a,:],'tt':t_trend[a,:],
                           'lab':lab[a]}
                jj+=1
            if len(bad_j)>0:
                t = t[good_j,]
                tx = tx[good_j,]

            ### position in glob and chunk segments
            # links to parent segments (see above loc, or glb)
            ri = pp_apply_along_axis(pp_link,1,tx,glb)
            rci = pp_apply_along_axis(pp_link,1,tx,chunk)
            # over segment idx
            for j in myl.idx(tx):
                is_init, is_fin = pp_initFin(ri,j)
                is_init_chunk, is_fin_chunk = pp_initFin(rci,j)
                r[k][j]['is_init'] = is_init
                r[k][j]['is_fin'] = is_fin
                r[k][j]['is_init_chunk'] = is_init_chunk
                r[k][j]['is_fin_chunk'] = is_fin_chunk
                
            # rates of tier_rate entries for each segment
            #   (all rate tiers of same channel as tn)
            if re.search('^rhy_',ft):
                #...['tier_rate'] -> npArray of time values (1- or 2-dim)
                tt = pp_tier_time_to_tab(annot_dat,opt['fsys'][ft]['tier_rate'],i,lab_pau)
                # add rate[myTier] = myRate
                #     ri[myTier] = list of reference idx
                # segment index
                #   (too short segments already removed from t)
                for a in range(len(t)):
                    r[k][a]['rate']={}
                    r[k][a]['ri']={}
                    # over rate tiers 
                    for b in tt:
                        rate, ri = pp_rate_in_interval(tt[b],r[k][a]['t'])
                        r[k][a]['rate'][b] = rate
                        r[k][a]['ri'][b] = ri
            k+=1
        
        copa['data'][ii][i][ft] = r
        if ft == 'rhy_f0':
            copa['data'][ii][i]['rate'] = pp_f_rate(annot_dat,opt,ft,i)
    #sys.exit() #!
    return copa

# checks for a segment/time stamp X whether in which position it
# is within the parent segment Y (+/- inital, +/- final)
# IN:
#   ri: list of reverse indices (index in list: index of X in its tier,
#                                value: index of Y in parent tier)
#   j: index of current X
# OUT:
#   is_init: 'yes'|'no' X is in initial position in Y
#   is_fin: 'yes'|'no' X is in final position in Y
def pp_initFin(ri,j):
    # does not belong to any parent segment
    if ri[j] < 0:
        return 'no', 'no'

    # initial?
    if j==0 or ri[j-1] != ri[j]:
        is_init='yes'
    else:
        is_init='no'

    # final?
    if j==len(ri)-1 or ri[j+1] != ri[j]:
        is_fin='yes'
    else:
        is_fin='no'

    return is_init, is_fin

        
# transforms glob point into segment tier
#   - points are considered to be right segment boundaries
#   - segments do not cross chunk boundaries
#   - pause labeled points are skipped
# IN:
#   pnt: [[timePoint]...] of global segment right boundaries
#   pnt_ut [[timePoint]...] same with orig time
#   pnt_lab: [label ...] f.a. timePoint
#   chunk [[on off] ...] of chunks not to be crossed
#   chunk_ut [[on off] ...] same with orig time
#   opt with key 'lab_pau':
# OUT:
#   seg [[on off]...] from pnt
#   seg_ut [[on off]...] from pnt_ut
#   seg_lab [lab ...] from pnt_lab
def pp_point2segment(pnt,pnt_ut,pnt_lab,chunk,chunk_ut,opt): #!g

    # output
    seg, seg_ut, seg_lab = myl.ea(), myl.ea(), []

    # phrase onset, current chunk idx
    t_on, t_on_ut, c_on = chunk[0,0], chunk_ut[0,0], 0

    for i in myl.idx_a(len(pnt)):

        # pause -> only shift onset
        if pp_is_pau(pnt_lab[i],opt['lab_pau']):
            t_on, t_on_ut = pnt[i,0], pnt_ut[i,0]
            c_on = myl.first_interval(t_on,chunk)
            continue

        # current offset
        t_off, t_off_ut = pnt[i,0], pnt_ut[i,0]

        # if needed right shift onset to chunk start
        c_off = myl.first_interval(t_off,chunk)

        if min(c_on,c_off)>-1 and c_off > c_on:
            t_on, t_on_ut = chunk[c_off,0], chunk_ut[c_off,0]

        # update output
        seg = myl.push(seg,[t_on, t_off])
        seg_ut = myl.push(seg_ut,[t_on_ut, t_off_ut])
        seg_lab.append(pnt_lab[i])

        # update time stamps
        t_on = t_off
        t_on_ut = t_off_ut
        c_on = c_off

    return seg, seg_ut, seg_lab

# normalization window for local segment
# - centered on locseg center
# - limited by parenting glb segment
# - not smaller than loc segment
# IN:
#   loc - current local segment [on off center]
#   glb - current global segment [on off]
#   opt - copa['config']
# OUT:
#   tn  - normalization window [on off]
def pp_loc_w_nrm(loc,glb,opt):
    # special window for loc?
    if (('loc' in opt['preproc']) and ('nrm_win' in opt['preproc']['loc'])):
        w = opt['preproc']['loc']['nrm_win']/2
    else:
        w = opt['preproc']['nrm_win']/2
    c = loc[2]
    tn = np.asarray([c-w,c+w])
    tn[0] = max([min([tn[0],loc[0]]),glb[0]])
    tn[1] = min([max([tn[1],loc[1]]),glb[1]])
    return myl.cellwise(myl.trunc2,tn)
    

# signals and log-warns if segment is too short
# IN:
#   type of segment 'chunk|glob|...'
#   seg row ([on off] or [on off center])
#   fileStem for log warning
# OUT:
#   True|False if too short
#   warning message in log file
def too_short(typ,seg,fstm):
    if ((seg[1] <= seg[0]) or
        (len(seg)>2 and (seg[2] < seg[0] or seg[2] > seg[1]))):
        myLog("WARNING! {}: {} segment too short: {} {}. Segment skipped.".format(fstm,typ,seg[0],seg[1]))
        return True
    return False

def rm_too_short(typ,dat,fstm):
    d = dat[:,1]-dat[:,0]
    bad_i = myl.find(d,'<=',0)
    if len(bad_i)>0:
        good_i = myl.find(d,'>',0)
        myLog("WARNING! {}: file contains too short {} segments, which were removed.".format(fstm,typ))
        dat = dat[good_i,]
    return dat


# F0 preprocessing:
#   - zero padding
#   - outlier identification
#   - interpolation over voiceless segments and outliers
#   - smoothing
#   - semtione transform
# IN:
#   f0: [[t f0]...]
#   t_max: max time to which contour is needed
#   opt: copa['config']
# OUT:
#   f0: [[t zero-padded]...]
#   t: time vector
#   y: preprocessed f0 vector
#   bv: f0 base value (Hz)
def pp_f0_preproc(f0,t_max,opt):
    # zero padding
    f0 = pp_zp(f0,t_max,opt)
    # detach time and f0
    t,y = f0[:,0], f0[:,1]
    # do nothing with zero-only segments
    # (-> will be reported as error later on)
    if np.max(y)==0:
        return y, t, y, 1
    # setting outlier to 0
    y = sif.pp_outl(y,opt['preproc']['out'])
    # interpolation over 0
    y = sif.pp_interp(y,opt['preproc']['interp'])
    # smoothing
    if 'smooth' in opt['preproc']:
        y = sif.pp_smooth(y,opt['preproc']['smooth'])
    # <0 -> 0
    y[myl.find(y,'<',0)]=0
    # semitone transform, base ref value (in Hz)
    #     later by calculating the base value over a grp factor (e.g. spk)
    if 'base_prct_grp' in opt['preproc']:
        bv = -1
    else:
        y, bv = pp_semton(y,opt)
    return f0, t, y, bv

# merging AG segment and ACC event tiers to n x 3 array [[on off center]...]
# opt['preproc']['loc_align']='skip': only keeping AGs and ACC for which exactly 1 ACC is within AG
#                             'left': if >1 acc in ag keeping first one
#                             'right': if >1 acc in ag keeping last one
# IN:
#   ag: nx2 [[on off]...] of AGs
#   lab_ag: list of AG labels
#   acc: mx1 [[timeStamp]...]
#   lab_acc: list of ACC labels
#   opt: opt['preproc']
# OUT:
#   d:  ox3 [[on off center]...] ox3, %.2f trunc times
#   d_ut: same not trunc'd 
#   lag: list of AG labels
#   lacc: list of ACC labels
def pp_loc_merge(ag,lab_ag,acc,lab_acc,opt):
    d = myl.ea()
    lag = []
    lacc = []
    for i in range(len(ag)):
        j = myl.find_interval(acc,ag[i,:])
        jj = -1

        #!aa
        #if len(j)>1:
        #    print('err > 1')
        #    myl.stopgo()
        #elif len(j)<1:
        #    print('err < 1')
        #    myl.stopgo()
            
        if len(j)==1:
            jj = j[0]
        elif len(j)>1 and opt['loc_align'] != 'skip':
            if opt['loc_align']=='left':
                jj = j[0]
            elif opt['loc_align']=='right':
                jj = j[-1]
        if jj < 0:
            continue
        
        d = myl.push(d,[ag[i,0],ag[i,1],acc[jj]])
        lag.append(lab_ag[i])
        lacc.append(lab_acc[jj])

    return myl.cellwise(myl.trunc2,d),d,lag,lacc


# grouping values from filename
# IN:
#   copa
#   ii fileIdx
#   i channelIdx
# OUT:
#   +['data'][ii][i]['grp'][myVar] = myVal
def pp_grp(copa,ii,i):
    copa['data'][ii][i]['grp']={}
    # grouping options
    opt = copa['config']['fsys']['grp']
    if len(opt['lab'])>0:
        myStm = copa['data'][ii][i]['fsys'][opt['src']]['stm']
        g = re.split(opt['sep'], myStm)
        for j in myl.idx_a(len(g)):
            if j >= len(opt['lab']):
                myLog("ERROR! {} cannot be split into grouping values".format(myStm),True)
            lab = opt['lab'][j]
            if len(lab)==0: continue
            copa['data'][ii][i]['grp'][lab]=g[j]
    return copa

# robustness wrapper (for non-empty lists only)
def pp_apply_along_axis(fun,dim,var,opt):
    if len(var)>0:
        return np.apply_along_axis(fun,dim,var,opt)
    return []


# file level rates
# IN:
#   tg - annot file dict
#   opt - from opt['fsys'][myFeatSet]
#   ft - featureSetName
#   i - channelIdx
# OUT:
#   rate - dict for rate of myRateTier events/intervals in file
def pp_f_rate(tg,opt,ft,i):
    fsys = opt['fsys'][ft]
    lab_pau = fsys['lab_pau']
    rate={}
    # over tier_rate names
    if 'tier_rate' not in fsys:
        return rate
    # tier names for resp channel
    tn_opt = {'ignore_sylbnd':True}
    for rt in pp_tiernames(opt['fsys'],ft,'tier_rate',i,tn_opt):
        if rt in rate: continue
        # hacky workaround since pp_tiernames() also outputs tier names not in TextGrid
        if rt not in tg['item_name']:
            continue
        if rt not in tg['item_name']:
            # if called by augmentation
            if 'sloppy' in opt:
                myLog("WARNING! Annotation file does not (yet) contain tier {} which is required by the tier_rate element for feature set {}. Might be added by augmentation. If not this missing tier will result in an error later on.".format(rt,ft))
                continue
            else:
                myLog("ERROR! Annotation file does not (yet) contain tier {} which is required by the tier_rate element for feature set {}.".format(rt,ft),True)
        t = tg['item'][tg['item_name'][rt]]
        # file duration
        l = tg['head']['xmax']-tg['head']['xmin']
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
            n=0
            for k in myl.numkeys(t[sk]):
                if pp_is_pau(t[sk][k][fld],lab_pau): continue
                n += 1
            rate[rt] = n/l

    return rate
 


# gives interval or event rate within on and offset in bnd
# IN:
#   t ndarray 1-dim for events, 2-dim for intervals
#   bnd [on off]
# OUT:
#   r - rate
#   ri - ndarray indices of contained segments
def pp_rate_in_interval(t,bnd):
    l = bnd[1]-bnd[0]
    # point tier
    if myl.ndim(t)==1:
        i = myl.find_interval(t,bnd)
        n = len(i)
    # interval tier
    else:
        i = myl.intersect(myl.find(t[:,0],'<',bnd[1]),
                          myl.find(t[:,1],'>',bnd[0]))
        n = len(i)
        # partial intervals within bnd
        if n>0:
            if t[0,0]<bnd[0]:
                n -= (1-((t[i[0],1]-bnd[0])/(t[i[0],1]-t[i[0],0])))
            if t[-1,1]>bnd[1]:
                n -= (1-((bnd[1]-t[i[-1],0])/(t[i[-1],1]-t[i[-1],0])))
        n = max([n,0])
    return n/l, i

# returns time info of tiers as table
# IN:
#   f - textGrid dict
#   rts - list of tiernames to be processed
#   ci - channelIdx
#   lab_pau - pause label
# OUT:
#   tt[myTier] = ndarray of time stamps, resp. on- offsets
# REMARKS:
#   as in pp_read() pause intervals are skipped, thus output of both functions is in sync
def pp_tier_time_to_tab(tg,rts,ci,lab_pau):
    tt={}
    # over tier names for which event rate is to be determined
    for rt in rts:
        x = rt
        if rt not in tg['item_name']:
            ##!!ci crt = "{}_{}".format(rt,ci)
            crt = "{}_{}".format(rt,int(ci+1))
            if crt not in tg['item_name']:
                myLog("WARNING! Tier {} does not exist. Cannot determine event rates for this tier.".format(rt))
                continue
            else:
               x = crt 
        tt[x] = myl.ea()
        t = tg['item'][tg['item_name'][x]]
        if 'intervals' in t:
            for i in myl.numkeys(t['intervals']):
                if pp_is_pau(t['intervals'][i]['text'],lab_pau): continue
                tt[x]=myl.push(tt[x],np.asarray([t['intervals'][i]['xmin'],t['intervals'][i]['xmax']]))
        elif 'points' in t:
            for i in myl.numkeys(t['points']):
                tt[x]=myl.push(tt[x],t['points'][i]['time'])
        tt[x] = myl.cellwise(myl.trunc2,tt[x])
    return tt
            


# returns list of tiernames from fsys of certain TYP in certain channel idx
# IN:
#   fsys subdict (=copa['config']['fsys'] or
#                  copa['config']['fsys']['augment'])
#   fld subfield: 'chunk', 'syl', 'glob', 'loc', 'rhy_*', etc
#   typ tierType: 'tier', 'tier_ag', 'tier_acc', 'tier_rate', 'tier_out_stm' ...
#   ci channelIdx
#   tn_opt <{}> caller-customized options
#      'ignore_sylbnd' TRUE 
# OUT:
#   tn listOfTierNames for channel ci in fsys[fld][typ]
# Remarks:
#    - tn elements are either given in fsys as full names or as stems
#       (for the latter: e.g. *['tier_out_stm']*, ['chunk']['tier'])
#    - In the latter case the full name 'myTierName_myChannelIdx' has been
#      added to the fsys['channel'] keys in copasul_analysis and
#      is added as full name to tn
#    - tn DOES NOT check, whether its elements are contained in annotation file
def pp_tiernames(fsys,fld,typ,ci,tn_opt={}):
    tn = []
    if ((fld not in fsys) or (typ not in fsys[fld])):
        return tn
    # over tiernames
    xx = cp.deepcopy(fsys[fld][typ])
    if type(xx) is not list:
        xx = [xx]
    for x in xx:
            
        # append channel idx for tiers generated in augmentation step
        # add bnd infix for syllable augmentation
        xc = "{}_{}".format(x,int(ci+1))
        if 'ignore_sylbnd' not in tn_opt.keys():
            xbc = "{}_bnd_{}".format(x,int(ci+1))
            yy = [x,xc,xbc]
        else:
            yy = [x,xc]
        for y in yy:
            if ((y in fsys['channel']) and (fsys['channel'][y]==ci)):
                tn.append(y)
    
    if (fld == 'glob' and typ == 'tier' and ('syl' in fsys) and ('tier_out_stm' in fsys['syl'])):
        add = "{}_bnd_{}".format(fsys['syl']['tier_out_stm'],int(ci+1))
        if add not in tn:
            tn.append(add)
    elif (fld == 'loc' and typ == 'tier_acc' and ('syl' in fsys) and ('tier_out_stm' in fsys['syl'])):
        add = "{}_{}".format(fsys['syl']['tier_out_stm'],int(ci+1))
        if add not in tn:
            tn.append(add)
    return tn

### returns input file lists
# IN:
#   option dict
# OUT:
#   ff['f0'|'aud'|'annot'|'pulse'] list of full/path/files
#                          only defined for those keys,
#                          where files are available
# checks for list lengths
def pp_file_collector(opt):
    ff={}
    
    for x in myl.lists('afa'):
        if x not in opt['fsys']:
            continue
        f = myl.file_collector(opt['fsys'][x]['dir'],
                               opt['fsys'][x]['ext'])
        if len(f)>0 or x=='annot':
            ff[x]=f

    # length check
    # file lists must have length 0 or equal length
    # at least one list must have length > 0
    # annotation files can be generated from scratch
    #    in this case stems of f0 (or aud) files are taken over
    for xy in ['f0', 'aud', 'annot']:
        if xy not in ff:
            myLog("ERROR! No {} files found!".format(xy),True)
    l = max(len(ff['f0']),len(ff['aud']),len(ff['annot']))

    #print(len(ff['f0']),len(ff['aud']),len(ff['annot']))
    
    if l==0:
        myLog("ERROR! Neither signal nor annotation files found!",True)
    for x in myl.lists('afa'):
        if x not in ff:
            continue
        if ((len(ff[x])>0) and (len(ff[x]) != l)):
            myLog("ERROR! Numbers of f0/annotation/audio/pulse files must be 0 or equal!",True)
    if len(ff['annot'])==0:
        if ((not opt['fsys']['annot']) or (not opt['fsys']['annot']['dir']) or
            (not opt['fsys']['annot']['ext']) or (not opt['fsys']['annot']['typ'])):
            myLog("ERROR! Directory, type, and extension must be specified for annotation files generated from scratch!",True)
        if len(ff['f0'])>0:
            gg = ff['f0']
        else:
            gg = ff['aud']
        
        for i in range(len(gg)):
            f = os.path.join(opt['fsys']['annot']['dir'],
                             "{}.{}".format(myl.stm(gg[i]),opt['fsys']['annot']['ext']))
            ff['annot'].append(f)

    return ff

### bnd data: time stamps to adjacent intervals
### gnl_*|rhy_* data: time stamps to intervals centered on time stamp
### ! chunk constraint: interval boundaries are limited by chunk boundaries if any
### normalizing time windows
### bb becomes copa...[myFeatSet]['t']
### bb_nrm becomes copa...[myFeatSet]['tn']
### bb_trend becomes copa...[myFeatSet]['tt']
# IN:
#   b - 2-dim time stamp [[x],[x],...] or interval [[x,x],[x,x]...] array
#   typ - 'bnd'|'gnl_*'|'rhy_*'
#   opt - copa['config']
#   t_chunks - mx2 array: [start end] of interpausal chunk segments (at least file [start end])
#   untrunc: <False>|True
#       if True, returns original (not truncated) time values,
#       if False: trunc2
# OUT:
#   bb - nx2 array: [[start end]...]
#           GNL_*, RHY_*: analysis window 
#              segment input: same as input
#              time stamp input: window centered on time stamp (length, see below)
#           BND: adjacent segments
#              segment input: same as input
#              time stamp input: segment between two time stamps (starting from chunk onset)
#   bb_nrm - nx2 array
#           GNL_*, RHY_*: [[start end]...] normalization windows
#              segment input: centered on segment midpoint
#                             minimum length: input segment
#              time stamp input: centered on time stamp
#           BND: uniform boundary styl window independent of length of adjacent segments
#              segment input: [[start segmentOFFSET (segmentONSET) end]...]
#              time stamp input: [[start timeStamp end]...]
#   bb_trend - nx3 array for trend window pairs in current chunk
#              probably relevant for BND only
#           GNL_*, RHY_*:
#              segment input: [[chunkOnset segmentMIDPOINT chunkOffset]...]
#              time stamp input : [[chunkOnset timeStamp chunkOffset]...]
#           BND:
#              segment input: 
#                   non-chunk-final: [[chunkOnset segmentOFFSET segmentONSET chunkOffset]...]
#                   chunk-final: [[chunk[I]Onset segmentOFFSET segmentONSET chunk[I+1]Offset]...]
#                       for ['cross_chunk']=False, chunk-final is same as non-chunk-final with
#                       segmentOFFSET=segmentONSET
#              time stamp input : [[chunkOnset timeStamp chunkOffset]...]
# REMARKS:
#   - all windows (analysis, norm, trend) are limited by chunk boundaries
#   - analysis window length: opt['preproc']['point_win']
#   - norm window length: opt['preproc']['nrm_win'] for GNL_*, RHY_*
#                         opt['bnd']['win'] for BND
#   - BND: - segmentOFFSET and segmentONSET refer to subsequent segments 
#            -- for non-chunk-final segments: always in same chunk
#            -- for chunk-final segments: if ['cross_chunk'] is False than
#               segmentOFFSET is set to segmentONSET.
#               If ['cross_chunk'] is True, then segmentONSET refers to
#               the initial segment of the next chunk and Offset refers to the next chunk
#            -- for non-final segments OR for ['cross_chunk']=True pauses between the 
#               segments are not considered for F0 stylization and pause length is an interpretable
#               boundary feature. For final segments AND ['cross_chunk']=False always zero pause
#               length is measured, so that it is not interpretable (since ONSET==OFFSET)
#               -> for segments always recommended to set ['cross_chunk']=TRUE
#               -> for time stamps depending on the meaning of the markers (if refering to labelled
#                  boundaries, then TRUE is recommended; if referring to other events restricted
#                  to a chunk only, then FALSE)
#          - analogously chunk-final time stamps are processed dep on the ['cross_chunk'] value
#          - for time stamps no difference between final- and non-final position
# a: analysis window
# n: normalization window
# t: trend window
# x: event
## GNL_*, RHY_*
# segment input: [... [...]_a ...]_n
# event input:   [... [.x.]_a ...]_n
## BND
# segment input: [... [...]_a ...]_n
#                  [             ]
# event input: [...|...]_a x [...|...]_a
#                  [             ]_n
def pp_t2i(b,typ,opt,t_chunks,untrunc=False):
    bb, bb_nrm, bb_trend = myl.ea(3)
    ## column number
    # nc=1: time stamps, =2: intervals
    nc = myl.ncol(b)
    ## window half lengths
    # wl - analysis -> bb
    # wl_nrm - normalization -> bb_nrm
    # for bnd: longer norm window in ['styl']['bnd']['win'] is taken
    if ((typ in opt['preproc']) and ('point_win' in opt['preproc'][typ])):
        wl = opt['preproc'][typ]['point_win']/2
    else:
        wl = opt['preproc']['point_win']/2
    if typ == 'bnd':
        wl_nrm = opt['styl']['bnd']['win']/2
    else:
        if ((typ in opt['preproc']) and ('nrm_win' in opt['preproc'][typ])):
            wl_nrm = opt['preproc'][typ]['nrm_win']/2
        else:
            wl_nrm = opt['preproc']['nrm_win']/2

    #### bnd
    if typ=='bnd':
        if nc==1:
            ### time stamp input
            # first onset
            o=t_chunks[0,0]
            for i in range(len(b)):
                # time point: current time stamp
                c = b[i,0]
                ## analysis window
                # chunk idx of onset and current time stamp
                ci1 = myl.first_interval(o,t_chunks)
                ci2 = myl.first_interval(c,t_chunks)
                # next segments chunk to get offset in case of wanted chunk-crossing
                # for trend windows
                if i+1 < len(b) and opt['styl']['bnd']['cross_chunk']:
                    ci3 = myl.first_interval(b[i+1,0],t_chunks)
                else:
                    ci3 = ci2
                # same chunk or chunk-crossing wanted -> adjacent
                if (ci1==ci2 or ci2<0 or opt['styl']['bnd']['cross_chunk']):
                    bb = myl.push(bb,[o,c])
                # different chunks -> onset is chunk onset of current time stamp
                else:
                    bb = myl.push(bb,[t_chunks[ci2,0],c])
                ## nrm window
                ww = pp_limit_win(c,wl_nrm,t_chunks[ci2,:])
                bb_nrm = myl.push(bb_nrm,[ww[0],c,ww[1]])
                ## trend window
                bb_trend = myl.push(bb_trend,[t_chunks[ci2,0],c,t_chunks[ci3,1]])
                # update onset
                o = c
            # last segment: current time stamp to chunk offset
            ci = myl.first_interval(o,t_chunks)
            if ci<0: ci = len(t_chunks)-1
            if o<t_chunks[ci,1]:
                bb = myl.push(bb,[o,t_chunks[ci,1]])
        else:
            ### segment input -> simple copy
            ## analysis windows
            bb = b
            for i in range(len(b)):
                # time point: segment offset
                c = b[i,1]
                # its chunk idx
                ci1 = myl.first_interval(c,t_chunks)
                # its chunk limitations
                r = pp_chunk_limit(c,t_chunks)
                
                # next segment
                if i+1<len(b):
                    # next segments onset
                    c2 = b[i+1,0]
                    # next segment's chunk
                    ci2 = myl.first_interval(c2,t_chunks)
                    # range-offset and next segment's onset for trend window
                    r2t = r[1]
                    c2t = c2
                    # crossing chunk boundaries
                    # -> adjust segmentOnset c2t and chunkOffset r2t for trend window
                    if ci2 > ci1:
                        if opt['styl']['bnd']['cross_chunk']:
                            r2t = t_chunks[ci2,1]
                        else:
                            c2t = c
                        # for norm window
                        c2 = c
                else:
                    c2 = c
                    c2t = c
                    r2t = r[1]
                ## nrm window: limit to chunk boundaries
                ww = pp_limit_win(c,wl_nrm,r)
                if c2 != c:
                    vv = pp_limit_win(c2,wl_nrm,r)
                    bb_nrm = myl.push(bb_nrm,[ww[0],c,c2,vv[1]])
                else:
                    bb_nrm = myl.push(bb_nrm,[ww[0],c,c2,ww[1]])
                ## trend window
                bb_trend = myl.push(bb_trend,[r[0],c,c2t,r2t])
                
    # gnl, rhy
    else:
        if nc>1:
            ### segment input (simple copy)
            ## analysis windows
            bb = b
        # if needed: event -> segment, nrm window
        for i in range(len(b)):
            # center (same for time stamps and segments)
            c = np.mean(b[i,:])
            # chunk bnd limits
            r = pp_chunk_limit(c,t_chunks)
            ### event input
            if nc==1:
                ## analysis window
                bb = myl.push(bb,pp_limit_win(c,wl,r))
            # nrm interval
            oo = pp_limit_win(c,wl_nrm,r)
            # set minimal length to analysis window
            on = min([bb[i,0],oo[0]])
            off = max([bb[i,1],oo[1]])
            ## nrm window
            bb_nrm = myl.push(bb_nrm,[on,off])
            ## trend window
            bb_trend = myl.push(bb_trend,[r[0],c,r[1]])

    if untrunc==False:
        bb = myl.cellwise(myl.trunc2,bb)
        bb_nrm = myl.cellwise(myl.trunc2,bb_nrm)
        bb_trend = myl.cellwise(myl.trunc2,bb_trend)

    

    return bb, bb_nrm, bb_trend

# limits window of HALF length w centered on time stamp c to range r
# IN:
#   c: timeStamp
#   w: window half length
#   r: limitating range
# OUT:
#   s: [on off]
def pp_limit_win(c,w,r):
    on = max([r[0],c-w])
    off = min([r[1],c+w])
    return np.asarray([on,off])

# returns [on off] of chunk in which time stamp is located
# IN:
#   c: time stamp
#   t_chunks [[on off]...] of chunks
def pp_chunk_limit(c,t_chunks):
    ci = myl.first_interval(c,t_chunks)
    if ci<0:
        # fallback: file boundaries
        r = [t_chunks[0,0],t_chunks[-1,1]]
    else:
        # current chunk boundaries
        r = t_chunks[ci,:]
    return r


### add file-system info to dict at file-level
# IN:
#   config['fsys']
#   ff['f0'|'aud'|'annot'] -> list of files
#   ii fileIdx
#   i channelIdx
# OUT:
#   fsys spec
#     [i] - fileIdx
#     ['f0'|'aud'|'glob'|'loc'|...]
#           [stm|dir|typ|tier*|lab*] stem|path|mime|tierNames|pauseEtcLabel
# REMARK:
#   tierNames only for respective channel i
def pp_fsys(fsys,ff,ii,i):
    fs = {'i':ii}
    # 'f0'|'aud'|'augment'|'pulse'|'glob'|'loc'...
    for x in myl.lists('facafa'):
        # skip 'pulse' etc if not available 
        if x not in fsys:
            continue
        # 'f0'|'aud'|'annot'|'pulse' or featSet keys
        if x in ff:
            fs[x]={'stm':myl.stm(ff[x][ii])}
        else:
            fs[x]={'stm':myl.stm(ff['annot'][ii])}
        for y in fsys[x]:
            if y == 'dir':
                if x in ff:
                    fs[x][y] = os.path.dirname(ff[x][ii])
                else:
                    fs[x][y] = os.path.dirname(ff['annot'][ii])
            else:
                fs[x][y] = fsys[x][y]
    return fs


### strict layer principle; limit loc bounds to globseg
# IN:
#   loc 1x3 row from locseg array [on off center]
#   glb 1x2 row spanning loc row from globseg array
# OUT:
#   loc 1x3 row limited to bounds of glob seg
def pp_slayp(loc,glb):
    loc[0] = np.max([loc[0],glb[0]])
    if loc[1] > loc[0]:
        loc[1] = np.min([loc[1],glb[1]])
    else:
        loc[1] = glb[1]
    loc[2] = np.min([loc[1],loc[2]])
    loc[2] = np.max([loc[0],loc[2]])
    return loc

### row linking from loc to globseg ##################
# IN:
#   x row in loc|glob etc (identified by its length;
#                         loc: len 3, other len 1 or 2)
#   y glb matrix
# OUT:
#   i rowIdx in glb
#      (-1 if not connected)
# REMARK: not yet strict layer constrained fulfilled, thus
#      robust assignment
def pp_link(x,y):
    if len(y)==0:
        return -1
    if len(x)>2:
        i = myl.intersect(myl.find(y[:,0],'<=',x[2]),
                          myl.find(y[:,1],'>=',x[2]))
    else:
        m = np.mean(x)
        i = myl.intersect(myl.find(y[:,0],'<=',m),
                          myl.find(y[:,1],'>=',m))
    if len(i)==0:
        i = -1
    else:
        i = i[0]
    return int(i)

# checks whether tier or tier_myChannelIdx is contained in annotation
# IN:
#   tn - tiername
#   an - annotation dict
#   typ - 'xml'|'TextGrid'
#   ci - <0> channel idx
# OUT:
#   True|False
def pp_tier_in_annot(tn,an,typ,ci=0):
    ##!!ci tc = "{}_{}".format(tn,ci)
    tc = "{}_{}".format(tn,int(ci+1))
    if (typ == 'xml' and
        (tn in an or tc in an)):
        return True
    if ((typ == 'TextGrid') and ('item_name' in an) and
        ((tn in an['item_name']) or (tc in an['item_name']))):
        return True
    return False

# checks whether tier is contained in annotation (not checking for
# tier_myChannelIdx as opposed to pp_tier_in_annot()
# IN:
#   tn - tiername
#   an - annotation dict
#   typ - 'xml'|'TextGrid'
# OUT:
#   True|False
def pp_tier_in_annot_strict(tn,an,typ):
    if (typ == 'xml' and (tn in an)):
        return True
    if ((typ == 'TextGrid') and ('item_name' in an) and
        (tn in an['item_name'])):
        return True
    return False

# returns class of tier 'segment'|'event'|''
# IN:
#   tn - tierName
#   annot - annot dict
#   typ - annot type
# OUT: 'segment'|'event'|'' (TextGrid types 'intervals','points' matched segment/event)
def pp_tier_class(tn,annot,typ):
    if typ=='xml':
        if tn in annot:
            return annot[tn]['type']
    elif typ=='TextGrid':
        if tn in annot['item_name']:
            if 'intervals' in annot['item'][annot['item_name'][tn]]:
                return 'segment'
            return 'event'
    return ''


# reads data from table or annotation file
# IN:
#   dat - table or xml or TextGridContent
#   opt - opt['fsys'][myDomain], relevant sub-keys: 'lab_pau', 'typ' in {'tab', 'xml', 'TextGrid'}
#         opt['fsys']['augment'][myDomain]  (relevant subdicts copied there in copasul_analysis:opt_init())
#   tn  - tierName to select content (only relevant for xml and TextGrid)
#   fn - fileName for error messages
#   call - 'glob'|'loc' etc for customized settings (e.g. pauses are not skipped for glob point tier input)
# OUT:
#   d  - 2-d array [[on off]...] or [[timeStamp] ...] values truncated as %.2f
#   d_ut - same as d with original untruncated time values
#   lab - list of labels (empty for 'tab' input)
# REMARK:
#   for TextGrid interval tier input, pause labelled segments are skipped
def pp_read(an,opt,tn='',fn='',call=''):

    lab = []

    ## tab input
    if opt['typ']=='tab':
        d = an
    
    ## xml input
    elif opt['typ']=='xml':
        if not pp_tier_in_annot(tn,an,opt['typ']):
            myLog("ERROR! {}: does not contain tier {}".format(fn,tn))
        d = myl.ea()
        # selected tier
        t = an[tn]
        # 'segment' or 'event'
        tt = t['type']
        for i in myl.numkeys(t['items']):
            lab.append(t['items'][i]['label'])
            if tt=='segment':
                d = myl.push(d,[float(t['items'][i]['t_start']),float(t['items'][i]['t_end'])])
            else:
                d = myl.push(d,float(t['items'][i]['t']))

    ## TextGrid input
    elif opt['typ']=='TextGrid':
        if not pp_tier_in_annot(tn,an,opt['typ']):
            myLog("ERROR! {}: does not contain tier {}".format(fn,tn))
        d = myl.ea()
        # selected tier
        #print(an['item_name']) #!v
        #!e
        t = an['item'][an['item_name'][tn]]
        # 'interals'/'text' or 'points'/'mark'
        if 'intervals' in t:
            kk='intervals'
            kt='text'
        else:
            kk='points'
            kt='mark'

        # skip empty tier
        if kk not in t:
            return d,d,lab

        for i in myl.numkeys(t[kk]):
            if pp_is_pau(t[kk][i][kt],opt['lab_pau']):
                # keep pauses for glob point tier input since used
                # for later transformation into segment tiers
                if not (kk=='points' and call=='glob'):
                    continue
            lab.append(t[kk][i][kt])
            if kk=='intervals':
                d = myl.push(d,[float(t[kk][i]['xmin']),float(t[kk][i]['xmax'])])
            else:
                d = myl.push(d,[float(t[kk][i]['time'])])

    # Warnings
    if len(d)==0:
        if  opt['typ']=='tab':
            myLog("WARNING! {}: empty table\n".format(fn))
        else:
            myLog("WARNING! {}: no labelled segments contained in tier {}. Replacing by default domain\n".format(fn,tn))

    return myl.cellwise(myl.trunc2,d), d, lab

# wrapper around pp_read for f0 input
#   - extract channel i
#   - resample to 100 Hz
# IN:
#   f0_dat: f0 table (1st col: time, 2-end column: channels)
#   opt: opt['fsys']['f0']
#   i: channelIdx
# OUT:
#   f0: [[time f0InChannelI]...]
#   f0_ut: same without %.2f trunc values
def pp_read_f0(f0_dat,opt,i):
    f0, f0_ut, dummy_lab_f0 = pp_read(f0_dat,opt)
    # extract channel from f0 [t f0FromChannelI]
    # i+1 since first column is time
    f0 = f0[:,[0,i+1]]
    f0_ut = f0_ut[:,[0,i+1]]
    # kind of resampling to 100 Hz
    # correct for praat rounding errors
    f0 = pp_t_uniq(f0)
    return f0, f0_ut

# checks whether opt contains field with list at least as long as channel idx,
# and non-empty list or string at this position
# IN:
#   opt dict
#   fld keyName
# OUT: boolean
def pp_opt_contains(opt,fld):
    if (fld in opt):
        return True
    return False

# returns TRUE if label indicates pause (length 0, or pattern match
# with string lab_pau). Else FALSE.
# IN:
#   s label
#   s pause-pattern
# OUT:
#   True|False
def pp_is_pau(lab,lab_pau):
    if lab_pau=='' and len(lab)==0:
        return True
    p = re.compile(lab_pau)
    if len(lab)==0 or p.search(lab):
        return True
    return False

# time stamps might be non-unique and there might be gaps
# due to praat rounding errors
# -> unique and continuous
def pp_t_uniq(x):
    sts = 0.01
    # 1. unique
    t, i = np.unique(x[:,0],return_index=True)
    x = np.concatenate((myl.lol(t).T,myl.lol(x[i,1]).T),axis=1)

    # 2. missing time values?
    tc = np.arange(t[0],t[-1]+sts,sts)
    if len(t)==len(tc):
        return x
    else:
        # add [missingTime 0] rows
        d = set(tc)-set(t)
        if len(d)==0:
            return x

    d = np.asarray(list(d))
    z = np.zeros(len(d))
    add = np.concatenate((myl.lol(d).T,myl.lol(z).T),axis=1)
    x = np.concatenate((x,add),axis=0)
    # include new rows at correct position
    t, i = np.unique(x[:,0],return_index=True)
    x = np.concatenate((myl.lol(t).T,myl.lol(x[i,1]).T),axis=1)
    #for ii in range(len(x)):
    #print(x[ii,])
    #myl.stopgo()
    return x
        

### local segments ###################################
# transform
#   [[center]...] | [[on off]...] 
# to
#   [[on off center]...]
# [center] -> symmetric window
# [on off] -> midpoint
def pp_loc(x,opt):
    # special point win for loc?
    if (('loc' in opt['preproc']) and 'point_win' in opt['preproc']['loc']):
        wl = opt['preproc']['loc']['point_win']/2
    else:
        wl = opt['preproc']['point_win']/2
    if len(x) == 1:
        x = [max(0,x[0]-wl), x[0]+wl, x[0]]
    elif len(x) == 2:
        x = [x[0], x[1], np.mean(x)]

    return myl.cellwise(myl.trunc2,x)
        
### semitone transformation ##########################
def pp_semton(y,opt,bv=-1):
    yi = myl.find(y,'>',0)
    if opt['preproc']['base_prct']>0 and bv < 0:
        bv, b = pp_bv(y[yi],opt)
    elif bv > 0:
        b = max(bv,1)
    else:
        bv, b = 0, 1
    if opt['preproc']['st']==1:
        y[yi] = 12*np.log2(y[yi]/b)
    else:
        y = y - bv
    return y, bv

# calculate base and semitone conversion reference value
# IN:
#   yp: f0 values (>0 only!, see pp_semton())
# OUT:
#   bv: base value in Hz
#   b: semtion conversion reference value (corrected bv)
def pp_bv(yp,opt):
    px = np.percentile(yp,opt['preproc']['base_prct'])
    yy = yp[myl.find(yp,'<=',px)]
    bv = np.median(yp[myl.find(yp,'<=',px)])
    b = max(bv,1)
    return bv, b

### zero padding ##############################
# IN:
#   f0: [[t f0]...]
#   rng_max: max time value for which f0 contour is needed
#   opt: copa['config']
#   extrap: <False> if set then horizontal extrapolation instead of zero pad
# OUT:
#   f0: [[t f0]...] with zero (or horizontal) padding left to first sampled value (at sts),
#                   right to t_max (in sec)
def pp_zp(f0,t_max,opt,extrap=False):
    # stepsize 
    sts = 1/opt['fs']

    if extrap:
        zpl, zpr = f0[0,1], f0[-1,1]
    else:
        zpl, zpr = 0, 0 

    #if sts < f0[0,0]:
    #    prf = np.arange(sts,f0[0,0],sts)
    if 0 < f0[0,0]:
        prf = np.arange(0,f0[0,0],sts)
    else:
        prf = myl.ea()

    if f0[-1,0] < t_max:
        sfx = np.arange(f0[-1,0]+sts,t_max+sts,sts)
    else:
        sfx = myl.ea()
        
    if len(prf)>0:
        zz = zpl*np.ones(len(prf))
        prf = np.concatenate(([prf],[zz]),axis=0).T
        f0 = np.append(prf,f0,axis=0)
    if len(sfx)>0:
        zz = zpr*np.ones(len(sfx))
        sfx = np.concatenate(([sfx],[zz]),axis=0).T
        f0 = np.append(f0,sfx,axis=0)

    return f0


### copa init/preproc diagnosis #######################################
# warnings to logfile:
#   not-linked globseg/locseg
def diagnosis(copa,h_log):
    h_log.write('# Diagnosis\n')
    # error code
    ec = 0
    c = copa['data']
    for ii in myl.numkeys(c):
        for i in myl.numkeys(c):
            ec = diagnosis_seg(c,ii,i,'glob',ec,h_log)
            ec = diagnosis_seg(c,ii,i,'loc',ec,h_log)

    #ec = diagnosis_config(copa['config'],ec,h_log)

    if ec==2:
        pp_log('Too many errors! Exit.',True)
    if ec==0:
        pp_log("Everything seems to be ok!\n")
    return ec

# log file output (if filehandle), else terminal output
# IN:
#   msg message string
#   e <False>|True  do exit
def myLog(msg,e=False):
    global f_log
    try: f_log
    except: f_log = ''
    if type(f_log) is not str:
        f_log.write("{}\n".format(msg))
        if e:
            f_log.close()
            sys.exit(msg)
    else:
        if e:
            sys.exit(msg)
        else:
            print(msg)
        

# checks config syntax
def diagnosis_config(opt,ec,h_log):
    for x in ['f0','glob','loc','bnd','gnl_f0','gnl_en','rhy_f0','rhy_en']:
        if x not in opt['fsys']:
            if x=='f0':
                h_log.write("ERROR! config.fsys does not contain f0 file field.\n")
                ec = 2
            continue
        for y in ['dir','ext','typ']:
            if y not in opt['fsys'][x]:
                 h_log.write("ERROR! config.fsys.{} does not contain {} field.\n".format(x,y))
                 ec = 2
                 continue
            

    # bnd, gnl_* lol
    for x in myl.lists('bgd'):
        ti = []
        tp = []
        ps = []
        if 'tier' in opt['fsys'][x]:
            ti = opt['fsys'][x]['tier']
        if 'lab_pau' in opt['fsys'][x]:
            ps = opt['fsys'][x]['lab_pau']
    return ec
        
# checks initialized copa subdicts glob, loc
# outputs warnings/errors to log file
# returns error code (0=ok, 1=erroneous, 2=fatal)
def diagnosis_seg(c,ii,i,dom,ec,h_log):
    # min seg length
    min_l = 0.03
    f = "{}/{}.{}".format(c[ii][i]['fsys'][dom]['dir'],c[ii][i]['fsys'][dom]['stm'],c[ii][i]['fsys'][dom]['ext'])
    for j in myl.numkeys(c[ii][i][dom]):
        # segment not linked
        if (('ri' not in c[ii][i][dom][j]) or
            ((type(c[ii][i][dom][j]['ri']) is list) and
             len(c[ii][i][dom][j]['ri'])==0) or
            c[ii][i][dom][j]['ri']==''):
                ec = 1
                if dom=='glob':
                    h_log.write("WARNING! {}:interval {} {}:global segment does not dominate any local segment\n".format(f,c[ii][i][dom][j]['to'][0],c[ii][i][dom][j]['to'][1]))
                else:
                    h_log.write("WARNING! {}:interval {} {}:local segment is not dominated by any gobal segment\n",f,c[ii][i][dom][j]['to'][0],c[ii][i][dom][j]['to'][1])
        # segment too short
        if c[ii][i][dom][j]['t'][1]-c[ii][i][dom][j]['t'][0] < min_l:
            h_log.write("ERROR! {}:interval {} {}:{} segment too short!\n".format(f,c[ii][i][dom][j]['to'][0],c[ii][i][dom][j]['to'][1],dom))
            ec = 2

    # locseg center (3rd col) not within [on off] (1st, 2nd col)
    if (dom=='loc' and len(c[ii][i][dom][0]['t'])==3):
        for j in myl.numkeys(c[ii][i][dom]):
            if ((c[ii][i][dom][j]['t'][2] <= c[ii][i][dom][j]['t'][0]) or
                (c[ii][i][dom][j]['t'][2] >= c[ii][i][dom][j]['t'][1])):
                h_log.write("WARNING! {}:interval {} {}:local segments center not within its intervals. Set to midpoint\n",f,c[ii][i][dom][j]['to'][0],c[ii][i][dom][j]['to'][1])
                c[ii][i][dom][j]['t'][2] = myl.trunc2((c[ii][i][dom][j]['t'][0]+c[ii][i][dom][j]['t'][1])/2)
        
    return ec

# returns empty time arrays and label lists (analogously to pp_read()) 
def pp_read_empty():
    return myl.ea(), myl.ea(), []
