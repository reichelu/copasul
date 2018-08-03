#!/usr/bin/env python3

# author: Uwe Reichel, Budapest, 2016

import mylib as myl
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copasul_plot as copl
import scipy.stats as sps
import copy as cp
import sigFunc as sif
import sys
import re
import math

###############################################################
#### general f0 and en features (mean and sd) #################
###############################################################
# IN:
#   copa
#   typ 'f0'|'en'
#   f_log handle
# OUT: (ii=fileIdx, i=channelIdx, j=tierIdx, k=segmentIdx)
#   + ['data'][ii][i]['gnl_f0|en_file']; see styl_std_feat()
#   + ['data'][ii][i]['gnl_f0|en'][j][k]['std'][*]; see styl_std_feat()
#   ['gnl_en']...['std'] additionally contains ...['sb'] for spectral balance
def styl_gnl(copa,typ,f_log_in=''):
    global f_log
    f_log = f_log_in

    fld = "gnl_{}".format(typ)
    myl.check_var({'var':copa,'env':'copasul',
                   'spec':{'step':'styl','dom':fld}})

    if fld in copa['config']['styl']:
        opt = cp.deepcopy(copa['config']['styl'][fld])
    else:
        opt={}
    opt['type']=typ
    myl.check_var({'var':copa,'env':'copasul',
                   'spec':{'step':'styl','dom':fld}})

    # over files
    for ii in myl.numkeys(copa['data']):
        copa = styl_gnl_file(copa,ii,fld,opt)

    return copa

# case typ='en':
#   y_raw_in, t_raw refer to raw signal
#   y_raw - resp channel of signal
#   y, t refer to energy contour
# myFs: sample freq for F0 or energy contour (derived by opt...['sts']); not for raw signal
def styl_gnl_file(copa,ii,fld,opt):

    # wav of f0 input (wav to be loaded once right here)
    if opt['type']=='en':
        iii = myl.numkeys(copa['data'][ii])
        fp = copa['data'][ii][iii[0]]['fsys']['aud']
        f = "{}/{}.{}".format(fp['dir'],fp['stm'],fp['ext'])
        y_raw_in, fs_sig = sif.wavread(f)
    else:
        myFs = copa['config']['fs']

    #print(copa['data'][ii][iii[0]]['fsys']['annot'])

    # over channels
    for i in myl.numkeys(copa['data'][ii]):
        if opt['type']=='f0':
            t = copa['data'][ii][i]['f0']['t']
            y = copa['data'][ii][i]['f0']['y']
            opt['fs']=myFs
        else:
            if np.ndim(y_raw_in)>1:
                y_raw = y_raw_in[:,i]
            else:
                y_raw = y_raw_in
            # preemphasis for spectral balance calculation
            # deprec
            #y_raw_pe = sif.pre_emphasis(y_raw,opt['alpha'],fs_sig)
            t_raw = myl.smp2sec(myl.idx_seg(1,len(y_raw)),fs_sig,0)
            # signal fs for energy extraction
            opt['fs']=fs_sig
            # energy
            y = sif.sig_energy(y_raw,opt)
            myFs = fs_sig2en(opt['sts'],opt['fs'])
            t = myl.smp2sec(myl.idx_seg(1,len(y)),myFs,0)
            # energy fs for standard feature extraction
            opt['fs']=myFs

        # file wide
        yz = "{}_file".format(fld)
        copa['data'][ii][i][yz] = styl_std_feat(y,opt)
        copa['data'][ii][i][yz] = styl_std_quot(y,opt,copa['data'][ii][i][yz])

        # over tiers
        for j in myl.numkeys(copa['data'][ii][i][fld]):
            # over segments (+ normalization ['*_nrm'])
            nk = myl.numkeys(copa['data'][ii][i][fld][j])
            for k in nk:
                # analysis window
                yi = styl_yi(copa['data'][ii][i][fld][j][k]['t'],myFs,y)

                # for interval tier input; for styl_std_feat duration calculation
                if len(copa['data'][ii][i][fld][j][k]['to'])>1:
                    to = copa['data'][ii][i][fld][j][k]['to'][0:2]
                else:
                    to = myl.ea()
                tn = copa['data'][ii][i][fld][j][k]['tn']
                sf = styl_std_feat(y[yi],opt,to)
                # nrm window
                yin = styl_yi(tn,myFs,y)
                # add normalization _nrm
                sf = styl_std_nrm(sf,y[yin],opt,tn)
                # quotients/shape
                sf = styl_std_quot(y[yi],opt,sf)
                # add spectral balance and rmsd for en
                if (opt['type']=='en'):
                    yi_raw = styl_yi(copa['data'][ii][i][fld][j][k]['t'],fs_sig,y_raw)
                    yin_raw = styl_yi(tn,fs_sig,y_raw)
                    rms_y = myl.rmsd(y_raw[yi_raw])
                    rms_yn = myl.rmsd(y_raw[yin_raw])
                    #sb = myl.rmsd(y_raw_pe[yi_raw])-rms_y
                    #sb = splh_spl_deprec(y_raw[yi_raw],y_raw_pe[yi_raw])
                    sb = sif.splh_spl(y_raw[yi_raw],fs_sig,opt['sb'])
                    if rms_yn==0: rms_yn==1
                    sf['sb'] = sb
                    sf['rms'] = rms_y
                    sf['rms_nrm'] = rms_y/rms_yn
                copa['data'][ii][i][fld][j][k]['std'] = sf
    return copa

# spectral balance SPLH-SPL
# IN:
#   y: raw signal
#   ype: pre-emphasized signal
# OUT:
#   sb: SPLH-SPL
# see https://de.wikipedia.org/wiki/Schalldruckpegel
def splh_spl_deprec(y,ype):

    p_ref = 2*10**(-5)
    # SPL (p_eff is rmsd)
    p = 20*math.log(myl.rmsd(y)/p_ref,10)
    pe = 20*math.log(myl.rmsd(ype)/p_ref,10)
    if p <= 0 or pe <= 0:
        return np.nan
    sb = pe-p
    #print('pe',pe,'p',p,10,'sb',sb) #!pe
    #myl.stopgo() #!pe
    return sb

# calculates quotients and 2nd order shape coefs for f0/energy contours
# IN:
#   y: f0 or energy contout
#   opt: config['styl']
#   r: <{}> dict to be updated
# OUT:
#   r: output dict + keys
#     qi mean_init/mean_nonInit
#     qf mean_fin/mean_nonFin
#     qb mean_init/mean_fin
#     qm mean_max(initFin)/mean_nonMax
#     c0 offset
#     c1 slope
#     c2 shape
def styl_std_quot(y,opt,r={}):
    # backward compatibility
    if 'gnl' not in opt:
        opt['gnl'] = {'win':0.3}

    # init
    for x in ['qi','qf','qm','qb','c0','c1','c2']:
        r[x] = np.nan
    
    if len(y)==0:
        return r

    # normalize to [0 1]
    #y = cp.deepcopy(y)
    #y = myl.nrm_vec(y,{'mtd':'minmax','rng':[0,1]})

    # final idx in y
    yl = len(y)
    
    # window length (smpl)
    wl = min(yl,int(opt['fs']*opt['gnl']['win']))
    
    # initial and final segment
    y_ini = y[0:wl],
    y_nin = y[wl:yl]
    y_fin = y[yl-wl:yl]
    y_nfi = y[0:yl-wl]

    # robustness: non-empty slices
    if len(y_ini)==0:
        y_ini = [y[0]]
    if len(y_nin)==0:
        y_nin = [y[-1]]
    if len(y_fin)==0:
        y_fin = [y[-1]]
    if len(y_nfi)==0:
        y_nfi = [y[0]]

    # means
    mi = np.mean(y_ini)
    mni = np.mean(y_nin)
    mf = np.mean(y_fin)
    mnf = np.mean(y_nfi)

    # quotients
    if mni>0:
        r['qi'] = mi/mni
    if mnf>0:
        r['qf'] = mf/mnf
    if mf>0:
        r['qb'] = mi/mf

    # max quot
    if np.isnan(r['qi']):
        r['qm'] = r['qf']
    elif np.isnan(r['qf']):
        r['qm'] = r['qi']
    else:
        r['qm'] = max(r['qi'],r['qf'])

    # polyfit
    t = myl.nrm_vec(myl.idx_a(len(y)),{'mtd':'minmax','rng':[0,1]})
    c = styl_polyfit(t,y,2)
    r['c0'] = c[2]
    r['c1'] = c[1]
    r['c2'] = c[0]

    return r


# adds normalized feature values to std feat dict
# IN:
#    sf  - std feat dict
#    ys  - vector f0/energy in normalization window
#    opt - copa['config']['styl']
#    tn  - <[]> t_on, t_off of normalization interval
# OUT:
#    sf + '*_nrm' entries
def styl_std_nrm(sf,ys,opt,tn=[]):
    sfn = styl_std_feat(ys,opt,tn)
    nrm_add = {}
    for x in sf:
        if sfn[x]==0: sfn[x]=1
        nrm_add[x]=sf[x]/sfn[x]
    for x in nrm_add:
        sf["{}_nrm".format(x)] = nrm_add[x]
    return sf

# transforms time interval to index array
# IN:
#   iv: 2-element array, time interval
#   fs: sample rate
#   y: vector (needed, since upper bound sometimes >=len(y))
# OUT:
#   i: indices between on and offset defined by iv
def styl_yi(iv,fs,y=[]):
    j = myl.sec2idx(np.asarray(iv),fs)
    if len(y)>0:
        return myl.idx_seg(j[0],min(j[1],len(y)-1))
    else:
        return myl.idx_seg(j[0],j[1])
# returns y segment
# IN:
#   iv: 2-element array, time interval
#   fs: sample rate
#   y
# OUT:
#   ys: y segment  
def styl_ys(iv,fs,y):
    j = myl.sec2idx(np.asarray(iv),fs)
    return y[myl.idx_seg(j[0],min(j[1],len(y)-1))]

##########################################################
#### voice characteristics ###############################
##########################################################

# IN:
#   copa
#   logFile
# OUT:
#   + .data...[voice|voice_file]
#          .shim
#             .v: shimmer
#             .c: 3rd order polycoefs
#          .jit
#             .v: jitter (relative)
#             .v_abs: jitter (absolute)
#             .m: mean period length (in sec)
#             .c: 3rd order polycoefs
def styl_voice(copa,f_log_in=''):
    global f_log
    f_log = f_log_in
    fld = 'voice'
    if fld in copa['config']['styl']:
        opt = cp.deepcopy(copa['config']['styl'][fld])
    else:
        opt={}
    opt = myl.opt_default(opt,{'shim':{},'jit':{}})

    # over files
    for ii in myl.numkeys(copa['data']):
        copa = styl_voice_file(copa,ii,fld,opt)

    return copa

# styl_voice() for single file
def styl_voice_file(copa,ii,fld,opt):
    chan_i = myl.numkeys(copa['data'][ii])
    fp = copa['data'][ii][chan_i[0]]['fsys']['pulse']
    fa = copa['data'][ii][chan_i[0]]['fsys']['aud']
    f = "{}/{}.{}".format(fp['dir'],fp['stm'],fp['ext'])
    fw = "{}/{}.{}".format(fa['dir'],fa['stm'],fa['ext'])
    # pulses as 2-dim list, one column per channel
    p = myl.input_wrapper(f,'lol',{'colvec':True})
    # signal
    sig_all, fs_sig = sif.wavread(fw)
    opt['fs_sig'] = fs_sig
    # over channels
    for i in chan_i:
        # jitter and shimmer file wide
        fld_f = "{}_file".format(fld)

        # channel-related column
        if np.ndim(sig_all)>1:
            sig = sig_all[:,i]
        else:
            sig = sig_all
        
        copa['data'][ii][i][fld_f] = styl_voice_feat(p[:,i],sig,opt)

        # over tiers
        for j in myl.numkeys(copa['data'][ii][i][fld]):
            nk = myl.numkeys(copa['data'][ii][i][fld][j])
            # over segments
            for k in nk:
                # resp. segments in pulse sequence and signal
                lb = copa['data'][ii][i][fld][j][k]['t'][0]
                ub = copa['data'][ii][i][fld][j][k]['t'][1]
                pul_i = myl.intersect(myl.find(p[:,i],'>=',lb),
                                      myl.find(p[:,i],'<=',ub))
                voi = styl_voice_feat(p[pul_i,i],sig,opt)
                for x in voi:
                    copa['data'][ii][i][fld][j][k][x] = voi[x]

    return copa


# IN:
#   pul: pulse time stamps
#   sig: amplitude values
#   opt:
#     'fs_sig' required
#     'shim': <not yet used> {}
#     'jit':
#        't_max' max distance in sec of subsequent pulses to be part of
#                same voiced segment <0.02> (=50 Hz; praat: Period ceiling)
#        't_min' min distance in sec <0.0001> (Period floor)
#        'fac_max' factor of max adjacent period difference <1.3>
#                (Maximum period factor)
# OUT:
#   dict
#       'jit': see styl_voice_jit()
#       'shim': see styl_voice_shim()
def styl_voice_feat(pul,sig,opt):

    # remove -1 padding from input file
    pul = pul[myl.find(pul,'>',0)]
    opt = myl.opt_default(opt,{'shim':{},'jit':{}})
    opt['shim'] = myl.opt_default(opt['shim'],{'fs_sig': opt['fs_sig']})
    opt['jit'] = myl.opt_default(opt['jit'],{'fs_sig': opt['fs_sig'],
                                             't_max': 0.02, 't_min': 0.0001,
                                             'fac_max': 1.3})
    jit = styl_voice_jit(pul,opt['jit'])
    shim = styl_voice_shim(pul,sig,opt['shim'])
    return {'jit': jit, 'shim': shim}

# shimmer: average of abs diff between amplitudes of subsequent pulses
#          divided by mean amplitude
# IN:
#   pul: vector of pulse sequence
#   sig: signal vector
#   opt: option dict ('fs_sig': sample rate)
# OUT:
#   shim dict
#     v: shimmer
#     c: 3rd order polycoefs describing changes of shimmer over normalized time 
def styl_voice_shim(pul,sig,opt):
    # get amplitude values at pulse time stamps
    a = myl.ea()
    # time stamps
    t = myl.ea()
    for i in myl.idx(pul):
        j = myl.sec2idx(pul[i],opt['fs_sig'])
        #print(j,'->',len(sig)) #!v
        if j >= len(sig):
            break
        a = np.append(a,np.abs(sig[j]))
        t = np.append(t,pul[i])
    d = np.abs(np.diff(a))
    ma = np.mean(a)
    # shimmer
    s = np.mean(d)/ma
    
    #print(t)

    ## 3rd order polynomial fit through normalized amplitude diffs
    # time points between compared periods
    # normalized to [-1 1] by range of [pul[0] pul[-1]]
    if ma>0:
        d = d/ma
    x = myl.nrm_vec(t[1:len(t)], {'mtd': 'minmax', 'rng': [-1,1]})
    c = styl_polyfit(x,d,3)

    #print(s,c) #!v
    #myl.stopgo() #!v

    return {'v': s, 'c': c}

# IN:
#   pul: vector of pulse sequence
#   opt: options
# OUT:
#   jit dict
#     v: jitter (relative)
#     v_abs: jitter (absolute)
#     m: mean period length (in sec)
#     c: 3rd order polycoefs describing changes of jitter over normalized time 
def styl_voice_jit(pul,opt):
    # all periods
    ta = np.diff(pul)
    # abs diffs of adjacent periods
    d = myl.ea()
    # indices of these periods
    ui = myl.ea().astype(int)
    i = 0
    while i < len(ta)-1:
        ec = skip_period(ta,i,opt)
        # skip first segment
        if ec > 0:
            i += ec
            continue
        d = np.append(d,np.abs(ta[i]-ta[i+1]))
        ui = np.append(ui,i)
        #print(len(d),len(ui)) #!v
        #myl.stopgo()
        i+=1

    # not enough values
    if len(ui)==0:
        v = np.nan
        return {'v': v, 'v_abs': v, 'm': v,
                'c': np.array([v,v,v,v])}

    # all used periods
    periods = ta[ui]
    # mean period
    mp = np.mean(periods)
    # jitter absolute
    jit_abs = np.sum(d)/len(d)
    # jitter relative
    jit = jit_abs/mp

    ## 3rd order polynomial fit through all normalized period diffs
    # time points between compared periods
    # normalized to [-1 1] by range of [pul[0] pul[-1]]
    if mp>0:
        d = d/mp
    x = myl.nrm_vec(pul[ui+1], {'mtd': 'minmax', 'rng': [-1,1]})
    c = styl_polyfit(x,d,3)

    return {'v': jit, 'v_abs': jit_abs, 'm': mp, 'c': c}



# returns error code which is increment in jitter calculation
# 0: minden rendben
def skip_period(d,i,opt):
    # 1st period problem
    if d[i] < opt['t_min'] or d[i] > opt['t_max']:
        return 1
    # 2nd period problem
    if d[i+1] < opt['t_min'] or d[i+1] > opt['t_max']:
        return 2
    # transition problem
    if max(d[i],d[i+1]) > opt['fac_max']*min(d[i],d[i+1]):
        return 1
    return 0

##########################################################
#### global segments #####################################
##########################################################

# IN:
#   copa
# OUT:  (ii=fileIdx, i=channelIdx, j=segmentIdx)
#   +['data'][ii][i]['glob']['c']  coefs
#   +['clst']['glob']['c']     coefs
#                    ['ij']    link to [fileIdx, channelIdx, segmentIdx]
#   +['data'][ii][i]['glob'][j]['decl'][bl|ml|tl|rng]['r']   reset
#   +['data'][ii][i]['f0']['r']    f0 residual
def styl_glob(copa,f_log_in=''):
    global f_log
    f_log = f_log_in

    myl.check_var({'var':copa,'env':'copasul',
                   'spec':{'step':'styl_glob','dom':'glob'}})
    
    # re-empty (needed if styl is repatedly applied)
    copa['clst']['glob'] = {'c':[], 'ij':[]}

    opt = copa['config']['styl']['glob']
    reg = copa['config']['styl']['register']
    err_sum = 0
    N = 0
    # over files
    for ii in myl.numkeys(copa['data']):
        #print(ii) #!v
        copa,err_sum,N = styl_glob_file(copa,ii,opt,reg,err_sum,N)
    
    if N>0:
        copa['val']['styl']['glob']['err_prop'] = err_sum/N
    else:
        copa['val']['styl']['glob']['err_prop'] = np.nan
    return copa


def styl_glob_file(copa,ii,opt,reg,err_sum,N):
    myFs = copa['config']['fs']
    # over channels
    for i in myl.numkeys(copa['data'][ii]):
        t = copa['data'][ii][i]['f0']['t']
        y = copa['data'][ii][i]['f0']['y']
        # residual
        r = np.zeros(len(y))

        # [[bl ml tl]...] medians over complete F0 contour (same length as y) 
        med = styl_reg_med(y,opt)

        # over glob segments
        for j in myl.numkeys(copa['data'][ii][i]['glob']):
            gt = copa['data'][ii][i]['glob'][j]['t']
            # to for std_feat dur calculation; for interval tier input only
            # tor: time for declination rate calculation
            if len(copa['data'][ii][i]['glob'][j]['to'])>1:
                to = copa['data'][ii][i]['glob'][j]['to'][0:2]
                tor = to
            else:
                to = myl.ea()
                tor = gt[[0,1]]
            yi = styl_yi(gt,myFs,y)
            ys = y[yi]
            df = styl_decl_fit(ys,opt,med[yi,:],tor)
            copl.plot_main({'call':'browse','state':'online',
                            'fit':df,'type':'glob','set':'decl','y':ys,
                            'infx':"{}-{}-{}".format(ii,i,j)},copa['config'])
            # coefs + fitted line
            copa['data'][ii][i]['glob'][j]['decl'] = df
            # standard features
            copa['data'][ii][i]['glob'][j]['gnl'] = styl_std_feat(ys,opt,to)
            # coefs without intersection for clustering
            copa['clst']['glob']['c'] = myl.push(copa['clst']['glob']['c'], df[reg]['c'][1:len(df[reg]['c'])])
            # reference indices: [file, channel, segment]
            copa['clst']['glob']['ij'] = myl.push(copa['clst']['glob']['ij'],[ii,i,j])
            
            # bl|ml|tl|rng reset
            for x in myl.lists():
                if j==0:
                    # first regline value ok, since basevalue subtracted before
                    py = 0
                else:
                    py = copa['data'][ii][i]['glob'][j-1]['decl'][x]['y'][-1]
                    
                copa['data'][ii][i]['glob'][j]['decl'][x]['r'] = copa['data'][ii][i]['glob'][j]['decl'][x]['y'][0] - py

            # residual
            r[yi] = styl_residual(y[yi],df,reg)

            # error
            err_sum += df['err']
            N += 1
            
        # entire residual
        copa['data'][ii][i]['f0']['r'] = r


    return copa, err_sum, N

# standard f0 features mean, std 
# IN:
#   y f0 segment
#   opt
#   t = [] time on/offset (for original time values instead of sample counting)
# OUT:
#   v['m']   arithmetic mean
#    ['sd']  standard dev
#    ['med'] median
#    ['iqr'] inter quartile range
#    ['max'] max
#    ['min'] min
#    ['dur'] duration in sec
def styl_std_feat(y,opt,t=[]):

    if len(y)==0:
        y=np.asarray([0])

    if len(t)>0:
        d = t[1]-t[0]
    else:
        d = myl.smp2sec(len(y),opt['fs']) ##!!t

    return {'m':np.mean(y), 'sd':np.std(y), 'med':np.median(y),
            'iqr':np.percentile(y,75)-np.percentile(y,25),
            'max':np.max(y),'min':np.min(y),
            'dur':d}

#### discontinuity stylizaion
# IN:
#   t   time seq of whole file 
#   y   f0 seq of whole file
#   a   decl dict of first segment returned by styl_decl_fit()
#   b   decl dict of second segment
#   opt: copa['config']['styl']['bnd']
#   plotDict {'copa'-copa, 'infx' key} <{}>
#   med: [[bl ml tl]...] medians
# OUT:
#   bnd['p'] - pause length in sec
#      ['t_on'] - onset time of post-boundary segment (evtl. end of pause)
#      ['t_off'] - offset time of pre-boundary segment (evtl. start of pause)
#      [myRegister][myDiscontinuity]
#     myRegister :=
#       bl - baseline
#       ml - midline
#       tl - topline
#       rng - range
#     myDiscontinuity :=
#       r - reset
#       rms - root mean squared deviation over seg a.b
#       rms_pre - rms for seg a
#       rms_post - rms for seg b
def styl_discont(t,y,a,b,opt,plotDict={},med=myl.ea()):
    # pause length, time
    bnd = {'p':b['to'][0]-a['to'][-1],'t_off':a['to'][-1],'t_on':b['to'][0]}

    ## joint segment
    ta = a['t']
    ia = styl_yi(ta,opt['fs'])
    tb = b['t']
    ib = styl_yi(tb,opt['fs'])

    # robust y-lim of indices
    ia, ib = sdw_robust(ia,ib,len(y)-1)

    # removing overlap
    if ta[-1]==tb[0]: ia = ia[0:len(ia)-1]

    # f0 segment
    ys = np.concatenate((y[ia],y[ib]))

    # corresponding median segment (only if available)
    # use 2x myl.push to ensure 2-dim list
    if len(med)==len(y):
        meds = np.concatenate((myl.lol(med[ia,:]),myl.lol(med[ib,:])),0)
    else:
        meds = myl.ea()

    # decl fit of joint segment
    df = styl_decl_fit(ys,opt,meds)

    # discontinuities for all register representations
    for x in myl.lists():

        # concat without overlap
        if ta[-1]==tb[0]:
            ya = a['decl'][x]['y'][0:len(a['decl'][x]['y'])-1]
        else:
            ya = a['decl'][x]['y']
        yb = b['decl'][x]['y']
        
        # joint segments' portions for rmsd, corr (pre, post, total)
        zab = df[x]['y']
        za = df[x]['y'][0:len(ya)]
        zb = df[x]['y'][len(ya):len(df[x]['y'])]

        # reset
        bnd[x]={}
        bnd[x]['r']=b['decl'][x]['y'][0]-a['decl'][x]['y'][-1]

        yab = np.concatenate((ya,yb))
        ## hack on: length adjustment
        yab, zab = myl.hal(yab,zab)
        ya, za = myl.hal(ya,za)
        yb, zb = myl.hal(yb,zb)
        ## hack off
        bnd[x]['rms'] = myl.rmsd(yab,zab)
        bnd[x]['rms_pre'] = myl.rmsd(ya,za)
        bnd[x]['rms_post'] = myl.rmsd(yb,zb)

    ### plot
    if 'copa' in plotDict:
        sts = 1/plotDict['copa']['config']['fs']
        # pause zeros
        zz = np.asarray([])
        tz = ta[1]+sts
        while tz < tb[0]:
            zz = myl.push(zz,0)
            tz += sts
        yy = np.concatenate((y[ia],zz,y[ib]))
        tt = np.linspace(ta[0],tb[1],len(yy))
        yya = y[ia]
        yyb = y[ib]
        tta = np.linspace(ta[0],ta[1],len(yya))
        ttb = np.linspace(tb[0],tb[1],len(yyb))

        copl.plot_main({'call':'browse','state':'online','type':'complex','set':'bnd',
                        'fit':{'a':a['decl'],'b':b['decl'],'ab':df},
                        'y':{'a':yya,'b':yyb,'ab':yy},
                        't':{'a':tta,'b':ttb,'ab':tt},'infx':plotDict['infx']},
                       plotDict['copa']['config'])
        
    return bnd

# returns list of paths through x to field with NaN
# IN:
#   typ: subdict type
#   z:   copa-subdict
def styl_contains_nan(typ,z):
    nanl=[]
    if typ=='bnd':
        for x in myl.lists('register'):
            for y in myl.lists('bndfeat'):
                if np.isnan(z[x][y]):
                    nanl.append("{}.{}".format(x,y))
    return nanl

#### removal of global component from f0 ###########
# IN:
#   y
#   df  decl_fit dict
#   r   'bl'|'ml'|'tl'|'rng'|'none'
# OUT:
#   y-residual
def styl_residual(y,df,r):
    y = cp.deepcopy(y)
    # do nothing
    if r == 'none':
        return y
    # level subtraction of bl, ml (or tl)
    elif r != 'rng':
        y = y-df[r]['y']
    # range normalization
    else:
        opt={'mtd':'minmax', 'rng':[0,1]}
        for i in myl.idx(y):
            opt['min'] = df['bl']['y'][i]
            opt['max'] = df['tl']['y'][i]
            yo=y[i]
            y[i] = myl.nrm(y[i],opt)
    return y


#### fit register level and range ###########
# IN:
#   y 1-dim array f0
#   opt['decl_win'] window length for median calculation in sec
#      ['prct']['bl']  <10>  value range for median calc
#              ['tl']  <90>
#      ['nrm']['mtd']   normalization method
#             ['range']
#   med <[]> median values (in some conexts pre-calculated)
#   t   <[]> time [on off] not provided in discontinuity context
# OUT:
#   df['tn']   [normalizedTime]
#     [myRegister]['c']   base/mid/topline/range coefs
#                 ['y']   line
#                 ['rate'] rate (ST per sec, only if input t is not empty)
#                 ['m']   mean value of line (resp. line dist)
#        myRegister :=
#          'bl' - baseline
#          'ml' - midline
#          'tl' - topline
#          'rng' - range
#     ['err'] - 1 if any line pair crossing, else 0
def styl_decl_fit(y,opt,med=myl.ea(),t=myl.ea()):

    med = myl.lol(med)
    if len(med) != len(y):
        med = styl_reg_med(y,opt)

    # normalized time
    tn = myl.nrm_vec(myl.idx_a(len(y)),opt['nrm'])  #yw
    # fit midline
    mc = styl_polyfit(tn,med[:,1],1)
    mv = np.polyval(mc,tn)
    # interpolate over midline-crossing bl and tl medians
    med[:,0] = styl_ml_cross(med[:,0],mv,'bl')
    med[:,2] = styl_ml_cross(med[:,2],mv,'tl')
    # return dict
    df = {}
    for x in myl.lists():
        df[x] = {}
    df['tn'] = tn
    df['ml']['c'] = mc
    df['ml']['y'] = mv
    df['ml']['m'] = np.mean(df['ml']['y'])

    # get c for 'none' register (evtl needed for clustering if register='none')
    df['none'] = {'c': styl_polyfit(tn,y,1)}

    # fit base and topline
    df['bl']['c'] = styl_polyfit(tn,med[:,0],1)
    df['bl']['y'] = np.polyval(df['bl']['c'],tn)
    df['bl']['m'] = np.mean(df['bl']['y'])
    df['tl']['c'] = styl_polyfit(tn,med[:,2],1)
    df['tl']['y'] = np.polyval(df['tl']['c'],tn)
    df['tl']['m'] = np.mean(df['tl']['y'])
 
    # fit range
    df['rng']['c'] = styl_polyfit(tn,med[:,2]-med[:,0],1)
    df['rng']['y'] = np.polyval(df['rng']['c'],tn)
    df['rng']['m'] = max(0,np.mean(df['rng']['y']))

    # declination rates
    if len(t)==2:
        for x in ['bl','ml','tl','rng']:
            if len(df[x]['y'])==0: continue
            if t[1]<=t[0]:
                df[x]['rate'] = 0
            else:
                df[x]['rate'] = (df[x]['y'][-1]-df[x]['y'][0])/(t[1]-t[0])
                #print(t, df[x]['y'][-1], df[x]['y'][0],df[x]['rate']) #!r

    # erroneous line crossing
    if ((min(df['tl']['y']-df['ml']['y']) < 0) or
        (min(df['tl']['y']-df['bl']['y']) < 0) or
        (min(df['ml']['y']-df['bl']['y']) < 0)):
        df['err'] = 1
    else:
        df['err'] = 0

    return df

# returns base/mid/topline medians
# IN:
#   y: n x 1 f0 vector
#   opt: config['styl']['glob']
# OUT:
#   med: n x 3 median sequence array
def styl_reg_med(y,opt):
    # window [on off] on F0 indices
    dw = round(opt['decl_win']*opt['fs'])
    yw = myl.seq_windowing({'win':dw,'rng':[0,len(y)]})

    # median sequence for base/mid/topline
    # med [[med_bl med_ml med_tl]...]
    med = myl.ea()
    for i in range(len(yw)):
        ys = y[yw[i,0]:yw[i,1]]
        qb, qt = np.percentile(ys, [opt['prct']['bl'], opt['prct']['tl']])
        if len(ys)<=2:
            ybl = ys
            ytl = ys
        else:
            ybl = ys[myl.find(ys,'<=',qb)]
            ytl = ys[myl.find(ys,'>=',qt)]
        # fallbacks
        if len(ybl)==0:
            ybl = ys[myl.find(ys,'<=',np.percentile(ys, 50))]
        if len(ytl)==0:
            ytl = ys[myl.find(ys,'>=',np.percentile(ys, 50))]

        if len(ybl)>0: med_bl = np.median(ybl)
        else: med_bl = np.nan
        if len(ytl)>0: med_tl = np.median(ytl)
        else: med_tl = np.nan
        med_ml = np.median(ys)
        med = myl.push(med,[med_bl,med_ml,med_tl])

    # interpolate over nan
    for ii in ([0,2]):
        xi = myl.find(med[:,ii],'is','nan')
        if len(xi)>0:
            xp = myl.find(med[:,ii],'is','finite')
            yp = med[xp,ii]
            yi = np.interp(xi,xp,yp)
            med[xi,ii]=yi

    return myl.lol(med)

        

### prevent from declination line crossing
# interpolate over medians for topline/baseline fit
# that are below/above the already fitted midline
# IN:
#   l median sequence
#   ml fitted midline
#   typ: 'bl'|'tl' for base/topline
# OUT:
#   l median sequence with linear interpolations
def styl_ml_cross(l,ml,typ):
    if typ=='bl':
        xi = myl.find(ml-l,'<=',0)
    else:
        xi = myl.find(l-ml,'<=',0)
    if len(xi)>0:
        if typ=='bl':
            xp = myl.find(ml-l,'>',0)
        else:
            xp = myl.find(l-ml,'>',0)
        if len(xp)>0:
            yp = l[xp]
            yi = np.interp(xi,xp,yp)
            l[xi]=yi

    return l


#################################################
#### local segments #############################
#################################################

# IN:
#   copa
# OUT: (ii=fileIdx, i=channelIdx, j=segmentIdx)
#   + ['data'][ii][i]['loc'][j]['acc'][*], see styl_loc_fit()
#                              ['gnl'][*], see styl_std_feat()
#   + ['clst']['loc']['c']  coefs
#                    ['ij'] link to [fileIdx, channelIdx, segmentIdx]
def styl_loc(copa,f_log_in=''):
    global f_log
    f_log = f_log_in
    myl.check_var({'var':copa,'env':'copasul',
                   'spec':{'step':'styl','dom':'loc'}})
    opt = copa['config']['styl']['loc']
    # re-empty (needed if styl is repatedly applied)
    copa['clst']['loc'] = {'c':[], 'ij':[]}

    # rmse
    rms_sum = 0
    N = 0

    # over files
    for ii in myl.numkeys(copa['data']):
        #print(ii) #!v
        copa,rms_sum,N = styl_loc_file(copa,ii,opt,rms_sum,N)

    copa['val']['styl']['loc']['rms_mean'] = rms_sum/N
    return copa

def styl_loc_file(copa,ii,opt,rms_sum,N):
    myFs = copa['config']['fs']
    # over channels
    for i in myl.numkeys(copa['data'][ii]):
        t = copa['data'][ii][i]['f0']['t']
        # add residual vector if not provided by styl_glob()
        # (for register = 'none')
        if 'r' not in copa['data'][ii][i]['f0']:
            copa['data'][ii][i]['f0']['r'] = cp.deepcopy(copa['data'][ii][i]['f0']['y'])
        y = copa['data'][ii][i]['f0']['r']
        # over loc segments
        jj = myl.numkeys(copa['data'][ii][i]['loc'])
        for j in jj:
            lt = copa['data'][ii][i]['loc'][j]['t']
            if len(copa['data'][ii][i]['loc'][j]['to'])>1:
                to = copa['data'][ii][i]['loc'][j]['to'][0:2]
            else:
                to = myl.ea()
            tn = copa['data'][ii][i]['loc'][j]['tn']
            #print('t:',t)
            #print('lt:',lt[[0,1]])
            yi = styl_yi(lt[[0,1]],myFs,y)
            ys = y[yi]
            lf = styl_loc_fit(lt,ys,opt)
            copa['data'][ii][i]['loc'][j]['acc'] = lf
            sf = styl_std_feat(ys,opt,to)
            # + nrm window
            yin = styl_yi(copa['data'][ii][i]['loc'][j]['tn'],myFs,y)
            # add normalization _nrm
            sf = styl_std_nrm(sf,y[yin],opt,tn)
            copa['data'][ii][i]['loc'][j]['gnl'] = sf
            copa['clst']['loc']['c'] = myl.push(copa['clst']['loc']['c'], lf['c'])
            copa['clst']['loc']['ij'] = myl.push(copa['clst']['loc']['ij'],[ii,i,j])

            # online plot: polyfit
            copl.plot_main({'call':'browse','state':'online','fit':lf,'type':'loc','set':'acc',
                            'y':ys,'infx':"{}-{}-{}".format(ii,i,j)},copa['config'])
            # online plot: superpos (after having processed all locsegs in globseg)
            if ((j+1 not in jj) or (copa['data'][ii][i]['loc'][j+1]['ri']>copa['data'][ii][i]['loc'][j]['ri'])):
                ri = copa['data'][ii][i]['loc'][j]['ri']
                copl.plot_main({'call':'browse','state':'online','fit':copa,'type':'complex',
                                'set':'superpos','i':[ii,i,ri],'infx':"{}-{}-{}".format(ii,i,ri)},
                               copa['config'])

            # rmse
            rms_sum += myl.rmsd(lf['y'],ys)
            N += 1

    return copa, rms_sum, N
    

# polyfit of local segment
# IN:
#   t timeSeq
#   y f0Seq
#   opt
# OUT:
#   f['c']  polycoefs (descending)
#    ['tn'] normalized time
#    ['y']  stylized f0
#    ['rms'] root mean squared error between original and resynthesized contour
#           for each polynomial order till opt['ord']; descending as for ['c']
def styl_loc_fit(t,y,opt):
    tt = np.linspace(t[0],t[1],len(y))
    # time normalization with zero placement
    tn = myl.nrm_zero_set(tt, {'t0':myl.trunc2(t[2]), 'rng':opt['nrm']['rng']})
    f = {'tn':tn}
    f['c'] = styl_polyfit(tn,y,opt['ord'])
    f['y'] = np.polyval(f['c'],tn)

    # errors from 0..opt['ord'] stylization
    # (same order as in f['c'], i.e. descending)
    f['rms'] = [myl.rmsd(f['y'],y)]
    for o in np.linspace(opt['ord']-1,0,opt['ord']):
        c = styl_polyfit(tn,y,o)
        r = np.polyval(c,tn)
        f['rms'].append(myl.rmsd(r,y))
        
    return f

### extended local segment feature set
# decl and gestalt features
# IN:
#   copa
# OUT: (ii=fileIdx, i=channelIdx, j=segmentIdx)
#   +[i]['loc'][j]['decl'][*], see styl_decl_fit()
#                 ['gst'][*], see styl_gestalt()
def styl_loc_ext(copa,f_log_in=''):
    global f_log
    f_log = f_log_in

    myl.check_var({'var':copa,'env':'copasul',
                   'spec':{'step':'styl','dom':'loc_ext','req':False,
                           'dep':['loc','glob']}})

    gopt = copa['config']['styl']['glob']
    lopt = copa['config']['styl']['loc']

    # over files
    for ii in myl.numkeys(copa['data']):
        #print(ii) #!v
        copa = styl_loc_ext_file(copa,ii,gopt,lopt)

    return copa

def styl_loc_ext_file(copa,ii,gopt,lopt):

    myFs = copa['config']['fs']

    # over channels
    for i in myl.numkeys(copa['data'][ii]):
        t = copa['data'][ii][i]['f0']['t']
        y = copa['data'][ii][i]['f0']['y']

        # [[bl ml tl]...] medians over complete F0 contour (same length as y) 
        med = styl_reg_med(y,gopt)

        ## decl, gestalt
        # over glob segments
        for j in myl.numkeys(copa['data'][ii][i]['glob']):
            dfg = copa['data'][ii][i]['glob'][j]['decl']
            gto = copa['data'][ii][i]['glob'][j]['t']
            # on:0.01:off
            gt = np.arange(gto[0],gto[1]+0.01,0.01)
            # needed, since f0 time starts with 0.1
            if gt[0]==0: gt=gt[1:]
            lsi = copa['data'][ii][i]['glob'][j]['ri']
            # over contained loc segments
            for k in lsi:
                # t and y of local segment
                lt = copa['data'][ii][i]['loc'][k]['t']
                # indices in entire utterance
                yi = styl_yi(lt[[0,1]],myFs)

                # orig time for rate calculation
                if len(copa['data'][ii][i]['loc'][k]['to'])>1:
                    to = copa['data'][ii][i]['loc'][k]['to'][0:2]
                else:
                    to = lt[[0,1]]

                # idx in copa['data'][ii][i]['glob'][j]['decl']['bl|ml|tl'][y]...
                ygi = myl.find_interval(gt,lt[[0,1]])  #! don't use styl_yi() !

                ## hack on: adjust lengths
                yi, ygi = myl.hal(yi,ygi)
                ## hack off

                # inform
                if len(yi)<4:
                    myLog("Warning from styl_loc_ext_file(): file num {}, channel num {}, local segment time interval {} {} too short for polynomial fitting.".format(ii,i,lt[0],lt[1]))

                ys = y[yi]
                dfl = styl_decl_fit(ys,gopt,med[yi,:],to) # !sic gopt
                copa['data'][ii][i]['loc'][k]['decl'] = dfl
                #print('t=', t)
                #print('gto=',gto)
                #print('gt=',gt)
                #print('lt=',lt)
                #print('yi=',yi)
                #print('ygi=',ygi)
                #print(len(gt))
                #print(len(ys))
                #print(len(copa['data'][ii][i]['loc'][k]['decl']['bl']['y']))
                #print(len(copa['data'][ii][i]['loc'][k]['decl']['ml']['y']))
                #print(len(copa['data'][ii][i]['loc'][k]['decl']['tl']['y']))
                #print(len(ygi))
                copa['data'][ii][i]['loc'][k]['gst'] = styl_gestalt({'dfl':dfl,'dfg':dfg,'idx':ygi,'opt':lopt,'y':ys})
                copl.plot_main({'call':'browse','state':'online','fit':dfl,'type':'loc',
                                'set':'decl','y':ys,'infx':"{}-{}-{}".format(ii,i,j)},
                               copa['config'])
    return copa

# measures deviation of locseg declination from corresponding stretch of globseg
# IN:
#   obj['dfl'] decl_fit dict of locseg
#      ['dfg'] decl_fit dict of globseg
#      ['idx'] y indices in globseg spanned by locseg 
#      ['y']   orig f0 values in locseg
#      ['opt']['nrm']['mtd'|'rng'] normalization specs
#             ['ord']   polyorder for residual
# OUT:
#   gst[myRegister]['rms'] RMSD between corresponding lines in globseg and locseg
#                  ['sd']  slope diff
#                  ['d_init'] y[0] diff
#                  ['d_fin']  y[-1] diff
#      ['residual'][myRegister]['c'] polycoefs of f0 residual
# myRegister := bl|ml|tl|rng
# residual: ml, bl, tl subtraction, pointwise rng [0 1] normalization
# REMARKS:
#   - all diffs: locseg-globseg
#   - sd is taken from (y_off-y_on)/lng since poly coefs 
#     of globseg and locseg decl are derived from different
#     time normalizations. That is, slopediff actually is a rate difference!
#   - no alignment in residual stylization (i.e. no defined center := 0)
def styl_gestalt(obj):
    (dfl,dfg_in,idx,opt,y) = (obj['dfl'],obj['dfg'],obj['idx'],obj['opt'],obj['y'])
    dcl = {}

    #print('in sub')

    # preps for residual calculation
    dfg={}
    for x in myl.lists():
        #print(idx)
        #print(len(dfg_in[x]['y']))
        dfg[x]={}
        dfg[x]['y']=dfg_in[x]['y'][idx]
    l = len(idx)
    dcl['residual']={}
    # gestalt features
    for x in myl.lists():
        dcl[x]={}
        yl = dfl[x]['y']
        yg = dfg[x]['y']
        # rms
        dcl[x]['rms'] = myl.rmsd(yl,yg)
        # slope diff
        dcl[x]['sd'] = ((yl[-1]-yl[0])-(yg[-1]-yg[0]))/l
        # d_init, d_fin
        dcl[x]['d_init'] = yl[0]-yg[0]
        dcl[x]['d_fin'] = yl[-1]-yg[-1]
        # residual
        r = styl_residual(y,dfg,x)
        t = myl.nrm_vec(myl.idx_a(l),opt['nrm'])
        dcl['residual'][x]={}
        dcl['residual'][x]['c'] = styl_polyfit(t,r,opt['ord'])
    return dcl

# try-catch wrapper around polyfit
# returns zeros if something goes wrong
def styl_polyfit(x,y,o):
    try:
        c = np.polyfit(x,y,o)
    except:
        c = np.zeros(o+1)
    return c

###########################################
####### boundary stylization ##############
###########################################
# IN:
#   copa
# OUT: (ii=fileIdx, i=channelIdx, j=tierIdx, k=segmentIdx)
#   + ['data'][ii][i]['bnd'][j][k]['decl']; see styl_decl_fit()
#   + ['data'][ii][i]['bnd'][j][k][myWindowing]; see styl_discount()
#          for non-final segments only
# myWindowing :=
#   std - standard, i.e. neighboring segments
#   win - symmetric window around segment boundary
#         (using 'tn': norm window limited by current chunk, see pp_t2i())
#   trend - from file onset to boundary, and from boundary to file offset
#         (using 'tt': trend window [chunkOn timePoint chunkOff], see p_t2i())
def styl_bnd(copa,f_log_in=''):
    global f_log
    f_log = f_log_in

    myl.check_var({'var':copa,'env':'copasul',
                   'spec':{'step':'styl','dom':'bnd'}})

    navi = copa['config']['navigate']
    opt = copa['config']['styl']['bnd']

    # over files
    for ii in myl.numkeys(copa['data']):
        copa = styl_bnd_file(copa,ii,navi,opt)
    return copa

# called by styl_bnd for file-wise processing
# IN:
#   copa
#   ii: fileIdx
#   navi: navigation dict
#   opt: copa['config']['styl']['bnd']
# OUT:
#   copa
#     +['bnd'][j][k]['decl']
#                   ['std']
#                   ['win']
#                   ['trend']
#         j - tier idx, k - segment idx
#         bnd features refer to boundary that FOLLOWS the segment k
def styl_bnd_file(copa,ii,navi,opt):

    myFs = copa['config']['fs']

    # over channels
    for i in myl.numkeys(copa['data'][ii]):
        t = copa['data'][ii][i]['f0']['t']

        # which f0 contour to take (+/- residual)
        if opt['residual']:
            y = copa['data'][ii][i]['f0']['r']
        else:
            y = copa['data'][ii][i]['f0']['y']

        # [[bl ml tl]...] medians over complete F0 contour (same length as y) 
        med = styl_reg_med(y,opt)

        # over tiers
        for j in myl.numkeys(copa['data'][ii][i]['bnd']):
            # over segments
            nk = myl.numkeys(copa['data'][ii][i]['bnd'][j])
            for k in nk:
                gt = copa['data'][ii][i]['bnd'][j][k]['t']
                yi = styl_yi(gt,myFs,y)
                ys = y[yi]
                # decl fit
                df = styl_decl_fit(ys,opt,med[yi,:])
                copa['data'][ii][i]['bnd'][j][k]['decl'] = df
                if k<1: continue
                a = copa['data'][ii][i]['bnd'][j][k-1]
                b = copa['data'][ii][i]['bnd'][j][k]
                    
                # plotting dict
                po = {'copa':copa,'infx':"{}-{}-{}-{}-std".format(ii,i,copa['data'][ii][i]['bnd'][j][k]['tier'],k-1)}
                    
                # discont
                copa['data'][ii][i]['bnd'][j][k-1]['std'] = styl_discont(t,y,a,b,opt,po,med)
                ## alternative bnd windows
                # trend
                if navi['do_styl_bnd_trend'] == True:
                    #print(copa['data'][ii][i]['bnd'][j])
                    po['infx'] = "{}-{}-{}-{}-trend".format(ii,i,copa['data'][ii][i]['bnd'][j][k]['tier'],k-1)
                    copa['data'][ii][i]['bnd'][j][k-1]['trend'] = styl_discont_wrapper(copa['data'][ii][i]['bnd'][j][k]['tt'],
                                                                                       t,y,opt,po,med)
                    
                if navi['do_styl_bnd_win'] == True:
                    po['infx'] = "{}-{}-{}-{}-win".format(ii,i,copa['data'][ii][i]['bnd'][j][k]['tier'],k-1)
                    copa['data'][ii][i]['bnd'][j][k-1]['win'] = styl_discont_wrapper(copa['data'][ii][i]['bnd'][j][k]['tn'],
                                                                                     t,y,opt,po,med)
                    
    return copa

# wrapper around styl_discont() for different calls
# IN:
#   tw: time array [on joint1 (joint2) off]
#       if joint2 is missing, i.e. 3-element array: joint2=joint1
#       discont is calculated between [on joint1] and [joint2 off]
#   t: all time array
#   y: all f0 array
#   opt: copa['config']
#   po: plotting options
#   med: [[bl ml tl]...], same length as y (will be calculated in this function if not provided)
# OUT:
#   dict from styl_discont()
# REMARKS:
#    tw: ['bnd'][j]['tn'|'tt'] can be passed on as is ('win', 'trend' discont)
#        ['bnd'][j]['t'] + ['bnd'][j+1]['t'] needs to be concatenated ('std' discont)
def styl_discont_wrapper(tw,t,y,opt,po={},med=myl.ea()):
    
    myFs = opt['fs']

    # called with entire opt dict
    if 'styl' in opt:
        opt=cp.deepcopy(opt['styl']['bnd'])

    # f0 medians [[bl ml tl]...], same length as y
    med = myl.lol(med)
    if len(med) != len(y):
        med = styl_reg_med(y,opt)

    ### segments
    if len(tw)==4:
        on1, off1, on2, off2 = tw[0], tw[1], tw[2], tw[3]
    else:
        on1, off1, on2, off2 = tw[0], tw[1], tw[1], tw[2]
    i1 = styl_yi([on1,off1],myFs)
    i2 = styl_yi([on2,off2],myFs)
    # robust extension
    i1, i2 = sdw_robust(i1,i2,len(y)-1)
    ys1 = y[i1]
    ys2 = y[i2]
    med1 = med[i1,:]
    med2 = med[i2,:]

    ### decl fits
    a = {'decl':styl_decl_fit(ys1,opt,med1),'t':[on1,off1],'to':[on1,off1]}
    b = {'decl':styl_decl_fit(ys2,opt,med2),'t':[on2,off2],'to':[on2,off2]}

    ### styl discontinuity
    return styl_discont(t,y,a,b,opt,po,med)

# for discont styl, padds indices so that index arrays have minlength 2
# IN:
#   i1, i2: index arrays of adjacent segments
#   u: upper index limit (incl; lower is 0)
def sdw_robust(i1,i2,u):
    #print('A',i2)
    i1 = myl.find(i1,'<=',u)
    i2 = myl.find(i2,'<=',u)
    #print('B',i2)
    if len(i1)==0:
        i1 = np.asarray([max(0,i2[0]-2)])
    if len(i2)==0:
        i2 = np.asarray([i1[-1]])
    #print('C',i2)
    while len(i1) < 2 and i1[-1] < u:
        i1 = np.append(i1,i1[-1]+1)
    while len(i2) < 2 and i2[-1] < u: 
        i2 = np.append(i2,i2[-1]+1)
    #print('D',i2)
    return i1, i2

####### speech rhythm ###################

# IN:
#   copa
#   typ: 'en'|'f0'
# OUT:
#   +['rhy'], see output of styl_speech_rhythm()
def styl_rhy(copa,typ,f_log_in=''):
    global f_log
    f_log = f_log_in

    fld = "rhy_{}".format(typ)
    myl.check_var({'var':copa,'env':'copasul',
                   'spec':{'step':'styl','dom':fld}})
    opt = cp.deepcopy(copa['config']['styl'][fld])
    opt['type']=typ
    opt['plot'] = copa['config']['plot']
    opt['fsys'] = copa['config']['fsys']
    opt['navigate'] = copa['config']['navigate']

    # over files
    for ii in myl.numkeys(copa['data']):
        #print(ii) #!v
        copa = styl_rhy_file(copa,ii,fld,opt)

    return copa

# IN:
#   copa
#   ii fileIdx
#   fld in copa 'rhy_en|f0'
#   opt augmented deepcopy of copa[config]
def styl_rhy_file(copa,ii,fld,opt):

    # over channels
    for i in myl.numkeys(copa['data'][ii]):
        if opt['type']=='f0':
            t = copa['data'][ii][i]['f0']['t']
            y = copa['data'][ii][i]['f0']['y']
            myFs = copa['config']['fs']
        # wav file
        else:
            fp = copa['data'][ii][i]['fsys']['aud']
            f = "{}/{}.{}".format(fp['dir'],fp['stm'],fp['ext'])
            y, fs_sig = sif.wavread(f)

            if np.ndim(y)>1: y = y[:,i]
            opt['fs']=fs_sig
            t = myl.smp2sec(myl.idx_seg(1,len(y)),fs_sig,0)
            myFs = fs_sig

            # rescale to max to make different recordings comparable
            if opt['sig']['scale']:
                amax = max(abs(y))
                fac = max(1,(1/amax)-1)
                y *= fac

        # file wide
        r = copa['data'][ii][i]['rate']
        #print('r1',r) #!
        # for plotting
        opt['infx'] = "{}-{}-file".format(ii,i)
        copa['data'][ii][i]["{}_file".format(fld)] = styl_speech_rhythm(y,r,opt,copa['config'])
        # over tiers
        for j in myl.numkeys(copa['data'][ii][i][fld]):
            # over segments
            nk = myl.numkeys(copa['data'][ii][i][fld][j])
            for k in nk:
                gt = copa['data'][ii][i][fld][j][k]['t']

                #!u
                #print(copa['data'][ii][i]['fsys']['annot'])
                #print(i,j,gt)

                yi = styl_yi(gt,myFs,y) 
                ys = y[yi]
                r = copa['data'][ii][i][fld][j][k]['rate']
                # for plotting
                opt['infx'] = "{}-{}-{}-{}".format(ii,i,copa['data'][ii][i][fld][j][k]['tier'],k)
                copa['data'][ii][i][fld][j][k]['rhy'] = styl_speech_rhythm(ys,r,opt,copa['config'])
                #print(i,j,j,fld,copa['data'][ii][i][fld][j][k]['rhy']) #!
                #myl.stopgo() #!
    return copa

# DCT-related features on energy or f0 contours
# y - array, f0 or amplitude sequence (raw signal, NOT energy contour)
# r[myDomain] <{}> rate of myDomain (e.g. 'syl', 'ag', 'ip')]
# opt['type'] - signal type: 'f0' or 'en'
# opt['fs']  - sample frequency
# opt['sig']           - options to extract energy contour
#                        (relevant only for opt['type']=='en')
#           ['wintyp'] - <'none'>, any type supported by
#                        scipy.signal.get_window()
#           ['winparam'] - <''> additionally needed window parameters,
#                        scalar, string, list ...
#           ['sts']    - stepsize of moving window
#           ['win']    - window length
#    ['rhy']           - DCT options
#           ['wintyp'] - <'kaiser'>, any type supported by
#                        scipy.signal.get_window()
#           ['winparam'] - <1> additionally needed window parameters,
#                        scalar, string, list ..., depends on 'wintyp'
#           ['nsm']    - <3> number of spectral moments
#           ['rmo']    - skip first (lowest) cosine (=constant offset)
#                        in spectral moment calculation <1>|0
#           ['lb']     - lower cutoff frequency for coef truncation <0> 
#           ['ub']     - upper cutoff frequency (if 0, no cutoff) <0>
#                        Recommended e.g. for f0 DCT, so that only influence
#                        of events with <= 10Hz on f0 contour is considered)
#           ['rb'] - <1> frequency catch band to measure influence
#                    of rate of events in rate dict R in DCT
#                    e.g. r['syl']=4, opt['rb']=1
#                    'syl' influence on DCT: summed abs coefs of 3,4,5 Hz
# OUT:
#   rhy['c_orig'] all coefs
#      ['f_orig'] all freq
#      ['c'] coefs with freq between lb and ub
#      ['f'] freq between lb and ub
#      ['i'] indices of 'c' in 'c_orig'
#      ['sm'] ndarray spectral moments
#      ['m'] weighted coef mean
#      ['sd'] weighted coef std
#      ['cbin'] ndarray, summed abs coefs in freq bins between lb and ub
#      ['fbin'] ndarray, corresponding frequencies
#      ['wgt'][myDomain]['mae'] - mean absolute error between
#                                 IDCT of coefs around resp rate
#                                 and IDCT of coefs between 'lb' and 'ub'
#                       ['prop'] - proportion of coefs around
#                                 resp rate relative to coef sum
#                       ['rate'] - rate in analysed domain
#                       ['dgm'] dist to glob max in dct !
#                       ['dlm'] dist to loc max in dct !
#      ['dur'] - segment duration (in sec) 
#      ['f_max'] - freq of max amplitude
#      ['n_peak'] - number of peaks in DCT spectrum
def styl_speech_rhythm(y,r={},opt={},copaConfig={}):
    err=0
    dflt={}
    dflt['sig']= {'sts':0.01,'win':0.05,'wintyp':'hamming','winparam':''}
    dflt['rhy'] = {'wintyp':'kaiser','winparam':1,'nsm':3,
                   'rmo':True,'lb':0,'ub':0,'wgt':{'rb':1}}
    for x in list(dflt.keys()):
        if type(dflt[x]) is not dict:
            if x not in opt:
                opt[x]=dflt[x]
        else:
            opt[x] = myl.opt_default(opt[x],dflt[x])

    for x in ['type','fs']:
        if x not in opt:
            myLog('speech_rhythm(): opt must contain {}'.format(x))
            err=1
        else:
            opt['sig'][x]=opt[x]
            opt['rhy'][x]=opt[x]
    if err==1: myLog('Fatal! Error in speech rhythm extraction',True)

    # adjust sample rate for type='en' (energy values per sec)
    if opt['type']=='en':
        opt['rhy']['fs'] = fs_sig2en(opt['sig']['sts'],opt['sig']['fs'])

    # energy contour
    if opt['type']=='en':
        y = sif.sig_energy(y,opt['sig'])
    
    #print(opt['fsys']['rhy_f0']) #!
    #myl.stopgo() #!

    # dct features
    rhy = sif.dct_wrapper(y,opt['rhy'])

    # number of local maxima
    rhy['n_peak'] = len(rhy['f_lmax'])

    # duration
    rhy['dur'] = myl.smp2sec(len(y),opt['rhy']['fs'])
    
    # domain weight features + ['wgt'][myDomain]['prop'|'mae'|'rate']
    rhy = rhy_sub(y,r,rhy,opt)

    #print(rhy) #!
    #myl.stopgo() #!

    copl.plot_main({'call':'browse','state':'online','fit':rhy,
                    'type':"rhy_{}".format(opt['type']),'set':'rhy',
                    'infx':opt['infx']},copaConfig)
    return rhy

# sample rate transformation from raw signal to energy contour
def fs_sig2en(sts,fs):
    return round(fs/(sts*fs))
    #sts = round(opt['sig']['sts']*opt['sig']['fs'])
    #opt['rhy']['fs'] = round(opt['sig']['fs']/sts)


# quantifying influence of events with specific rates on DCT
# IN:
#   y - ndarray contour
#   r - dict {myEventId:myRate}
#   rhy - output dict of dct_wrapper
#   opt - dict, see speech_rhythm()
# OUT:
#   rhy
#     +['wgt'][myDomain]['mae']
#                       ['prop']
#                       ['rate']
#                       ['dgm'] dist to glob max in dct
#                       ['dlm'] dist to loc max in dct
def rhy_sub(y,r,rhy,opt):
    opt_rhy = cp.deepcopy(opt['rhy'])

    # catch region around event rate
    rb = opt['rhy']['wgt']['rb']

    # sum(abs(coeff)) between opt['lb'] and opt['ub']
    ac = abs(rhy['c'])
    sac = sum(ac)

    # freqs of max values
    gpf, lpf = rhy['f_max'], rhy['f_lmax']

    # IDCT by coefficients with freq between lb and ub
    #   (needed for MAE and prop normalization. Otherwise
    #    weights depend on length of y)
    yr = sif.idct_bp(rhy['c_orig'],rhy['i'])

    rhy['wgt'] = {}

    # over rate keys
    for x in r:

        # distances to global and nearest local peak
        dg = r[x] - gpf
        dl = np.exp(10)
        for z in lpf:
            dd = r[x]-z
            if abs(dd) < dl:
                dl = dd

        # define catch band around respective event rate of x
        lb = r[x]-rb
        ub = r[x]+rb

        # 'prop': DCT coef abs ampl around rate
        if len(ac)==0:
            j = myl.ea()
            prp = 0
        else:
            j = myl.intersect(myl.find(rhy['f'],'>=',lb),
                              myl.find(rhy['f'],'<=',ub))
            prp = sum(ac[j])/sac

        # 'mae': mean abs error between rhy[c] IDCT and IDCT of
        #        coefs between event-dependent lb and ub
        yrx = sif.idct_bp(rhy['c_orig'],j)
        ae = myl.mae(yr,yrx)

        rhy['wgt'][x] = {'mae':ae, 'prop':prp, 'rate':r[x],
                         'dlm': dl, 'dgm': dg}

    return rhy


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
