#!/usr/bin/env python3

import scipy.io.wavfile as sio
import scipy.signal as sis
import numpy as np
import math
import matplotlib.pyplot as plt
import mylib as myl
import sys
import copy as cp
import scipy.fftpack as sf

# NOTE: int2float might be removed after scipy update (check defaults in myl.sig_preproc)

# read wav file
# IN:
#   fileName
# OUT:
#   signal ndarray
#   sampleRate
def wavread(f,opt={'do_preproc':1}):
    ## signal input
    fs, s_in = sio.read(f)
    # int -> float
    s = myl.wav_int2float(s_in)
    # preproc
    if opt['do_preproc']==1:
        s = myl.sig_preproc(s)

    return s, fs

# DCT
# IN:
#   y - 1D signal vector
#   opt        
#      ['fs'] - sample rate  
#      ['wintyp'] - <'kaiser'>, any type supported by
#                   scipy.signal.get_window()
#      ['winparam'] - <1> additionally needed window parameters,
#                   scalar, string, list ..., depends on 'wintyp'
#      ['nsm']    - <3> number of spectral moments
#      ['rmo']    - skip first (lowest) cosine (=constant offset)
#                   in spectral moment calculation <1>|0
#      ['lb']     - lower cutoff frequency for coef truncation <0> 
#      ['ub']     - upper cutoff frequency (if 0, no cutoff) <0>
#                   Recommended e.g. for f0 DCT, so that only influence
#                   of events with <= 10Hz on f0 contour is considered)
# OUT:
#   dct
#      ['c_orig'] all coefs
#      ['f_orig'] their frequencies
#      ['c'] coefs with freq between lb and ub
#      ['f'] their freqs
#      ['i'] their indices in c_orig
#      ['sm'] spectral moments based on c
#      ['opt'] input options
#      ['m'] y mean
#      ['sd'] y standard dev
#      ['cbin'] array of sum(abs(coef)) in frequency bins
#      ['fbin'] corresponding lower boundary freqs
def dct_wrapper(y,opt):
    dflt={'wintyp':'kaiser','winparam':1,'nsm':3,'rmo':True,
          'lb':0,'ub':0}
    opt = myl.opt_default(opt,dflt)

    # weight window
    w = sig_window(opt['wintyp'],len(y),opt['winparam'])
    y = y*w

    # centralize
    y = y-np.mean(y)

    # DCT coefs
    c = sf.dct(y,norm='ortho')

    # indices (starting with 0)
    ly = len(y)
    ci = myl.idx_a(ly)

    # corresponding cos frequencies
    f = ci+1 * (opt['fs']/(ly*2))

    # band pass truncation of coefs
    # indices of coefs with lb <= freq <= ub
    i = dct_trunc(f,ci,opt)
    
    # mean abs error from band-limited IDCT
    #mae = dct_mae(c,i,y)

    # remove constant offset with index 0
    # already removed by dct_trunc in case lb>0. Thus checked for i[0]==0
    #  (i[0] indeed represents constant offset; tested by
    #   cr = np.zeros(ly); cr[0]=c[0]; yr = sf.idct(cr); print(yr)
    if opt['rmo']==True and len(i)>1 and i[0]==0:
        j = i[1:len(i)]
    else:
        j = i

    if type(j) is not list: j = [j]

    # coefs and their frequencies between lb and ub
    #   (+ constant offset removed)
    fi = f[j]
    ci = c[j]

    # spectral moments
    if len(j)>0:
        sm = specmom(fi,ci,opt['nsm'])
    else:
        sm = np.zeros(opt['nsm'])
    
    # frequency bins
    fbin, cbin = dct_fbin(fi,ci,opt)

    # return
    return {'c_orig':c,'f_orig':f,'c':ci,'f':fi,'i':j,'sm':sm,'opt':opt,
            'm':np.mean(y),'sd':np.std(y),'cbin':cbin,'fbin':fbin}

# pre-emphasis
# s'[n] = s[n]-alpha*s[n-1]
def pre_emphasis(y,opt={'alpha':0.95}):
    # shifted signal
    z = cp.deepcopy(y)
    z = np.append([0],z[0:-1])
    #z = z[0:-1]
    return y-opt['alpha']*z

# frequency bins: symmetric 2-Hz windows around freq integers
# in bandpass overlapped by 1 Hz
# IN:
#   f - ndarray frequencies
#   c - ndarray coefs
#   opt['lb'] - lower and upper truncation freqs
#      ['ub']
# OUT:
#   fbin - ndarray, lower bnd of freq bins
#   cbin - ndarray, summed abs coef values in these bins
def dct_fbin(f,c,opt):
    fb = myl.idx_seg(math.floor(opt['lb']),math.ceil(opt['ub']))
    cbin = np.zeros(len(fb)-1);
    for j in myl.idx_a(len(fb)-1):
        k = myl.intersect(myl.find(f,'>=',fb[j]),
                          myl.find(f,'<=',fb[j+1]))
        cbin[j] = sum(abs(c[k]))

    fbin = fb[myl.idx_a(len(fb)-1)]
    return fbin, cbin

    

# spectral moments
# IN:
#   c - ndarray, coefficients
#   f - ndarray, related frequencies <1:len(c)>
#   n - number of spectral moments <3>
# OUT:
#   m - ndarray moments (increasing)
def specmom(c,f=[],n=3):
    if len(f)==0:
        f = myl.idx_a(len(c))+1
    c = abs(c)
    s = sum(c)
    k=0;
    m = np.asarray([]) 
    for i in myl.idx_seg(1,n):
        m = myl.push(m, sum(c*((f-k)**i))/s)
        k = m[-1]
    return m

# wrapper around IDCT
# IN:
#   c - coef vector derived by dct
#   i - indices of coefs to be taken for IDCT; if empty (default),
#       all coefs taken)
# OUT:
#   y - IDCT result
def idct_bp(c,i=myl.ea()):
    if len(i)==0:
        return sf.idct(c,norm='ortho')
    cr = np.zeros(len(c))
    cr[i]=c[i]
    return sf.idct(cr)

# mean abs error from IDCT
def dct_mae(c,i,y):    
    cr = np.zeros(len(c))
    cr[i]=c[i]
    yr = sf.idct(cr)
    return myl.mae(yr,y)
    
# indices to truncate DCT output to freq band
# IN:
#   f - ndarray, all frequencies
#   ci - all indices of coef ndarray
#   opt['lb'] - lower cutoff freq
#      ['ub'] - upper cutoff freq
# OUT:
#   i - ndarray, indices in F of elements to be kept
def dct_trunc(f,ci,opt):
    if opt['lb']>0:
        ihp = myl.find(f,'>=',opt['lb'])
    else:
        ihp = ci
    if opt['ub']>0:
        ilp = myl.find(f,'<=',opt['ub'])
    else:
        ilp = ci
    return myl.intersect(ihp,ilp)


# calculates energy contour from acoustic signal
#    do_preproc per default False. If not yet preprocessed by myl.sig_preproc()
#    set to True
# IN:
#   x ndarray signal
#   opt['fs']  - sample frequency
#      ['wintyp'] - <'none'>, any type supported by
#                   scipy.signal.get_window()
#      ['winparam'] - <''> additionally needed window parameters,
#                     scalar, string, list ...
#      ['sts']    - stepsize of moving window
#      ['win']    - window length
# OUT:
#   y ndarray energy contour
def sig_energy(x,opt):
    dflt={'wintyp':'hamming','winparam':'','sts':0.01,'win':0.05}
    opt = myl.opt_default(opt,dflt)
    # stepsize and winlength in samples
    sts = round(opt['sts']*opt['fs'])
    win = min([len(x)/2,round(opt['win']*opt['fs'])])
    # weighting window
    w = sig_window(opt['wintyp'],win,opt['winparam'])
    # energy values
    y = np.asarray([])
    for j in myl.idx_a(len(x)-win,sts):
        s = x[j:j+len(w)]*w
        y = myl.push(y,myl.rmsd(s))
    return y

# wrapper around windows
# IN:
#   typ: any type supported by scipy.signal.get_window()
#   lng: <1> length
#   par: <''> additional parameters as string, scalar, list etc
# OUT:
#   window array
def sig_window(typ,l=1,par=''):
    if typ=='none' or typ=='const':
        return np.ones(l)
    if ((type(par) is str) and (len(par) == 0)):
        return sis.get_window(typ,l)
    return sis.get_window((typ,par),l)
    
   
# pause detection
# IN:
#   s - mono signal
#   opt['fs']  - sample frequency
#      ['ons'] - idx onset <0> (to be added to time output)
#      ['flt']['f']     - filter options, boundary frequencies in Hz
#                         (2 values for btype 'band', else 1): <8000> (evtl. lowered by fu_filt())
#             ['btype'] - <'band'>|'high'|<'low'>
#             ['ord']   - butterworth order <5>
#             ['fs']    - (internally copied)
#      ['l']     - analysis window length (in sec)
#      ['l_ref'] - reference window length (in sec)
#      ['e_rel'] - min energy quotient analysisWindow/referenceWindow
#      ['fbnd']  - True|<False> assume pause at beginning and end of file
#      ['n']     - <-1> extract exactly n pauses (if > -1)
#      ['min_pau_l'] - min pause length <0.5> sec
#      ['min_chunk_l'] - min inter-pausal chunk length <0.2> sec
#      ['force_chunk'] - <False>, if True, pause-only is replaced by chunk-only
# OUT:
#    pau['tp'] 2-dim array of pause [on off] (in sec)
#       ['tpi'] 2-dim array of pause [on off] (indices in s = sampleIdx-1 !!)
#       ['tc'] 2-dim array of speech chunks [on off] (i.e. non-pause, in sec)
#       ['tci'] 2-dim array of speech chunks [on off] (indices)
#       ['e_ratio'] - energy ratios corresponding to pauses in ['tp'] (analysisWindow/referenceWindow)
def pau_detector(s,opt={}):
    if 'fs' not in opt:
        sys.exit('pau_detector: opt does not contain key fs.')
    dflt = {'e_rel':0.0767,'l':0.1524,'l_ref':5,'n':-1,'fbnd':False,'ons':0,'force_chunk':False,
            'min_pau_l':0.4,'min_chunk_l':0.2,'flt':{'btype':'low','f':np.asarray([8000]),'ord':5}}
    opt = myl.opt_default(opt,dflt)
    opt['flt']['fs'] = opt['fs']

    ## removing DC, low-pass filtering
    flt = fu_filt(s,opt['flt'])
    y = flt['y']

    ## pause detection for >=n pauses
    t, e_ratio = pau_detector_sub(y,opt)

    if len(t)>0:

        ## extending 1st and last pause to file boundaries
        if opt['fbnd']==True:
            t[0,0]=0
            t[-1,-1]=len(y)-1

        ## merging pauses across too short chunks
        ## merging chunks across too small pauses
        if (opt['min_pau_l']>0 or opt['min_chunk_l']>0):
            t, e_ratio = pau_detector_merge(t,e_ratio,opt)

        ## too many pauses?
        # -> subsequently remove the ones with highest e-ratio
        if (opt['n']>0 and len(t)>opt['n']):
            t, e_ratio = pau_detector_red(t,e_ratio,opt)
        
    ## speech chunks
    tc = pau2chunk(t,len(y))

    ## pause-only -> chunk-only
    if (opt['force_chunk']==True and len(tc)==0):
        tc = cp.deepcopy(t)
        t = np.asarray([])
        e_ratio = np.asarray([])

    ## add onset
    t = t+opt['ons']
    tc = tc+opt['ons']
    
    ## return dict
    ## incl fields with indices to seconds (index+1=sampleIndex)
    pau={'tpi':t, 'tci':tc, 'e_ratio': e_ratio}
    pau['tp'] = myl.idx2sec(t,opt['fs'])
    pau['tc'] = myl.idx2sec(tc,opt['fs'])

    #print(pau)

    return pau

# merging pauses across too short chunks
# merging chunks across too small pauses
# IN:
#   t [[on off]...] of pauses
#   e [e_rat ...]
# OUT:
#   t [[on off]...] merged
#   e [e_rat ...] merged (simply mean of merged segments taken)
def pau_detector_merge(t,e,opt):
    # min lengths in samples
    mpl = myl.sec2smp(opt['min_pau_l'],opt['fs'])
    mcl = myl.sec2smp(opt['min_chunk_l'],opt['fs'])
    #print("\n\nt:\n", t, "\nmpl:\n", mpl, "\nmcl:\n", mcl)
    ## merging pauses across short chunks
    tm = np.asarray([t[0,:]])
    em = np.asarray([e[0]])
    if (tm[0,0]<mcl): tm[0,0]=0
    for i in np.arange(1,len(t),1):
        if (t[i,0] - tm[-1,1] < mcl):
            tm[-1,1] = t[i,1]
            em[-1] = np.mean([em[-1],e[i]])
        else:
            tm = myl.push(tm,t[i,:])
            em = myl.push(em,e[i])
    ## merging chunks across short pauses
    tn = np.asarray([])
    en = np.asarray([])
    for i in myl.idx_a(len(tm)):
        if ((tm[i,1]-tm[i,0] >= mpl) or
            (opt['fbnd']==True and (i==0 or i==len(tm)-1))):
            tn = myl.push(tn,tm[i,:])
            en = myl.push(en,em[i])
    #print("tm:\n", tm, "\ntn:\n", tn)
    return tn, en


# pause to chunk intervals
# IN:
#    t [[on off]] of pause segments (indices in signal)
#    l length of signal vector
# OUT:
#    tc [[on off]] of speech chunks
def pau2chunk(t,l):
    if len(t)==0:
        return np.asarray([[0,l-1]])
    if t[0,0]>0:
        tc = np.asarray([[0,t[0,0]-1]])
    else:
        tc = np.asarray([])
    for i in np.arange(0,len(t)-1,1):
        if t[i,1] < t[i+1,0]-1:
            tc = myl.push(tc,[t[i,1]+1,t[i+1,0]-1])
    if t[-1,1]<l-1:
        tc = myl.push(tc,[t[-1,1]+1,l-1])
    return tc

# called by pau_detector
# IN:
#    as for pau_detector
# OUT:
#    t [on off]
#    e_ratio
def pau_detector_sub(y,opt):
    ## settings
    # reference window span
    rl = math.floor(opt['l_ref']*opt['fs'])
    # signal length
    ls = len(y)
    # min pause length
    ml = opt['l']*opt['fs']
    # global rmse and pause threshold
    e_rel = cp.deepcopy(opt['e_rel'])
    e_glob = myl.rmsd(y)
    t_glob = opt['e_rel']*e_glob
    # stepsize
    sts=max([1,math.floor(0.05*opt['fs'])])
    # energy calculation in analysis and reference windows
    wopt_en = {'win':ml,'rng':[0,ls]}
    wopt_ref = {'win':rl,'rng':[0,ls]}
    # loop until opt.n criterion is fulfilled
    # increasing energy threshold up to 1
    while e_rel < 1:
        # pause [on off], pause index
        t=np.asarray([])
        j=0
        # [e_y/e_rw] indices as in t
        e_ratio=np.asarray([])
        i_steps = np.arange(1,ls,sts)
        for i in i_steps:
            # window
            yi = myl.windowing_idx(i,wopt_en)
            e_y = myl.rmsd(y[yi])
            # reference window
            e_r = myl.rmsd(y[myl.windowing_idx(i,wopt_ref)])
            if (e_r <= t_glob):
                e_r = e_glob
            # if rmse in window below threshold
            if e_y <= e_r*opt['e_rel']:
                #print(i,e_y,e_r)
                if len(t)-1==j:
                    # values belong to already detected pause
                    if len(t)>0 and yi[0]<t[j,1]:
                        t[j,1]=yi[-1]
                        # evtl. needed to throw away superfluous
                        # pauses with high e_ratio
                        e_ratio[j]=np.mean([e_ratio[j],e_y/e_r])
                    else:
                        t = myl.push(t,[yi[0], yi[-1]])
                        e_ratio = myl.push(e_ratio,e_y/e_r)
                        j=j+1
                else:
                    t=myl.push(t,[yi[0], yi[-1]])
                    e_ratio = myl.push(e_ratio,e_y/e_r)
        # (more than) enough pauses detected?
        if len(t) >= opt['n']: break
        e_rel = e_rel+0.1

    return t, e_ratio

def pau_detector_red(t,e_ratio,opt):
    # keep boundary pauses
    if opt['fbnd']==True:
        n=opt['n']-2
        bp = [t[0,],t[-1,]]
        ii = np.arange(1,len(t)-1,1)
        t = t[ii,]
        e_ratio=e_ratio[ii]
    else:
        n=opt['n']
        bp=np.asarray([])

    # remove pause with highest e_ratio
    while len(t)>n:
        i = myl.find(e_ratio,'is','max')
        j = myl.find(np.arange(1,len(e_ratio),1),'!=',i[0])
        t = t[j,]
        e_ratio = e_ratio[j]

    # re-add boundary pauses if removed
    if opt['fbnd']==True:
        t=[bp[1,],t,bp[2,]]

    return t, e_ratio


# syllable nucleus detection
# IN:
#   s - mono signal
#   opt['fs'] - sample frequency
#      ['ons'] - onset in sec <0> (to be added to time output)
#      ['flt']['f']     - filter options, boundary frequencies in Hz
#                         (2 values for btype 'band', else 1): <np.asarray([200,4000])>
#             ['btype'] - <'band'>|'high'|'low'
#             ['ord']   - butterworth order <5>
#             ['fs']    - (internally copied)
#      ['l']     - analysis window length
#      ['l_ref'] - reference window length
#      ['d_min'] - min distance between subsequent nuclei (in sec)
#      ['e_min'] - min energy required for nucleus as a proportion to max energy <0.16>
#      ['e_rel'] - min energy quotient analysisWindow/referenceWindow
# OUT:
#   ncl['t'] - vector of syl ncl time stamps (in sec)
#      ['ti'] - corresponding vector idx in s
#      ['e_ratio'] - corresponding energy ratios (analysisWindow/referenceWindow)
#   bnd['t'] - vector of syl boundary time stamps (in sec)
#      ['ti'] - corresponding vector idx in s
#      ['e_ratio'] - corresponding energy ratios (analysisWindow/referenceWindow)

def syl_ncl(s,opt={}):
    ## settings
    if 'fs' not in opt:
        sys.exit('syl_ncl: opt does not contain key fs.')
    dflt = {'flt':{'f':np.asarray([200,4000]),'btype':'band','ord':5},
            'e_rel':1.07, 'l':0.08,'l_ref':0.15, 'd_min':0.05, 'e_min':0.16,
            'ons':0}
    opt = myl.opt_default(opt,dflt)
    opt['flt']['fs'] = opt['fs']
    
    if syl_ncl_trouble(s,opt):
        t = np.asarray([round(len(s)/2+opt['ons'])])
        ncl = {'ti':t, 't':myl.idx2sec(t,opt['fs']), 'e_ratio':[0]}
        bnd = cp.deepcopy(ncl)
        return ncl, bnd

    # reference window length
    rws = math.floor(opt['l_ref']*opt['fs'])
    # energy win length
    ml = math.floor(opt['l']*opt['fs'])
    # stepsize
    sts = max([1,math.floor(0.03*opt['fs'])])
    # minimum distance between subsequent nuclei
    # (num of items left and right potential locmax are compared with)
    md = math.floor(opt['d_min']*opt['fs']/sts)
    # bandpass filtering
    flt = fu_filt(s,opt['flt'])
    y = flt['y']
    # signal length
    ls = len(y)
    # minimum energy as proportion of maximum energy found
    e_y = np.asarray([])
    i_steps = np.arange(1,ls,sts)
    for i in i_steps:
        yi = np.arange(i,min([ls,i+ml-1]),1)
        e_y = myl.push(e_y,myl.rmsd(y[yi]))
    e_min = opt['e_min']*max(e_y)
    # output vector collecting nucleus sample indices
    t = np.asarray([])
    all_i = np.asarray([])
    all_e = np.asarray([])
    all_r = np.asarray([])

    # energy calculation in analysis and reference windows
    wopt_en = {'win':ml,'rng':[0,ls]}
    wopt_ref = {'win':rws,'rng':[0,ls]}
    for i in i_steps:
        yi = myl.windowing_idx(i,wopt_en)
        #yi = np.arange(yw[0],yw[1],1)
        ys = y[yi]
        e_y = myl.rmsd(ys)
        #print(ys,'->',e_y)
        ri = myl.windowing_idx(i,wopt_ref)
        #ri = np.arange(rw[0],rw[1],1)
        rs = y[ri]
        e_rw = myl.rmsd(rs)
        all_i = myl.push(all_i,i)
        all_e = myl.push(all_e,e_y)
        all_r = myl.push(all_r,e_rw)

    # local energy maxima
    idx = sis.argrelmax(all_e,order=md)

    # maxima related to syl ncl
    t = np.asarray([])
    e_ratio = np.asarray([])
    # idx in all_i
    ti = np.asarray([])
    for i in idx[0]:
        if ((all_e[i] >= all_r[i]*opt['e_rel']) and (all_e[i] > e_min)):
            t = myl.push(t,all_i[i])
            ti = myl.push(ti,i)
            e_ratio = myl.push(e_ratio, all_e[i]/all_r[i])

    # minima related to syl bnd
    tb = np.asarray([])
    e_ratio_b = np.asarray([])
    if len(t)>1:
        for i in range(len(ti)-1):
            j = myl.idx_seg(ti[i],ti[i+1])
            j_min = myl.find(all_e[j],'is','min')
            if len(j_min)==0: j_min=[0]
            # bnd idx
            bj = j[0]+j_min[0]
            tb = myl.push(tb,all_i[bj])
            e_ratio_b = myl.push(e_ratio_b, all_e[bj]/all_r[bj])

    # add onset
    t = t+opt['ons']
    tb = tb+opt['ons']

    # output dict,
    # incl idx to seconds
    ncl = {'ti':t, 't':myl.idx2sec(t,opt['fs']), 'e_ratio':e_ratio}
    bnd = {'ti':tb, 't':myl.idx2sec(tb,opt['fs']), 'e_ratio':e_ratio_b}
    #print(ncl['t'], e_ratio)

    return ncl, bnd

def syl_ncl_trouble(s,opt):
    if len(s)/opt['fs'] < 0.1:
        return True
    return False

# wrapper around Butter filter
# IN:
#   1-dim vector
#   opt['fs'] - sample rate
#      ['f']  - scalar (high/low) or 2-element vector (band) of boundary freqs
#      ['order'] - order
#      ['btype'] - band|low|high
# OUT:
#   flt['y'] - filtered signal
#      ['b'] - coefs
#      ['a']
def fu_filt(y,opt):
    # check f<fs/2
    if (opt['btype'] == 'low' and opt['f']>=opt['fs']/2):
        opt['f']=opt['fs']/2-100
    elif (opt['btype'] == 'band' and opt['f'][1]>=opt['fs']/2):
        opt['f'][1]=opt['fs']/2-100
    fn = opt['f']/(opt['fs']/2)
    b, a = sis.butter(opt['ord'], fn, btype=opt['btype'])
    #yf = sis.lfilter(b,a,y) # forward, shifts maxima
    yf = sis.filtfilt(b,a,y)
    return {'y':yf,'b':b,'a':a}

#### discontinuity analysis
# measures delta and linear fit discontinuities between 
# adjacent array elements in terms of:
#  - delta  
#  - reset of regression lines
#  - root mean squared deviation between overall regression line and
#       -- preceding segment's regression line
#       -- following segment's regression line
# IN:
#   x: nx2 array [[time val] ...]
#          OR
#      nx1 array [val ...]
#          for the latter indices are taken as time stamps
# OUT:
#   d: (n-1)x6 array [[residuum delta reset rms_total rms_pre rms_post] ...]
#         d[i,] refers to discontinuity between x[i-1,] and x[i,]
# Example:
# >> import numpy as np
# >> import discont as ds
# >> x = np.random.rand(20)
# >> d = ds.discont(x)

def discont(x):

    do_plot=False

    # time: first column or indices
    lx = len(x)
    if np.ndim(x)==1:
        t = np.arange(0,lx)
        x = np.asarray(x)
    else:
        t = x[:,0]
        x = x[:,1]

    # output
    d = np.asarray([])

    # overall linear regression
    c = myPolyfit(t,x,1)
    y = np.polyval(c,t)

    if do_plot:
        fig = plot_newfig()
        plt.plot(t,x,":b",t,y,"-r")
        plt.show()

    # residuums
    resid = x-y

    # deltas
    ds = np.diff(x)

    # linear fits
    for i in np.arange(1,lx):
        # preceding, following segment
        i1, i2 = np.arange(0,i), np.arange(i,lx)
        t1, t2, x1, x2 = t[i1], t[i2], x[i1], x[i2]
        # linear fit coefs
        c1 = myPolyfit(t1,x1,1)
        c2 = myPolyfit(t2,x2,1)
        # fit values
        y1 = np.polyval(c1,t1)
        y2 = np.polyval(c2,t2)
        # reset
        res = y2[0] - y1[-1]
        # RMSD: pre, post, all
        r1 = myl.rmsd(y[i1],y1)
        r2 = myl.rmsd(y[i2],y2)
        r12 = myl.rmsd(y,np.concatenate((y1,y2)))
        # append to output
        d = myl.push(d,[resid[i],ds[i-1],res,r1,r2,r12])

    return d

# robust wrapper around polyfit to
# capture too short inputs
# IN:
#   x
#   y
#   o: order <1>
# OUT:
#   c: coefs
def myPolyfit(x,y,o=1):
    if len(x)==0:
        return np.zeros(o+1)
    if len(x)<=o:
        return myl.push(np.zeros(o),np.mean(y))
    return np.polyfit(x,y,1)


#syl_ncl_wrapper()


# init new figure with onclick->next, keypress->exit
# OUT:
#   figureHandle
def plot_newfig():
    fig = plt.figure()
    cid1 = fig.canvas.mpl_connect('button_press_event', onclick_next)
    cid2 = fig.canvas.mpl_connect('key_press_event', onclick_exit)
    return fig

# klick on plot -> next one
def onclick_next(event):
    plt.close()

# press key -> exit
def onclick_exit(event):
    sys.exit()
