
import scipy.io.wavfile as sio
import scipy.signal as sis
from scipy import interpolate
import numpy as np
import math
import matplotlib.pyplot as plt
import copasul_utils as utils
import sys
import copy as cp
import re
import scipy.fftpack as sf



def wavread(f, opt={'do_preproc': True}):

    '''
    read wav file
    
    Args:
      fileName
    
    Returns:
      signal ndarray
      sampleRate
    '''

    # signal input
    fs, s_in = sio.read(f)
    # int -> float
    s = utils.wav_int2float(s_in)
    # preproc
    if opt['do_preproc']:
        s = utils.sig_preproc(s)

    return s, fs



def dct_wrapper(y, opt):

    '''
    DCT
    
    Args:
      y - 1D signal vector
      opt
         ['fs'] - sample rate
         ['wintyp'] - <'kaiser'>, any type supported by
                      scipy.signal.get_window()
         ['winparam'] - <1> additionally needed window parameters,
                      scalar, string, list ..., depends on 'wintyp'
         ['nsm']    - <3> number of spectral moments
         ['rmo']    - skip first (lowest) cosine (=constant offset)
                      in spectral moment calculation <1>|0
         ['lb']     - lower cutoff frequency for coef truncation <0>
         ['ub']     - upper cutoff frequency (if 0, no cutoff) <0>
                      Recommended e.g. for f0 DCT, so that only influence
                      of events with <= 10Hz on f0 contour is considered)
         ['peak_prct'] - <80> lower percentile threshold to be superseeded for
                      amplitude maxima in DCT spectrum
    
    Returns:
      dct
         ['c_orig'] all coefs
         ['f_orig'] their frequencies
         ['c'] coefs with freq between lb and ub
         ['f'] their freqs
         ['i'] their indices in c_orig
         ['sm'] spectral moments based on c
         ['opt'] input options
         ['m'] y mean
         ['sd'] y standard dev
         ['cbin'] array of sum(abs(coef)) in frequency bins
         ['fbin'] corresponding lower boundary freqs
         ['f_max'] frequency of global amplitude maximum
         ['f_lmax'] frequencies of local maxima (array of minlen 1)
         ['c_cog'] the coef amplitude of the cog freq (sm[0])
    PROBLEMS:
      - if segment is too short (< 5 samples) lowest freqs associated to
        DCT components are too high for ub, that is dct_trunc() returns
        empty array.
        -> np.nan assigned to respective variables
    '''

    dflt = {'wintyp': 'kaiser', 'winparam': 1, 'nsm': 3, 'rmo': True,
            'lb': 0, 'ub': 0, 'peak_prct': 80}
    opt = utils.opt_default(opt, dflt)

    # weight window
    w = sig_window(opt['wintyp'], len(y), opt['winparam'])
    y = y*w


    # print(1,len(y))
    # centralize
    y = y-np.mean(y)


    # print(2,len(y))
    # DCT coefs
    c = sf.dct(y, norm='ortho')


    # print(3,len(c))
    # indices (starting with 0)
    ly = len(y)
    ci = utils.idx_a(ly)

    # corresponding cos frequencies
    f = ci+1 * (opt['fs']/(ly*2))

    # band pass truncation of coefs
    # indices of coefs with lb <= freq <= ub
    i = dct_trunc(f, ci, opt)


    # print('f ci i',f,ci,i)
    # analysis segment too short -> DCT freqs above ub
    if len(i) == 0:
        sm = utils.ea()
        while len(sm) <= opt['nsm']:
            sm = np.append(sm, np.nan)
        return {'c_orig': c, 'f_orig': f, 'c': utils.ea(), 'f': utils.ea(), 'i': [], 'sm': sm, 'opt': opt,
                'm': np.nan, 'sd': np.nan, 'cbin': utils.ea(), 'fbin': utils.ea(),
                'f_max': np.nan, 'f_lmax': utils.ea(), 'c_cog': np.nan}


    # mean abs error from band-limited IDCT
    # mae = dct_mae(c,i,y)
    # remove constant offset with index 0
    # already removed by dct_trunc in case lb>0. Thus checked for i[0]==0
    #  (i[0] indeed represents constant offset; tested by
    #   cr = np.zeros(ly); cr[0]=c[0]; yr = sf.idct(cr); print(yr)
    if opt['rmo'] == True and len(i) > 1 and i[0] == 0:
        j = i[1:len(i)]
    else:
        j = i

    if type(j) is not list:
        j = [j]

    # coefs and their frequencies between lb and ub
    #   (+ constant offset removed)
    fi = f[j]
    ci = c[j]

    # spectral moments
    if len(j) > 0:
        sm = specmom(ci, fi, opt['nsm'])
    else:
        sm = np.zeros(opt['nsm'])

    # frequency bins
    fbin, cbin = dct_fbin(fi, ci, opt)

    # frequencies of global and local maxima in DCT spectrum
    f_max, f_lmax, px = dct_peak(ci, fi, sm[0], opt)

    # return
    return {'c_orig': c, 'f_orig': f, 'c': ci, 'f': fi, 'i': j, 'sm': sm, 'opt': opt,
            'm': np.mean(y), 'sd': np.std(y), 'cbin': cbin, 'fbin': fbin,
            'f_max': f_max, 'f_lmax': f_lmax, 'c_cog': px}



def dct_peak(x, f, cog, opt):

    '''
    returns local and max peak frequencies
    
    Args:
     x: array of abs coef amplitudes
     f: corresponding frequencies
     cog: center of gravity
    
    Returns:
     f_gm: freq of global maximu
     f_lm: array of freq of local maxima
     px: threshold to be superseeded (derived from prct specs)
    '''


    x = abs(cp.deepcopy(x))

    # global maximum
    i = utils.find(x, 'is', 'max') # keep
    # i = np.argmax(x)
    if len(i) > 1:
        i = int(np.mean(i))

    f_gm = float(f[i])

    # local maxima
    # threshold to be superseeded
    px = dct_px(x, f, cog, opt)
    idx = np.where(x >= px)[0]
    # 2d array of neighboring+1 indices
    # e.g. [[0,1,2],[5,6],[9,10]]
    ii = []
    # min freq distance between maxima
    fd_min = 1
    for i in utils.idx(idx):
        if len(ii) == 0:
            ii.append([idx[i]])
        elif idx[i] > ii[-1][-1]+1:
            xi = x[ii[-1]]
            fi = f[ii[-1]]
            j = utils.find(xi, 'is', 'max') # keep
            # j = np.argmax(xi)
            if len(j) > 0 and f[idx[i]] > fi[j[0]]+fd_min:
                # print('->1')
                ii.append([idx[i]])
            else:
                # print('->2')
                ii[-1].append(idx[i])
            # utils.stopgo() #!c
        else:
            ii[-1].append(idx[i])

    # get index of x maximum within each subsegment
    # and return corresponding frequencies
    f_lm = []
    for si in ii:
        zi = utils.find(x[si], 'is', 'max') # keep
        # zi = np.argmax(x[si])
        if len(zi) > 1:
            zi = int(np.mean(zi))
        else:
            zi = zi[0]
        i = si[zi]
        if not np.isnan(i):
            f_lm.append(f[i])


    # print('px',px)
    # print('i',ii)
    # print('x',x)
    # print('f',f)
    # print('m',f_gm,f_lm)
    # utils.stopgo()
    return f_gm, f_lm, px



def dct_px(x, f, cog, opt):

    '''
    return center-of-gravity related amplitude
    
    Args:
       x: array of coefs
       f: corresponding freqs
       cog: center of gravity freq
       opt
    
    Returns:
       coef amplitude related to cog
    '''


    x = abs(cp.deepcopy(x))

    # cog outside freq range
    if cog <= f[0]:
        return x[0]
    elif cog >= f[-1]:
        return x[-1]
    # find f-indices adjacent to cog
    for i in range(len(f)-1):
        if f[i] == cog:
            return x[i]
        elif f[i+1] == cog:
            return x[i+1]
        elif f[i] < cog and f[i+1] > cog:
            # interpolate
            # xi = np.interp(cog,f[i:i+2],x[i:i+2])
            # print('cog:',cog,'xi',f[i:i+2],x[i:i+2],'->',xi)
            return np.interp(cog, f[i:i+2], x[i:i+2])
    return np.percentile(x, opt['peak_prct'])


def pre_emphasis(y, a=0.95, fs=-1, do_scale=False):

    '''
    pre-emphasis
     alpha > 1 (interpreted as lower cutoff freq)
        alpha <- exp(-2 pi alpha delta)
     s'[n] = s[n]-alpha*s[n-1]
    
    Args:
      signal
      alpha - s[n-1] weight <0.95>
      fs - sample rate <-1>
      do_scale - <FALSE> if TRUE than the pre-emphasized signal is scaled to
            same abs_mean value as original signal (in general pre-emphasis
            leads to overall energy loss)
    '''

    # determining alpha directly or from cutoff freq
    if a > 1:
        if fs <= 0:
            print('pre emphasis: alpha cannot be calculated deltaT. Set to 0.95')
            a = 0.95
        else:
            a = math.exp(-2*math.pi*a*1/fs)


    # print('alpha',a)
    # shifted signal
    ype = np.append(y[0], y[1:] - a * y[:-1])

    # scaling
    if do_scale:
        sf = np.mean(abs(y))/np.mean(abs(ype))
        ype *= sf


    # plot
    # ys = y[30000:40000]
    # ypes = ype[30000:40000]
    # t = np.linspace(0,len(ys),len(ys))
    # fig, spl = plt.subplots(2,1,squeeze=False)
    # cid1 = fig.canvas.mpl_connect('button_press_event', onclick_next)
    # cid2 = fig.canvas.mpl_connect('key_press_event', onclick_exit)
    # spl[0,0].plot(t,ys)
    # spl[1,0].plot(t,ypes)
    # plt.show()
    ##
    return ype



def dct_fbin(f, c, opt):

    '''
    frequency bins: symmetric 2-Hz windows around freq integers
    in bandpass overlapped by 1 Hz
    
    Args:
      f - ndarray frequencies
      c - ndarray coefs
      opt['lb'] - lower and upper truncation freqs
         ['ub']
    
    Returns:
      fbin - ndarray, lower bnd of freq bins
      cbin - ndarray, summed abs coef values in these bins
    '''

    fb = utils.idx_seg(math.floor(opt['lb']), math.ceil(opt['ub']))
    cbin = np.zeros(len(fb)-1)
    for j in utils.idx_a(len(fb)-1):
        k = np.where((f >= fb[j]) & (f <= fb[j+1]))[0]
        cbin[j] = sum(abs(c[k]))

    fbin = fb[utils.idx_a(len(fb)-1)]
    return fbin, cbin


def specmom(c, f=[], n=3):

    '''
    spectral moments
    
    Args:
      c - ndarray, coefficients
      f - ndarray, related frequencies <1:len(c)>
      n - number of spectral moments <3>
    
    Returns:
      m - ndarray moments (increasing)
    '''

    if len(f) == 0:
        f = utils.idx_a(len(c))+1
    c = abs(c)
    s = sum(c)
    k = 0
    m = np.asarray([])
    for i in utils.idx_seg(1, n):
        m = utils.push(m, sum(c*((f-k)**i))/s)
        k = m[-1]

    return m



def idct_bp(c, i=utils.ea()):

    '''
    wrapper around IDCT
    
    Args:
      c - coef vector derived by dct
      i - indices of coefs to be taken for IDCT; if empty (default),
          all coefs taken)
    
    Returns:
      y - IDCT result
    '''

    if len(i) == 0:
        return sf.idct(c, norm='ortho')
    cr = np.zeros(len(c))
    cr[i] = c[i]
    return sf.idct(cr)



def dct_mae(c, i, y):

    '''
    mean abs error from IDCT
    '''

    cr = np.zeros(len(c))
    cr[i] = c[i]
    yr = sf.idct(cr)
    return utils.mae(yr, y)



def dct_trunc(f, ci, opt):

    '''
    indices to truncate DCT output to freq band
    
    Args:
      f - ndarray, all frequencies
      ci - all indices of coef ndarray
      opt['lb'] - lower cutoff freq
         ['ub'] - upper cutoff freq
    
    Returns:
      i - ndarray, indices in F of elements to be kept
    '''

    if opt['lb'] > 0:
        ihp = np.where(f >= opt['lb'])[0]
    else:
        ihp = ci
    if opt['ub'] > 0:
        ilp = np.where(f <= opt['ub'])[0]
    else:
        ilp = ci
    return utils.intersect(ihp, ilp)



def wrapper_energy(f, opt={}, fs=-1):

    '''
    wrapper around wavread and energy calculation
    
    Args:
      f: wavFileName (any number of channels) or array containing
             the signal (any number of channels=columns)
      opt: energy extraction and postprocessing
           .win, .wintyp, .winparam: window parameters
           .sts: stepsize for energy contour
           .do_preproc: centralizing signal
           .do_out: remove outliers
           .do_interp: linear interpolation over silence
           .do_smooth: smoothing (median or savitzky golay)
           .out dict; see pp_outl()
           .smooth dict; see pp_smooth()
     fs: <-1> needed if f is array
    
    Returns:
      y: time + energy contour 2-dim np.array
        (1st column: time, other columns: energy)
    '''

    opt = utils.opt_default(opt, {'wintyp': 'hamming',
                                  'winparam': None,
                                  'sts': 0.01,
                                  'win': 0.05,
                                  'do_preproc': True,
                                  'do_out': False,
                                  'do_interp': False,
                                  'do_smooth': False,
                                  'out': {},
                                  'smooth': {}})
    opt['out'] = utils.opt_default(opt['out'], {'f': 3,
                                                'm': 'mean'})
    opt['smooth'] = utils.opt_default(opt['smooth'], {"mtd": "sgolay",
                                                      "win": 7,
                                                      "ord": 3})
    if type(f) is str:
        s, fs = wavread(f, opt)
    else:
        if fs < 0:
            sys.exit("array input requires sample rate fs. Exit.")
        s = f

    opt['fs'] = fs
    # convert to 2-dim array; each column represents a channel
    if np.ndim(s) == 1:
        s = np.expand_dims(s, axis=1)

    # output (.T-ed later, reserve first list for time)
    y = utils.ea()

    # over channels
    for i in np.arange(0, s.shape[1]):
        e = sig_energy(s[:, i], opt)
        # setting outlier to 0
        if opt['do_out']:
            e = pp_outl(e, opt['out'])
        # interpolation over 0
        if opt['do_interp']:
            e = pp_interp(e)
        # smoothing
        if opt['do_smooth']:
            e = pp_smooth(e, opt['smooth'])
        # <0 -> 0
        e[e < 0] = 0
        
        y = utils.push(y, e)

    # output
    if np.ndim(y) == 1:
        y = np.expand_dims(y, axis=1)
    else:
        y = y.T

    # concat time as 1st column
    sts = opt['sts']
    t = np.arange(0, sts*y.shape[0], sts)
    if len(t) != y.shape[0]:
        while len(t) > y.shape[0]:
            t = t[0:len(t)-1]
        while len(t) < y.shape[0]:
            t = np.append(t, t[-1]+sts)
    t = np.expand_dims(t, axis=1)
    y = np.concatenate((t, y), axis=1)

    return y



def pp_outl(y, opt):

    '''
    replacing outliers by 0
    '''

    if "m" not in opt:
        return y
    # ignore zeros
    opt['zi'] = True
    io = utils.outl_idx(y, opt)
    if np.size(io) > 0:
        y[io] = 0
    return y



def pp_interp(y, opt={}):

    '''
    interpolation over 0 (+constant extrapolation)
    '''

    xi = np.where(y == 0)[0]
    xp = np.where(y > 0)[0]    
    yp = y[xp]
    if "kind" in opt:
        f = interpolate.interp1d(xp, yp, kind=opt["kind"],
                                 fill_value=(yp[0], yp[-1]))
        yi = f(xi)
    else:
        yi = np.interp(xi, xp, yp)
    y[xi] = yi

    return y

def pp_smooth(y, opt):

    '''
    smoothing
    remark: savgol_filter() causes warning
    Using a non-tuple sequence for multidimensional indexing is deprecated
    will be out with scipy.signal 1.2.0
    (https://github.com/scipy/scipy/issues/9086)
    '''

    if opt['mtd'] == 'sgolay':
        if len(y) <= opt['win']:
            return y
        y = sis.savgol_filter(y, opt['win'], opt['ord'])
    elif opt['mtd'] == 'med':
        y = sis.medfilt(y, opt['win'])
    return y


def sig_energy(x, opt):

    '''
    calculates energy contour from acoustic signal
       do_preproc per default False. If not yet preprocessed by utils.sig_preproc()
       set to True
    
    Args:
      x ndarray signal
      opt['fs']  - sample frequency
         ['wintyp'] - <'hamming'>, any type supported by
                      scipy.signal.get_window()
         ['winparam'] - <''> additionally needed window parameters,
                        scalar, string, list ...
         ['sts']    - stepsize of moving window
         ['win']    - window length
    
    Returns:
      y ndarray energy contour
    '''

    dflt = {'wintyp': 'hamming', 'winparam': None, 'sts': 0.01, 'win': 0.05}
    opt = utils.opt_default(opt, dflt)
    # stepsize and winlength in samples
    sts = round(opt['sts']*opt['fs'])
    win = min([math.floor(len(x)/2), round(opt['win']*opt['fs'])])
    # weighting window
    w = sig_window(opt['wintyp'], win, opt['winparam'])
    # energy values
    y = np.asarray([])
    for j in utils.idx_a(len(x)-win, sts):
        s = x[j:j+len(w)]*w
        y = utils.push(y, utils.rmsd(s))
    return y



def sig_window(typ, l=1, par=None):

    '''
    wrapper around windows
    
    Args:
      typ: any type supported by scipy.signal.get_window()
      lng: <1> length
      par: None additional parameters as string, scalar, list etc
    
    Returns:
      window array
    '''

    if typ == 'none' or typ == 'const':
        return np.ones(l)
    if par is None:
        par = ""
    if ((type(par) is str) and (len(par) == 0)):
        return sis.get_window(typ, l)
    return sis.get_window((typ, par), l)


def pau_detector(s, opt={}):

    '''
    pause detection
    
    Args:
      s - mono signal
      opt['fs']  - sample frequency
         ['ons'] - idx onset <0> (to be added to time output)
         ['flt']['f']     - filter options, boundary frequencies in Hz
                            (2 values for btype 'band', else 1): <8000> (evtl. lowered by fu_filt())
                ['btype'] - <'band'>|'high'|<'low'>
                ['ord']   - butterworth order <5>
                ['fs']    - (internally copied)
         ['l']     - analysis window length (in sec)
         ['l_ref'] - reference window length (in sec)
         ['e_rel'] - min energy quotient analysisWindow/referenceWindow
         ['fbnd']  - True|<False> assume pause at beginning and end of file
         ['n']     - <-1> extract exactly n pauses (if > -1)
         ['min_pau_l'] - min pause length <0.5> sec
         ['min_chunk_l'] - min inter-pausal chunk length <0.2> sec
         ['force_chunk'] - <False>, if True, pause-only is replaced by chunk-only
         ['margin'] - <0> time to reduce pause on both sides (sec; if chunks need init and final silence)
    
    Returns:
       pau['tp'] 2-dim array of pause [on off] (in sec)
          ['tpi'] 2-dim array of pause [on off] (indices in s = sampleIdx-1 !!)
          ['tc'] 2-dim array of speech chunks [on off] (i.e. non-pause, in sec)
          ['tci'] 2-dim array of speech chunks [on off] (indices)
          ['e_ratio'] - energy ratios corresponding to pauses in ['tp'] (analysisWindow/referenceWindow)
    '''

    if 'fs' not in opt:
        sys.exit('pau_detector: opt does not contain key fs.')
    dflt = {'e_rel': 0.0767, 'l': 0.1524, 'l_ref': 5, 'n': -1, 'fbnd': False, 'ons': 0, 'force_chunk': False,
            'min_pau_l': 0.4, 'min_chunk_l': 0.2, 'margin': 0,
            'flt': {'btype': 'low', 'f': np.asarray([8000]), 'ord': 5}}
    opt = utils.opt_default(opt, dflt)
    opt['flt']['fs'] = opt['fs']

    # removing DC, low-pass filtering
    flt = fu_filt(s, opt['flt'])
    y = flt['y']

    # pause detection for >=n pauses
    t, e_ratio = pau_detector_sub(y, opt)

    if len(t) > 0:

        # extending 1st and last pause to file boundaries
        if opt['fbnd'] == True:
            t[0, 0] = 0
            t[-1, -1] = len(y)-1

        # merging pauses across too short chunks
        # merging chunks across too small pauses
        if (opt['min_pau_l'] > 0 or opt['min_chunk_l'] > 0):
            t, e_ratio = pau_detector_merge(t, e_ratio, opt)

        # too many pauses?
        # -> subsequently remove the ones with highest e-ratio
        if (opt['n'] > 0 and len(t) > opt['n']):
            t, e_ratio = pau_detector_red(t, e_ratio, opt)

    # speech chunks
    tc = pau2chunk(t, len(y))

    # pause-only -> chunk-only
    if (opt['force_chunk'] == True and len(tc) == 0):
        tc = cp.deepcopy(t)
        t = np.asarray([])
        e_ratio = np.asarray([])

    # add onset
    t = t+opt['ons']
    tc = tc+opt['ons']

    # return dict
    # incl fields with indices to seconds (index+1=sampleIndex)
    pau = {'tpi': t, 'tci': tc, 'e_ratio': e_ratio}
    pau['tp'] = utils.idx2sec(t, opt['fs'])
    pau['tc'] = utils.idx2sec(tc, opt['fs'])


    # print(pau)
    return pau



def pau_detector_merge(t, e, opt):

    '''
    merging pauses across too short chunks
    merging chunks across too small pauses
    
    Args:
      t [[on off]...] of pauses
      e [e_rat ...]
    
    Returns:
      t [[on off]...] merged
      e [e_rat ...] merged (simply mean of merged segments taken)
    '''

    # min pause and chunk length in samples
    mpl = utils.sec2smp(opt['min_pau_l'], opt['fs'])
    mcl = utils.sec2smp(opt['min_chunk_l'], opt['fs'])

    # merging chunks across short pauses
    tm = np.asarray([])
    em = np.asarray([])
    for i in utils.idx_a(len(t)):
        if ((t[i, 1]-t[i, 0] >= mpl) or
                (opt['fbnd'] == True and (i == 0 or i == len(t)-1))):
            tm = utils.push(tm, t[i, :])
            em = utils.push(em, e[i])

    # nothing done in previous step?
    if len(tm) == 0:
        tm = cp.deepcopy(t)
        em = cp.deepcopy(e)
    if len(tm) == 0:
        return t, e

    # merging pauses across short chunks
    tn = np.asarray([tm[0, :]])
    en = np.asarray([em[0]])
    if (tn[0, 0] < mcl):
        tn[0, 0] = 0
    for i in np.arange(1, len(tm), 1):
        if (tm[i, 0] - tn[-1, 1] < mcl):
            tn[-1, 1] = tm[i, 1]
            en[-1] = np.mean([en[-1], em[i]])
        else:
            tn = utils.push(tn, tm[i, :])
            en = utils.push(en, em[i])

    # print("t:\n", t, "\ntm:\n", tm, "\ntn:\n", tn) #!v
    return tn, en


def pau2chunk(t, l):

    '''
    pause to chunk intervals
    
    Args:
       t [[on off]] of pause segments (indices in signal)
       l length of signal vector
    
    Returns:
       tc [[on off]] of speech chunks
    '''

    if len(t) == 0:
        return np.asarray([[0, l-1]])
    if t[0, 0] > 0:
        tc = np.asarray([[0, t[0, 0]-1]])
    else:
        tc = np.asarray([])
    for i in np.arange(0, len(t)-1, 1):
        if t[i, 1] < t[i+1, 0]-1:
            tc = utils.push(tc, [t[i, 1]+1, t[i+1, 0]-1])
    if t[-1, 1] < l-1:
        tc = utils.push(tc, [t[-1, 1]+1, l-1])
    return tc



def pau_detector_sub(y, opt):

    '''
    called by pau_detector
    
    Args:
       as for pau_detector
    
    Returns:
       t [on off]
       e_ratio
    '''


    # settings
    # reference window span
    rl = math.floor(opt['l_ref']*opt['fs'])
    # signal length
    ls = len(y)
    # min pause length
    ml = opt['l']*opt['fs']
    # global rmse and pause threshold
    e_rel = cp.deepcopy(opt['e_rel'])

    # global rmse
    # as fallback in case reference window is likely to be pause
    # almost-zeros excluded (cf percentile) since otherwise pauses
    # show a too high influence, i.e. lower the reference too much
    # so that too few pauses detected
    # e_glob = utils.rmsd(y)
    ya = abs(y)
    qq = np.percentile(ya, [50])
    e_glob = utils.rmsd(ya[ya > qq[0]])

    t_glob = opt['e_rel']*e_glob

    # stepsize
    sts = max([1, math.floor(0.05*opt['fs'])])
    # energy calculation in analysis and reference windows
    wopt_en = {'win': ml, 'rng': [0, ls]}
    wopt_ref = {'win': rl, 'rng': [0, ls]}
    # loop until opt.n criterion is fulfilled
    # increasing energy threshold up to 1
    while e_rel < 1:
        # pause [on off], pause index
        t = np.asarray([])
        j = 0
        # [e_y/e_rw] indices as in t
        e_ratio = np.asarray([])
        i_steps = np.arange(1, ls, sts)
        for i in i_steps:
            # window
            yi = utils.windowing_idx(i, wopt_en)
            e_y = utils.rmsd(y[yi])
            # energy in reference window
            e_r = utils.rmsd(y[utils.windowing_idx(i, wopt_ref)])
            # take overall energy as reference if reference window is pause
            if (e_r <= t_glob):
                e_r = e_glob
            # if rmse in window below threshold
            if e_y <= e_r*e_rel:
                yis = yi[0]
                yie = yi[-1]
                if len(t)-1 == j:
                    # values belong to already detected pause
                    if len(t) > 0 and yis < t[j, 1]:
                        t[j, 1] = yie
                        # evtl. needed to throw away superfluous
                        # pauses with high e_ratio
                        e_ratio[j] = np.mean([e_ratio[j], e_y/e_r])
                    else:
                        t = utils.push(t, [yis, yie])
                        e_ratio = utils.push(e_ratio, e_y/e_r)
                        j = j+1
                else:
                    t = utils.push(t, [yis, yie])
                    e_ratio = utils.push(e_ratio, e_y/e_r)
        # (more than) enough pauses detected?
        if len(t) >= opt['n']:
            break
        e_rel = e_rel+0.1

    if opt['margin'] == 0 or len(t) == 0:
        return t, e_ratio

    # shorten pauses by margins
    mar = int(opt['margin']*opt['fs'])
    tm, erm = utils.ea(), utils.ea()
    for i in utils.idx_a(len(t)):
        # only slim non-init and -fin pauses
        if i > 0:
            ts = t[i, 0]+mar
        else:
            ts = t[i, 0]
        if i < len(t)-1:
            te = t[i, 1]-mar
        else:
            te = t[i, 1]

        # pause disappeared
        if te <= ts:
            # ... but needs to be kept
            if opt['n'] > 0:
                tm = utils.push(tm, [t[i, 0], t[i, 1]])
                erm = utils.push(erm, e_ratio[i])
            continue
        # pause still there
        tm = utils.push(tm, [ts, te])
        erm = utils.push(erm, e_ratio[i])

    return tm, erm


def pau_detector_red(t, e_ratio, opt):
    # keep boundary pauses
    if opt['fbnd'] == True:
        n = opt['n']-2
        # bp = [t[0,],t[-1,]]
        bp = np.concatenate((np.array([t[0,]]), np.array([t[-1,]])), axis=0)
        ii = np.arange(1, len(t)-1, 1)
        t = t[ii,]
        e_ratio = e_ratio[ii]
    else:
        n = opt['n']
        bp = np.asarray([])

    if n == 0:
        t = []

    # remove pause with highest e_ratio
    while len(t) > n:
        i = np.argmax(e_ratio)
        aran = np.arange(1, len(e_ratio), 1)
        j = np.where(aran != i)[0]
        
        t = t[j,]
        e_ratio = e_ratio[j]

    # re-add boundary pauses if removed
    if opt['fbnd'] == True:
        if len(t) == 0:
            t = np.concatenate(
                (np.array([bp[0,]]), np.array([bp[1,]])), axis=0)
        else:
            t = np.concatenate(
                (np.array([bp[0,]]), np.array([t]), np.array([bp[1,]])), axis=0)

    return t, e_ratio


def splh_spl(sig, fs, opt_in={}):

    '''
    spectral balance calculation according to Fant 2000
    
    Args:
      sig: signal (vowel segment)
      fs: sampe rate
      opt:
         'win': length of central window in ms <len(sig)>; -1 is same as len(sig)
         'ub': upper freq boundary in Hz <-1> default: no low-pass filtering
         'domain': <'freq'>|'time'; pre-emp in frequency (Fant) or time domain
         'alpha': <0.95> for time domain only y[n] = x[n]-alpha*x[n-1]
               if alpha>0 it is interpreted as lower freq threshold for pre-emp
    
    Returns:
      sb: spectral tilt
    '''

    opt = cp.deepcopy(opt_in)
    opt = utils.opt_default(opt, {'win': len(sig), 'f': -1, 'btype': 'none',
                                  'domain': 'freq', 'alpha': 0.95})


    # print(opt)
    # utils.stopgo()
    ## cut out center window ##################################
    ls = len(sig)
    if opt['win'] <= 0:
        opt['win'] = ls
    if opt['win'] < ls:
        wi = utils.windowing_idx(int(ls/2),
                                 {'rng': [0, ls],
                                  'win': int(opt['win']*fs)})
        y = sig[wi]
    else:
        y = cp.deepcopy(sig)

    if len(y) == 0:
        return np.nan

    # reference sound pressure level
    p_ref = pRef('spl')

    ## pre-emp in time domain ####################################
    if opt['domain'] == 'time':
        # low pass filtering
        if opt['btype'] != 'none':
            flt = fu_filt(y, {'fs': fs, 'f': opt['f'], 'ord': 6,
                              'btype': opt['btype']})
            y = flt['y']
        yp = pre_emphasis(y, opt['alpha'], fs, False)
        y_db = 20*np.log10(utils.rmsd(y)/p_ref)
        yp_db = 20*np.log10(utils.rmsd(yp)/p_ref)
        # print(yp_db - y_db)
        return yp_db - y_db


    ## pre-emp in frequency domain ##############################
    # according to Fant
    # actual length of cut signal
    n = len(y)
    # hamming windowing
    y *= np.hamming(n)
    # spectrum
    Y = np.fft.fft(y, n)
    N = int(len(Y)/2)
    # frequency components
    XN = np.fft.fftfreq(n, d=1/fs)
    X = XN[0:N]

    # same as X = np.linspace(0, fs/2, N, endpoint=True)
    # amplitudes
    # sqrt(Y.real**2 + Y.imag**2)
    # to be normalized:
    # *2 since only half of transform is used
    # /N since output needs to be normalized by number of samples
    # (tested on sinus, cf
    # http://www.cbcity.de/die-fft-mit-python-einfach-erklaert)
    a = 2*np.abs(Y[:N])/N
    # vowel-relevant upper frequency boundary
    if opt['btype'] != 'none':
        vi = fu_filt_freq(X, opt)
        if len(vi) > 0:
            X = X[vi]
            a = a[vi]
    # Fant preemphasis filter (Fant et al 2000, p10f eq 20)
    preemp = 10*np.log10((1+X**2/200**2)/(1+X**2/5000**2))
    ap = 10*np.log10(a)+preemp
    # retransform to absolute scale
    ap = 10**(ap/10)


    # corresponds to gain values in Fant 2000, p11
    # for i in utils.idx(a):
    #    print(X[i],preemp[i])
    # utils.stopgo()
    # get sound pressure level of both spectra
    #       as 20*log10(P_eff/P_ref)
    spl = 20*np.log10(utils.rmsd(a)/p_ref)
    splh = 20*np.log10(utils.rmsd(ap)/p_ref)


    # get energy level of both spectra
    # spl = 20*np.log10(utils.mse(a)/p_ref)
    # splh = 20*np.log10(utils.mse(ap)/p_ref)
    # spectral balance
    sb = splh-spl


    # print(spl,splh,sb)
    # utils.stopgo()

    # fig = plt.figure()
    # plt.plot(X,20*np.log10(a),'b')
    # plt.plot(X,20*np.log10(preemp),'g')
    # plt.plot(X,20*np.log10(ap),'r')
    # plt.show()
    return sb


def fu_filt_freq(X, opt):

    '''
    returns indices of freq in x fullfilling conditions in opt
    
    Args:
      X: freq array
      opt: 'btype' - 'none'|'low'|'high'|'band'|'stop'
           'f': 1 freq for low|high, 2 freq for band|stop
    
    Returns:
      i: indices in X fulfilling condition
    '''

    typ = opt['btype']
    f = opt['f']

    # all indices
    if typ == 'none':
        return utils.idx_a(len(X))

    # error handling
    if re.search('(band|stop)', typ) and (not utils.listType(f)):
        print('filter type requires frequency list. Done nothing.')
        return utils.idx_a(len(X))
    if re.search('(low|high)', typ) and utils.listType(f):
        print('filter type requires only 1 frequency value. Done nothing.')
        return utils.idx_a(len(X))

    if typ == 'low':
        return np.nonzero(X <= f)
    elif typ == 'high':
        return np.nonzero(X >= f)
    elif typ == 'band':
        i = set(np.nonzero(X >= f[0]))
        return np.sort(np.array(i.intersection(set(np.nonzero(X <= f[1])))))
    elif typ == 'stop':
        i = set(np.nonzero(X <= f[0]))
        return np.sort(np.array(i.union(set(np.nonzero(X >= f[1])))))

    return utils.idx_a(len(X))


def pRef(typ):

    '''
    returns reverence levels for typ
    
    Args:
      typ
       'spl': sound pressure level
       'i': intensity level
    
    Returns:
      corresponding reference level
    '''

    if typ == 'spl':
        return 2*10**(-5)
    return 10**(-12)



def syl_ncl(s, opt={}):

    '''
    syllable nucleus detection
    
    Args:
      s - mono signal
      opt['fs'] - sample frequency
         ['ons'] - onset in sec <0> (to be added to time output)
         ['flt']['f']     - filter options, boundary frequencies in Hz
                            (2 values for btype 'band', else 1): <np.asarray([200,4000])>
                ['btype'] - <'band'>|'high'|'low'
                ['ord']   - butterworth order <5>
                ['fs']    - (internally copied)
         ['l']     - analysis window length
         ['l_ref'] - reference window length
         ['d_min'] - min distance between subsequent nuclei (in sec)
         ['e_min'] - min energy required for nucleus as a proportion to max energy <0.16>
         ['e_rel'] - min energy quotient analysisWindow/referenceWindow
         ['e_val'] - quotient, how sagged the energy valley between two nucleus
                     candidates should be. Measured relative to the lower energy
                     candidate. The lower, the deeper the required valley between
                     two peaks. Meaningful range ]0, 1]. Recommended range:
                     [0.9 1[
         ['center'] - boolean; subtract mean energy
    
    Returns:
      ncl['t'] - vector of syl ncl time stamps (in sec)
         ['ti'] - corresponding vector idx in s
         ['e_ratio'] - corresponding energy ratios (analysisWindow/referenceWindow)
      bnd['t'] - vector of syl boundary time stamps (in sec)
         ['ti'] - corresponding vector idx in s
         ['e_ratio'] - corresponding energy ratios (analysisWindow/referenceWindow)
    '''

    # settings
    if 'fs' not in opt:
        sys.exit('syl_ncl: opt does not contain key fs.')
    dflt = {'flt': {'f': np.asarray([200, 4000]), 'btype': 'band', 'ord': 5},
            'e_rel': 1.05, 'l': 0.08, 'l_ref': 0.15, 'd_min': 0.12, 'e_min': 0.1,
            'ons': 0, 'e_val': 1, 'center': False}
    opt = utils.opt_default(opt, dflt)
    opt['flt']['fs'] = opt['fs']

    if syl_ncl_trouble(s, opt):
        t = np.asarray([round(len(s)/2+opt['ons'])])
        ncl = {'ti': t, 't': utils.idx2sec(t, opt['fs']), 'e_ratio': [0]}
        bnd = cp.deepcopy(ncl)
        return ncl, bnd

    # reference window length
    rws = math.floor(opt['l_ref']*opt['fs'])
    # energy win length
    ml = math.floor(opt['l']*opt['fs'])
    # stepsize
    sts = max([1, math.floor(0.03*opt['fs'])])
    # minimum distance between subsequent nuclei
    # (in indices)
    # md = math.floor(opt['d_min']*opt['fs']/sts)
    md = math.floor(opt['d_min']*opt['fs'])

    # bandpass filtering
    flt = fu_filt(s, opt['flt'])
    y = flt['y']
    # signal length
    ls = len(y)
    # minimum energy as proportion of maximum energy found
    e_y = np.asarray([])
    i_steps = np.arange(1, ls, sts)
    for i in i_steps:
        yi = np.arange(i, min([ls, i+ml-1]), 1)
        e_y = np.append(e_y, utils.rmsd(y[yi]))

    if bool(opt['center']):
        e_y -= np.mean(e_y)

    e_min = opt['e_min']*max(e_y)
    # output vector collecting nucleus sample indices
    t = np.asarray([])
    all_i = np.asarray([])
    all_e = np.asarray([])
    all_r = np.asarray([])

    # energy calculation in analysis and reference windows
    wopt_en = {'win': ml, 'rng': [0, ls]}
    wopt_ref = {'win': rws, 'rng': [0, ls]}
    for i in i_steps:
        yi = utils.windowing_idx(i, wopt_en)
        # yi = np.arange(yw[0],yw[1],1)
        ys = y[yi]
        e_y = utils.rmsd(ys)
        # print(ys,'->',e_y)
        ri = utils.windowing_idx(i, wopt_ref)
        # ri = np.arange(rw[0],rw[1],1)
        rs = y[ri]
        e_rw = utils.rmsd(rs)
        all_i = np.append(all_i, i)
        all_e = np.append(all_e, e_y)
        all_r = np.append(all_r, e_rw)

    # local energy maxima
    # (do not use min duration md for order option, since local
    #  maximum might be obscured already by energy increase
    #  towards neighboring peak further away than md, and not only by
    #  closer than md peaks)
    idx = sis.argrelmax(all_e, order=1)


    # plot_sylncl(all_e,idx) #!v
    # print(opt["ons"]/opt["fs"] + np.array(idx)*sts/opt["fs"]) #!v
    # utils.stopgo() #!v
    # maxima related to syl ncl
    # a) energy constraints
    # timestamps (idx)
    tx = np.asarray([])
    # energy ratios
    e_ratiox = np.asarray([])
    # idx in all_i
    tix = np.asarray([]).astype(int)
    for i in idx[0]:

        # valley between this and previous nucleus deep enough?
        if len(tix) > 0:
            ie = all_e[tix[-1]:i]
            if len(ie) < 3:
                continue
            valley = np.min(ie)
            nclmin = np.min([ie[0], all_e[i]])
            if valley >= opt['e_val'] * nclmin:
                # replace previous nucleus by current one
                if all_e[i] > ie[0]:  # !n
                    all_e[tix[-1]] = all_e[i]  # !n
                    tx[-1] = all_i[i]  # !n
                    tix[-1] = i  # !n
                    e_ratiox[-1] = all_e[i]/all_r[i]  # !n

                # print("valley constraint -- tx:", all_i[i]/opt["fs"], "nclmin:", nclmin, "valley:", valley, "ie0:", ie[0], "all_e:", all_e[i], "--> skip!") #!v
                continue

        if ((all_e[i] >= all_r[i]*opt['e_rel']) and (all_e[i] > e_min)):
            tx = np.append(tx, all_i[i])
            tix = np.append(tix, i)
            e_ratiox = np.append(e_ratiox, all_e[i]/all_r[i])

        # else: #!v
        #    print("min_en constraint -- tx:", all_i[i]/opt["fs"], "all_e:", all_e[i], "all_r:", all_r[i], "e_min:", e_min, "--> skip!") #!v

    # print(len(tx)) #!v
    if len(tx) == 0:
        dflt = {'ti': utils.ea(),
                't': utils.ea(),
                'e_ratio': utils.ea()}
        return dflt, dflt


    # plot_sylncl(all_e,tix) #!v
    # b) min duration constraints
    # init by first found ncl
    t = np.array([tx[0]])
    e_ratio = np.array([e_ratiox[0]])
    # idx in all_i
    ti = np.array([tix[0]]).astype(int)
    for i in range(1, len(tx)):
        # ncl too close
        if np.abs(tx[i]-t[-1]) < md:
            # current ncl with higher energy: replace last stored one
            if e_ratiox[i] > e_ratio[-1]:
                t[-1] = tx[i]
                ti[-1] = tix[i]
                e_ratio[-1] = e_ratiox[i]
        else:
            t = np.append(t, tx[i])
            ti = np.append(ti, tix[i])
            e_ratio = np.append(e_ratio, e_ratiox[i])


    # plot_sylncl(all_e,ti) #!v
    # minima related to syl bnd
    tb = np.asarray([])
    e_ratio_b = np.asarray([])
    if len(t) > 1:
        for i in range(len(ti)-1):
            j = utils.idx_seg(ti[i], ti[i+1])
            j_min = np.argmin(all_e[j])
            
            # bnd idx
            bj = j[0] + j_min
            tb = np.append(tb, all_i[bj])
            e_ratio_b = np.append(e_ratio_b, all_e[bj]/all_r[bj])

    # add onset
    t = t+opt['ons']
    tb = tb+opt['ons']

    # output dict,
    # incl idx to seconds
    ncl = {'ti': t, 't': utils.idx2sec(t, opt['fs']), 'e_ratio': e_ratio}
    bnd = {'ti': tb, 't': utils.idx2sec(tb, opt['fs']), 'e_ratio': e_ratio_b}

    # print(ncl['t'], e_ratio)
    return ncl, bnd


def syl_ncl_trouble(s, opt):
    if len(s)/opt['fs'] < 0.1:
        return True
    return False



def fu_filt(y, opt):

    '''
    wrapper around Butter filter
    
    Args:
      1-dim vector
      opt['fs'] - sample rate
         ['f']  - scalar (high/low) or 2-element vector (band) of boundary freqs
         ['order'] - order
         ['btype'] - band|low|high; all other values: signal returned as is
    
    Returns:
      flt['y'] - filtered signal
         ['b'] - coefs
         ['a']
    '''

    # do nothing
    if not re.search('^(high|low|band)$', opt['btype']):
        return {'y': y, 'b': utils.ea(), 'a': utils.ea()}
    # check f<fs/2
    if (opt['btype'] == 'low' and opt['f'] >= opt['fs']/2):
        opt['f'] = opt['fs']/2-100
    elif (opt['btype'] == 'band' and opt['f'][1] >= opt['fs']/2):
        opt['f'][1] = opt['fs']/2-100
    fn = opt['f']/(opt['fs']/2)
    b, a = sis.butter(opt['ord'], fn, btype=opt['btype'])
    yf = sis.filtfilt(b, a, y)
    return {'y': yf, 'b': b, 'a': a}




def discont(x, ts=[], opt={}):

    '''
    discontinuity measurement

    measures delta and linear fit discontinuities between
    adjacent array elements in terms of:
    - delta
    - reset of regression lines
    - root mean squared deviation between overall regression line and
      -- preceding segment's regression line
      -- following segment's regression line
      -- both, preceding and following, regression lines
    - extrapolation rmsd between following regression line
      and following regression line, extrapolated by regression
      on preceding segment

    Args:
    x: nx2 array [[time val] ...]
         OR
     nx1 array [val ...]
         for the latter indices are taken as time stamps
    ts: nx1 array [time ...] of time stamps (or indices for size(x)=nx1)
        at which to calculate discontinuity; if empty, discontinuity is
        calculated at each point in time. If size(x)=nx1 ts MUST contain
        indices
      nx2 array [[t_off t_on] ...] to additionally account for pauses
    opt: dict
        .win: <'glob'>|'loc' calculate discontinuity over entire sequence
                             or within window
        .l: <3> if win==loc, length of window in sec or idx
            (splitpoint - .l : splitpoint + .l)
        .do_plot: <0> plots orig contour and linear stylization
        .plot: <{}> dict with plotting options; cf. discont_seg()
    Returns:
    d dict
    (s1: pre-bnd segment [i-l,i[,
    s2: post-bnd segment [i,i+l]
    sc: joint segment [i-l,i+l])
      dlt: delta
      res: reset
      ry1: s1, rmsd between joint vs pre-bnd fit
      ry2: s2, rmsd between joint vs post-bnd fit
      ryc: sc, rmsd between joint vs pre+post-bnd fit
      ry2e: s2: rmsd between pre-bnd fit extrapolated to s2 and post-bnd fit
      rx1: s1, rmsd between joint fit and pre-boundary x-values
      rx2: s2, rmsd between joint fit and post-boundary x-values
      rxc: sc, rmsd between joint fit and pre+post-boundary x-values
      rr1: s1, ratio rmse(joint_fit)/rmse(pre-bnd_fit)
      rr2: s2, ratio rmse(joint_fit)/rmse(post-bnd_fit)
      rrc: sc, ratio rmse(joint_fit)/rmse(pre+post-bnd_fit)
      ra1: c1-rate s1
      ra2: c1-rate s2
      dlt_ra: ra2-ra1
      s1_c3: cubic fitting coefs of s1
      s1_c2
      s1_c1
      s1_c0
      s2_c3: cubic fitting coefs of s2
      s2_c2
      s2_c1
      s2_c0
      dlt_c3: s2_c3-s1_c3
      dlt_c2: s2_c2-s1_c2
      dlt_c1: s2_c1-s1_c1
      dlt_c0: s2_c0-s1_c0
      eucl_c: euclDist(s1_c*,s2_c*)
      corr_c: corr(s1_c*,s2_c*)
      v1: variance in s1
      v2: variance in s2
      vc: variance in sc
      vr: variance ratio (mean(v1,v2))/vc
      dlt_v: v2-v1
      m1: mean in s1
      m2: mean in s2
      dlt_m: m2-m1
      p: pause length (in sec or idx depending on numcol(x);
                       always 0, if t is empty or 1-dim)

       i in each list refers to discontinuity between x[i-1] and x[i]
       dimension of each list: if len(ts)==0: n-1 array (first x-element skipped)
                               else: mx6; m is number of ts-elements in range of x[:,0],
                                     resp. in index range of x[1:-1]

    REMARKS:
    for all variables but corr_c and vr higher values indicate higher discontinuity
    variables:
       x1: original f0 contour for s1
       x2: original f0 contour for s2
       xc: original f0 contour for sc
       y1: line fitted on segment a
       y2: line fitted on segment b
       yc: line fitted on segments a+b
       yc1: yc part for x1
       yc2: yc part for x2
       ye: x1/y1-fitted line for x2
       cu1: cubic fit coefs of time-nrmd s1
       cu2: cubic fit coefs of time-nrmd s2
       yu1: polyval(cu1)
       yu2: polyval(cu2); yu1 and yu2 are cut to same length
    '''


    # time: first column or indices
    if np.ndim(x) == 1:
        t = np.arange(0, len(x))
        x = np.asarray(x)
    else:
        t = x[:, 0]
        x = x[:, 1]

    # tsi: index pairs in x for which to derive discont values
    #      [[infimum supremum]...] s1 right-aligned to infimum, s2 left-aligne to supremum
    #      for 1-dim ts both values are adjacent [[i-1, i]...]
    # zp: zero pause True for 1-dim ts input, False for 2-dim
    tsi, zp = discont_tsi(t, ts)

    # opt init
    opt = utils.opt_default(opt, {'win': 'glob', 'l': 3, 'do_plot': False,
                                  'plot': {}})

    # output
    d = discont_init()

    # linear fits
    # over time stamp pairs
    for ii in tsi:

        # delta
        d['dlt'].append(x[ii[1]]-x[ii[0]])

        # segments (x, y values of pre-, post, joint segments)
        t1, t2, tc, x1, x2, xc, y1, y2, yc, yc1, yc2, ye, cu1, cu2, yu1, yu2 = discont_seg(
            t, x, ii, opt)
        d = discont_feat(d, t1, t2, tc, x1, x2, xc, y1, y2, yc,
                         yc1, yc2, ye, cu1, cu2, yu1, yu2, zp)

    # to np.array
    for x in d:
        d[x] = np.asarray(d[x])

    return d



def discont_init():

    '''
    init discont dict
    '''

    return {"dlt": [],
            "res": [],
            "ry1": [],
            "ry2": [],
            "ryc": [],
            "ry2e": [],
            "rx1": [],
            "rx2": [],
            "rxc": [],
            "rr1": [],
            "rr2": [],
            "rrc": [],
            "ra1": [],
            "ra2": [],
            "dlt_ra": [],
            "s1_c3": [],
            "s1_c2": [],
            "s1_c1": [],
            "s1_c0": [],
            "s2_c3": [],
            "s2_c2": [],
            "s2_c1": [],
            "s2_c0": [],
            "dlt_c3": [],
            "dlt_c2": [],
            "dlt_c1": [],
            "dlt_c0": [],
            "eucl_c": [],
            "corr_c": [],
            "eucl_y": [],
            "corr_y": [],
            "v1": [],
            "v2": [],
            "vc": [],
            "vr": [],
            "dlt_v": [],
            "m1": [],
            "m2": [],
            "dlt_m": [],
            "p": []}



def discont_seg(t, x, ii, opt):

    '''
    pre/post-boundary and joint segments
    '''

    # preceding, following segment indices
    i1, i2 = discont_idx(t, ii, opt)

    # print(ii,"\n-> ", i1,"\n-> ", i2) #!v
    # utils.stopgo() #!v
    t1, t2, x1, x2 = t[i1], t[i2], x[i1], x[i2]
    tc = np.concatenate((t1, t2))
    xc = np.concatenate((x1, x2))

    # normalized time (only needed for reported polycoefs, not
    # for output lines
    tn1 = utils.nrm_vec(t1, {'mtd': 'minmax',
                             'rng': [-1, 1]})
    tn2 = utils.nrm_vec(t2, {'mtd': 'minmax',
                             'rng': [-1, 1]})

    # linear fit coefs
    c1 = myPolyfit(t1, x1, 1)
    c2 = myPolyfit(t2, x2, 1)
    cc = myPolyfit(tc, xc, 1)

    # cubic fit coefs (for later shape comparison)
    cu1 = myPolyfit(tn1, x1, 3)
    cu2 = myPolyfit(tn2, x2, 3)
    yu1 = np.polyval(cu1, tn1)
    yu2 = np.polyval(cu2, tn2)

    # cut to same length (from boundary)
    ld = len(yu1)-len(yu2)
    if ld > 0:
        yu1 = yu1[ld:len(yu1)]
    elif ld < 0:
        yu2 = yu2[0:ld]
    # robust treatment
    while len(yu2) < len(yu1):
        yu2 = np.append(yu2, yu2[-1])
    while len(yu1) < len(yu2):
        yu1 = np.append(yu1, yu1[-1])

    # fit values
    y1 = np.polyval(c1, t1)
    y2 = np.polyval(c2, t2)
    yc = np.polyval(cc, tc)
    # distrib yc over t1 and t2
    yc1, yc2 = yc[0:len(y1)], yc[len(y1):len(yc)]
    # linear extrapolation
    ye = np.polyval(c1, t2)


    # legend_loc: 'upper left'
    # plotting linear fits
    # segment boundary
    xb = []
    xb.extend(yu1)
    xb.extend(yu2)
    xb.extend(ye)
    xb.extend(x1)
    xb.extend(x2)
    xb = np.asarray(xb)
    if opt['do_plot'] and len(xb) > 0:
        lw1, lw2 = 5, 3
        yb = [np.min(xb), np.max(xb)]
        tb = [t1[-1], t1[-1]]
        po = opt["plot"]
        po = utils.opt_default(po, {"legend_loc": "best",
                                    "fs_legend": 35,
                                    "fs": (20, 12),
                                    "fs_title": 40,
                                    "fs_ylab": 30,
                                    "fs_xlab": 30,
                                    "title": "",
                                    "xlab": "time",
                                    "ylab": ""})
        po["ls"] = {"o": "--k", "b": "-k", "s1": "-g", "s2": "-g",
                    "sc": "-r", "se": "-c"}
        po["lw"] = {"o": lw2, "b": lw2, "s1": lw1,
                    "s2": lw1, "sc": lw1, "se": lw2}
        po["legend_order"] = ["o", "b", "s1", "s2", "sc", "se"]
        po["legend_lab"] = {"o": "orig", "b": "bnd", "s1": "fit s1", "s2": "fit s2",
                            "sc": "fit joint", "se": "pred s2"}
        utils.myPlot({"o": tc, "b": tb, "s1": t1, "s2": t2, "sc": tc, "se": t2},
                     {"o": xc, "b": yb, "s1": y1, "s2": y2, "sc": yc, "se": ye},
                     po)
    return t1, t2, tc, x1, x2, xc, y1, y2, yc, yc1, yc2, ye, cu1, cu2, yu1, yu2


def discont_feat(d, t1, t2, tc, x1, x2, xc, y1, y2, yc, yc1, yc2, ye, cu1, cu2, yu1, yu2, zp):

    '''
    features
    '''

    # reset
    d["res"].append(y2[0]-y1[-1])
    # y-RMSD between regression lines: 1-pre, 2-post, c-all
    d["ry1"].append(utils.rmsd(yc1, y1))
    d["ry2"].append(utils.rmsd(yc2, y2))
    d["ryc"].append(utils.rmsd(yc, np.concatenate((y1, y2))))
    # extrapolation y-RMSD
    d["ry2e"].append(utils.rmsd(y2, ye))
    # xy-RMSD between regression lines and input values: 1-pre, 2-post, c-all
    rx1 = utils.rmsd(yc1, x1)
    rx2 = utils.rmsd(yc2, x2)
    rxc = utils.rmsd(yc, xc)
    d["rx1"].append(rx1)
    d["rx2"].append(rx2)
    d["rxc"].append(rxc)
    # xy-RMSD ratios of joint fit divided by single fits RMSD
    # (the higher, the more discontinuity)
    d["rr1"].append(utils.robust_div(rx1, utils.rmsd(y1, x1)))
    d["rr2"].append(utils.robust_div(rx2, utils.rmsd(y2, x2)))
    d["rrc"].append(utils.robust_div(
        rxc, utils.rmsd(np.concatenate((y1, y2)), xc)))
    # rates
    d["ra1"].append(drate(t1, y1))
    d["ra2"].append(drate(t2, y2))
    d["dlt_ra"].append(d["ra2"][-1]-d["ra1"][-1])
    # means
    d["m1"].append(np.mean(x1))
    d["m2"].append(np.mean(x2))
    d["dlt_m"].append(d["m2"][-1]-d["m1"][-1])
    # variances
    d["v1"].append(np.var(x1))
    d["v2"].append(np.var(x2))
    d["vc"].append(np.var(xc))
    d["vr"].append(np.mean([d["v1"][-1], d["v2"][-1]])/d["vc"][-1])
    d["dlt_v"].append(d["v2"][-1]-d["v1"][-1])
    # shapes
    d["s1_c3"].append(cu1[0])
    d["s1_c2"].append(cu1[1])
    d["s1_c1"].append(cu1[2])
    d["s1_c0"].append(cu1[3])
    d["s2_c3"].append(cu2[0])
    d["s2_c2"].append(cu2[1])
    d["s2_c1"].append(cu2[2])
    d["s2_c0"].append(cu2[3])
    d["eucl_c"].append(utils.dist_eucl(cu1, cu2))
    rr = np.corrcoef(cu1, cu2)
    d["corr_c"].append(rr[0, 1])
    d["dlt_c3"].append(d["s2_c3"][-1]-d["s1_c3"][-1])
    d["dlt_c2"].append(d["s2_c2"][-1]-d["s1_c2"][-1])
    d["dlt_c1"].append(d["s2_c1"][-1]-d["s1_c1"][-1])
    d["dlt_c0"].append(d["s2_c0"][-1]-d["s1_c0"][-1])
    d["eucl_y"].append(utils.dist_eucl(yu1, yu2))
    rry = np.corrcoef(yu1, yu2)
    d["corr_y"].append(rry[0, 1])
    # pause
    if zp:
        d["p"].append(0)
    else:
        d["p"].append(t2[0]-t1[-1])

    return d



def drate(t, y):

    '''
    returns declination rate of y over time t
    
    Args:
      t: time vector
      y: vector of same length
    
    Returns:
      r: change in y over time t
    '''

    if len(t) == 0 or len(y) == 0:
        return np.nan
    return (y[-1]-y[0])/(t[-1]/t[0])


def discont_tsi(t, ts):

    '''
    indices in t for which to derive discont values
    
    Args:
      t: all time stamps/indices
      ts: selected time stamps/indices, can be empty, 1-dim or 2-dim
    
    Returns:
      ii
        ==t-index pairs [[i-1, i]...] for i>=1, if ts empty
        ==index of [[infimum supremum]...] t-elements for ts stamps or intervals, else
      zp
        zero pause; True for 1-dim ts, False for 2-dim
    '''

    ii = []
    # return all index pairs [i-1, i]
    if len(ts) == 0:
        for i in np.arange(1, len(t)):
            ii = utils.push(ii, [i-1, i])
        return ii
    # zero pause
    if utils.of_list_type(ts[0]):
        zp = False
    else:
        zp = True
    # return selected index pairs
    for x in ts:
        # supremum and infimum
        if utils.of_list_type(x):
            xi, xs = x[0], x[1]
        else:
            xi, xs = x, x

        sup = np.where(t >= xs)[0]
        if xi == xs:
            inf = np.where(t < xi)[0]
        else:
            inf = np.where(t <= xi)[0]

        
        if len(sup) == 0 or len(inf) == 0 or sup[0] == 0 or inf[-1] == 0:
            continue
        ii.append([inf[-1], sup[0]])

    return ii, zp



def discont_idx(t, ii, opt):

    '''
    preceding, following segment indices around t[i]
    defined by opt[win|l]
    
    Args:
      t: 1- or 2-dim time array [timeStamp ...] or [[t_off t_on] ...], the latter
            accounting for pauses
      ii: current idx pair in t
      opt: cf discont
    
    Returns:
      i1, i2: pre/post boundary index arrays
    REMARK:
      i is part of i2
    '''

    lx = len(t)
    i, j = ii[0], ii[1]
    # glob: preceding, following segment from start/till end
    if opt['win'] == 'glob':
        return np.arange(0, ii[0]), np.arange(ii[1], lx)
    i1 = utils.find_interval(t, [t[i]-opt['l'], t[i]])
    i2 = utils.find_interval(t, [t[j], t[j]+opt['l']])
    return i1, i2



def discont_deprec(x):

    '''
    discontinuity analysis: some bugs, use discont() instead
    measures delta and linear fit discontinuities between
    adjacent array elements in terms of:
     - delta
     - reset of regression lines
     - root mean squared deviation between overall regression line and
          -- preceding segment's regression line
          -- following segment's regression line
    
    Args:
      x: nx2 array [[time val] ...]
             OR
         nx1 array [val ...]
             for the latter indices are taken as time stamps
    
    Returns:
      d: (n-1)x6 array [[residuum delta reset rms_total rms_pre rms_post] ...]
            d[i,] refers to discontinuity between x[i-1,] and x[i,]
    Example:
    >> import numpy as np
    >> import discont as ds
    >> x = np.random.rand(20)
    >> d = ds.discont(x)
    '''


    do_plot = False

    # time: first column or indices
    lx = len(x)
    if np.ndim(x) == 1:
        t = np.arange(0, lx)
        x = np.asarray(x)
    else:
        t = x[:, 0]
        x = x[:, 1]

    # output
    d = np.asarray([])

    # overall linear regression
    c = myPolyfit(t, x, 1)
    y = np.polyval(c, t)

    if do_plot:
        fig = plot_newfig()
        plt.plot(t, x, ":b", t, y, "-r")
        plt.show()

    # residuums
    resid = x-y

    # deltas
    ds = np.diff(x)

    # linear fits
    for i in np.arange(1, lx):
        # preceding, following segment
        i1, i2 = np.arange(0, i), np.arange(i, lx)
        t1, t2, x1, x2 = t[i1], t[i2], x[i1], x[i2]
        # linear fit coefs
        c1 = myPolyfit(t1, x1, 1)
        c2 = myPolyfit(t2, x2, 1)
        # fit values
        y1 = np.polyval(c1, t1)
        y2 = np.polyval(c2, t2)
        # reset
        res = y2[0] - y1[-1]
        # RMSD: pre, post, all
        r1 = utils.rmsd(y[i1], y1)
        r2 = utils.rmsd(y[i2], y2)
        r12 = utils.rmsd(y, np.concatenate((y1, y2)))
        # append to output
        d = utils.push(d, [resid[i], ds[i-1], res, r1, r2, r12])

    return d



def myPolyfit(x, y, o=1):

    '''
    robust wrapper around polyfit to
    capture too short inputs
    
    Args:
      x
      y
      o: order <1>
    
    Returns:
      c: coefs
    '''

    if len(x) == 0:
        return np.zeros(o+1)
    if len(x) <= o:
        return utils.push(np.zeros(o), np.mean(y))
    return np.polyfit(x, y, o)


def plot_sylncl(y, idx):

    '''
    plot extracted yllable nuclei (can be plotted before pruning, too)
    
    Args:
      y: energy contour
      idx: ncl indices (in y)
    '''

    x_dict = {"y": utils.idx(y)}
    y_dict = {"y": y}
    r = [0, 0.15]
    opt = {"ls": {"y": "-k"}}
    # over locmax idxs
    for i in utils.idx(idx):
        z = "s{}".format(i)
        x_dict[z] = [idx[i], idx[i]]
        y_dict[z] = r
        opt["ls"][z] = "-b"
    utils.myPlot(x_dict, y_dict, opt)



def plot_newfig():

    '''
    init new figure with onclick->next, keypress->exit
    
    Returns:
      figureHandle
    '''

    fig = plt.figure()
    cid1 = fig.canvas.mpl_connect('button_press_event', onclick_next)
    cid2 = fig.canvas.mpl_connect('key_press_event', onclick_exit)
    return fig



def onclick_next(event):

    '''
    klick on plot -> next one
    '''

    plt.close()



def onclick_exit(event):

    '''
    press key -> exit
    '''

    sys.exit()

