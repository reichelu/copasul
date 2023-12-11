import copy as cp
import math
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy import interpolate
import scipy.fftpack as sf
import scipy.io.wavfile as sio
import scipy.signal as sis
import sys

import copasul.copasul_utils as utils

def wavread(f, opt=None):

    '''
    read wav file
    
    Args:
      f: (str) file name
      opt: (dict)
    
    Returns:
      signal ndarray
      sampleRate
    '''

    opt = utils.opt_default(opt, {'do_preproc': True})
    
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
      y: (nx1 np.array) signal vector
      opt: (dict)
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
    
    Remarks:
      - if segment is too short (< 5 samples) lowest freqs associated to
        DCT components are too high for ub, that is dct_trunc() returns
        empty array -> np.nan assigned to respective variables
    
    '''

    dflt = {'wintyp': 'kaiser', 'winparam': 1, 'nsm': 3, 'rmo': True,
            'lb': 0, 'ub': 0, 'peak_prct': 80}
    opt = utils.opt_default(opt, dflt)

    # weight window
    w = sig_window(opt['wintyp'], len(y), opt['winparam'])
    y = y * w

    # centralize
    y = y - np.mean(y)

    # DCT coefs
    c = sf.dct(y, norm='ortho')

    # indices (starting with 0)
    ly = len(y)
    ci = utils.idx_a(ly)

    # corresponding cos frequencies
    f = ci + 1 * (opt['fs'] / (ly * 2))

    # band pass truncation of coefs
    # indices of coefs with lb <= freq <= ub
    i = dct_trunc(f, ci, opt)
    
    # analysis segment too short -> DCT freqs above ub
    if len(i) == 0:
        sm = np.array([])
        while len(sm) <= opt['nsm']:
            sm = np.append(sm, np.nan)
        return {'c_orig': c, 'f_orig': f, 'c': np.array([]), 'f': np.array([]), 'i': [],
                'sm': sm, 'opt': opt, 'm': np.nan, 'sd': np.nan, 'cbin': np.array([]),
                'fbin': np.array([]), 'f_max': np.nan, 'f_lmax': np.array([]),
                'c_cog': np.nan}

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
     x: (np.array) of abs coef amplitudes
     f: (np.array) of corresponding frequencies
     cog: (float) center of gravity
    
    Returns:
     f_gm: (float) freq of global maximu
     f_lm: (np.array) of freq of local maxima
     px: (float) threshold to be superseeded (derived from prct specs)
    '''


    x = np.abs(cp.deepcopy(x))

    # global maximum (collect all instead of using np.argmax)
    i = utils.find(x, 'is', 'max')
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
            if len(j) > 0 and f[idx[i]] > fi[j[0]]+fd_min:
                ii.append([idx[i]])
            else:
                ii[-1].append(idx[i])
        else:
            ii[-1].append(idx[i])

    # get index of x maximum within each subsegment
    # and return corresponding frequencies
    f_lm = []
    for si in ii:
        zi = utils.find(x[si], 'is', 'max') # keep
        if len(zi) > 1:
            zi = int(np.mean(zi))
        else:
            zi = zi[0]
        i = si[zi]
        if not np.isnan(i):
            f_lm.append(f[i])

    return f_gm, f_lm, px


def dct_px(x, f, cog, opt):

    '''
    return amplitude threshold for local maxima relative 
    to center of gravity
    
    Args:
       x: (np.array) of coefs
       f: (np.array) of corresponding freqs
       cog: (float) center of gravity freq
       opt: (dict)
    
    Returns:
       t (float) min threshold amplitude
    '''

    x = np.abs(cp.deepcopy(x))

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
            return np.interp(cog, f[i:i+2], x[i:i+2])

    return np.percentile(x, opt['peak_prct'])


def pre_emphasis(y, a=0.95, fs=-1, do_scale=False):

    '''
    pre-emphasis
     alpha > 1 (interpreted as lower cutoff freq)
        alpha <- exp(-2 pi alpha delta)
     s'[n] = s[n]-alpha*s[n-1]
    
    Args:
      signal: (np.array)
      alpha: (float) s[n-1] weight <0.95>
      fs: (int) sample rate <-1>
      do_scale: (boolean) <FALSE> if TRUE than the pre-emphasized signal is scaled to
            same abs_mean value as original signal (in general pre-emphasis
            leads to overall energy loss)

    Returns:
    ype: (np.array) pre-emphasized signal

    '''

    # determining alpha directly or from cutoff freq
    if a > 1:
        if fs <= 0:
            print('pre emphasis: alpha cannot be calculated. Set to 0.95')
            a = 0.95
        else:
            a = math.exp(-2 * math.pi * a * 1 / fs)

    # shifted signal
    ype = np.append(y[0], y[1:] - a * y[:-1])

    # scaling
    if do_scale:
        sf = np.mean(np.abs(y)) / np.mean(np.abs(ype))
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

    return ype


def dct_fbin(f, c, opt):

    '''
    frequency bins: symmetric 2-Hz windows around freq integers
    in bandpass overlapped by 1 Hz
    
    Args:
      f: (np.array) frequencies
      c: (np.array) coefs
      opt: (dict)
         ['lb'] - lower and upper truncation freqs
         ['ub']
    
    Returns:
      fbin: (np.array) lower bnd of freq bins
      cbin: (np.array), summed abs coef values in these bins
    '''

    fb = utils.idx_seg(math.floor(opt['lb']), math.ceil(opt['ub']))
    cbin = np.zeros(len(fb) - 1)
    for j in utils.idx_a(len(fb) - 1):
        k = np.where((f >= fb[j]) & (f <= fb[j+1]))[0]
        cbin[j] = np.sum(np.abs(c[k]))

    fbin = fb[utils.idx_a(len(fb) - 1)]
    return fbin, cbin


def specmom(c, f=[], n=3):

    '''
    spectral moments
    
    Args:
      c: (np.array), coefficients
      f: (np.array) related frequencies <1:len(c)>
      n: (int) number of spectral moments <3>
    
    Returns:
      m: (np.array) moments (increasing)
    '''

    if len(f) == 0:
        f = utils.idx_a(len(c))+1
    c = np.abs(c)
    s = np.sum(c)
    k = 0
    m = []
    for i in utils.idx_seg(1, n):
        m.append(np.sum(c * ((f - k) ** i)) / s)
        k = m[-1]
        
    return np.array(m)


def idct_bp(c, i=np.array([])):

    '''
    wrapper around IDCT
    
    Args:
      c: (np.array) coef vector derived by dct
      i: (np.array) indices of coefs to be taken for IDCT; if empty (default),
          all coefs taken)
    
    Returns:
      y: (np.array) IDCT output
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
      f: (np.array) all frequencies
      ci: (np.array) all indices of coef
      opt: (dict)
         ['lb'] - lower cutoff freq
         ['ub'] - upper cutoff freq
    
    Returns:
      i: (np.array) indices in F of elements to be kept
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


def wrapper_energy(f, opt={}, fs=None):

    '''
    wrapper around wavread and energy calculation
    
    Args:
      f: (str) wavFileName (any number of channels) or array containing
             the signal (any number of channels=columns)
      opt: (opt) energy extraction and postprocessing
           .win, .wintyp, .winparam: window parameters
           .sts: stepsize for energy contour
           .do_preproc: centralizing signal
           .do_out: remove outliers
           .do_interp: linear interpolation over silence
           .do_smooth: smoothing (median or savitzky golay)
           .out dict; see pp_outl()
           .smooth dict; see pp_smooth()
     fs: (int) sampling rate, needed if f is array
    
    Returns:
      y: (nx2 np.array) [[time, energy], ...]

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
    y = []

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
        y.append(list(e))

    y = np.array(y)
    
    # output
    if np.ndim(y) == 1:
        y = np.expand_dims(y, axis=1)
    else:
        y = y.T
        
    # concat time as 1st column
    sts = opt['sts']
    t = np.arange(0, sts * y.shape[0], sts)
    if len(t) != y.shape[0]:
        while len(t) > y.shape[0]:
            t = t[0:len(t) - 1]
        while len(t) < y.shape[0]:
            t = np.append(t, t[-1] + sts)
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

    # "kind" not working if xi values are outside of xp range
    if "kind" in opt:
        if xp[0] < xi[0] or xp[-1] > xi[-1]:
            del opt["kind"]
    
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
      x: (np.array) signal
      opt: (dict)
         ['fs']  - sample frequency
         ['wintyp'] - <'hamming'>, any type supported by
                      scipy.signal.get_window()
         ['winparam'] - <''> additionally needed window parameters,
                        scalar, string, list ...
         ['sts']    - stepsize of moving window
         ['win']    - window length
    
    Returns:
      y: (np.array) energy contour
    '''

    dflt = {'wintyp': 'hamming', 'winparam': None, 'sts': 0.01, 'win': 0.05}
    opt = utils.opt_default(opt, dflt)

    # stepsize and winlength in samples
    sts = round(opt['sts'] * opt['fs'])
    win = min([math.floor(len(x)/2), round(opt['win']*opt['fs'])])

    # weighting window
    w = sig_window(opt['wintyp'], win, opt['winparam'])

    # energy values
    y = []
    for j in utils.idx_a(len(x)-win, sts):
        s = x[j:j+len(w)]*w
        y.append(utils.rmsd(s))
        
    return np.array(y)


def sig_window(typ, l=1, par=None):

    '''
    wrapper around weight window creation
    
    Args:
      typ: (str) any type supported by scipy.signal.get_window()
      lng: (int) <1> length
      par: (any) additional parameters as string, scalar, list etc
    
    Returns:
      window (np.array) weighting window
    '''

    if typ == 'none' or typ == 'const':
        return np.ones(l)
    if par is None:
        par = ""
    if ((type(par) is str) and (len(par) == 0)):
        return sis.get_window(typ, l)

    return sis.get_window((typ, par), l)


def splh_spl(sig, fs, opt_in={}):

    '''
    spectral balance calculation according to Fant 2000
    
    Args:
      sig: (np.array) signal (vowel segment)
      fs: (int) sampe rate
      opt: (dict)
         'domain': <'freq'>|'time'; pre-emp in frequency (Fant) or time domain
         'win': length of central window in ms <len(sig)>; -1 is same as len(sig)
          'alpha': <0.95> for time domain only y[n] = x[n]-alpha*x[n-1]
               if alpha>0 it is interpreted as lower freq threshold for pre-emp
    
    Returns:
      sb: (float) spectral tilt
    '''

    opt = cp.deepcopy(opt_in)
    opt = utils.opt_default(opt, {'win': len(sig), 'f': -1, 'btype': 'none',
                                  'domain': 'freq', 'alpha': 0.95})

    # cut out center window
    ls = len(sig)
    if opt['win'] <= 0:
        opt['win'] = ls
    if opt['win'] < ls:
        wi = utils.windowing_idx(int(ls / 2),
                                 {'rng': [0, ls],
                                  'win': int(opt['win'] * fs)})
        y = sig[wi]
    else:
        y = cp.deepcopy(sig)

    if len(y) == 0:
        return np.nan

    # reference sound pressure level
    p_ref = pRef('spl')

    # pre-emp in time domain ######################
    if opt['domain'] == 'time':
        # low pass filtering
        if opt['btype'] != 'none':
            flt = fu_filt(y, {'fs': fs, 'f': opt['f'], 'ord': 6,
                              'btype': opt['btype']})
            y = flt['y']
        yp = pre_emphasis(y, opt['alpha'], fs, False)
        y_db = 20 * np.log10(utils.rmsd(y) / p_ref)
        yp_db = 20 * np.log10(utils.rmsd(yp) / p_ref)
        return yp_db - y_db

    # pre-emp in frequency domain ################
    # according to Fant
    # actual length of cut signal
    n = len(y)
    # hamming windowing
    y *= np.hamming(n)
    # spectrum
    Y = np.fft.fft(y, n)
    N = int(len(Y) / 2)
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
    a = 2 * np.abs(Y[:N]) / N
    # vowel-relevant upper frequency boundary
    
    if opt['btype'] != 'none':
        vi = fu_filt_freq(X, opt)
        if len(vi) > 0:
            X = X[vi]
            a = a[vi]
            
    # Fant preemphasis filter (Fant et al 2000, p10f eq 20)
    preemp = 10 * np.log10((1 + X ** 2 / 200 ** 2) / (1 + X ** 2 / 5000 ** 2))
    ap = 10 * np.log10(a) + preemp

    # retransform to absolute scale
    ap = 10 ** (ap / 10)

    # corresponds to gain values in Fant 2000, p11
    # get sound pressure level of both spectra
    #       as 20 * log10(P_eff / P_ref)
    spl = 20 * np.log10(utils.rmsd(a) / p_ref)
    splh = 20 * np.log10(utils.rmsd(ap) / p_ref)

    # get energy level of both spectra
    # spl = 20 * np.log10(utils.mse(a) / p_ref)
    # splh = 20 * np.log10(utils.mse(ap) / p_ref)
    # spectral balance
    sb = splh - spl

    # fig = plt.figure()
    # plt.plot(X, 20 * np.log10(a), 'b')
    # plt.plot(X, 20 * np.log10(preemp), 'g')
    # plt.plot(X, 20 * np.log10(ap), 'r')
    # plt.show()

    return sb


def fu_filt_freq(X, opt):

    '''
    returns indices of freq in x fullfilling conditions in opt
    
    Args:
      X: (np.array) freq array
      opt: (dict)
           'btype' - 'none'|'low'|'high'|'band'|'stop'
           'f': (float or list) freq for low|high, band|stop
    
    Returns:
      i: (np.array) indices in X fulfilling condition
    '''

    typ = opt['btype']
    f = opt['f']
    
    # all indices
    if typ == 'none':
        return utils.idx_a(len(X))
    
    # error handling
    if re.search(r'(band|stop)', typ) and (not utils.listType(f)):
        print('filter type requires frequency list. Done nothing.')
        return utils.idx_a(len(X))
    if re.search(r'(low|high)', typ) and utils.listType(f):
        print('filter type requires only 1 frequency value. Done nothing.')
        return utils.idx_a(len(X))

    if typ == 'low':
        return np.where(X <= f)[0]
    elif typ == 'high':
        return np.where(X >= f)[0]
    elif typ == 'band':
        return np.where((X >= f[0]) & (X <= f[1]))[0]
    elif typ == 'stop':
        return np.where((X <= f[0]) | (X >= f[1]))[0]

    return utils.idx_a(len(X))


def pRef(typ):

    '''
    returns reverence levels for typ
    
    Args:
      typ: (str)
       'spl': sound pressure level
       'i': intensity level
    
    Returns:
      ref: (float) corresponding reference level
    '''

    if typ == 'spl':
        return 2 * 10 ** (-5)
    return 10 ** (-12)


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
            t[-1, -1] = len(y) - 1

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
    t = t + opt['ons']
    tc = tc + opt['ons']

    # return dict
    # incl fields with indices to seconds (index+1=sampleIndex)
    pau = {'tpi': t, 'tci': tc, 'e_ratio': e_ratio}
    pau['tp'] = utils.idx2sec(t, opt['fs'])
    pau['tc'] = utils.idx2sec(tc, opt['fs'])

    return pau


def pau_detector_sub(y, opt):

    '''
    called by pau_detector
    
    Args:
       as for pau_detector
    
    Returns:
       t: (np.array) [[on off], ...]
       e_ratio: (np.array)
    '''

    # settings
    # reference window span
    rl = math.floor(opt['l_ref'] * opt['fs'])
    
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
    ya = np.abs(y)
    qq = np.percentile(ya, [50])
    e_glob = utils.rmsd(ya[ya > qq[0]])
    t_glob = opt['e_rel'] * e_glob

    # stepsize
    sts = max([1, math.floor(0.05*opt['fs'])])
    
    # energy calculation in analysis and reference windows
    wopt_en = {'win': ml, 'rng': [0, ls]}
    wopt_ref = {'win': rl, 'rng': [0, ls]}

    # loop until opt.n criterion is fulfilled
    # increasing energy threshold up to 1
    while e_rel < 1:
        # pause [on off], pause index
        t = []
        j = 0
        # [e_y/e_rw] indices as in t
        e_ratio = []
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
            if e_y <= e_r * e_rel:
                yis = yi[0]
                yie = yi[-1]
                if len(t) - 1 == j:
                    # values belong to already detected pause
                    if len(t) > 0 and yis < t[j][1]:
                        t[j][1] = yie
                        # evtl. needed to throw away superfluous
                        # pauses with high e_ratio
                        e_ratio[j] = np.mean([e_ratio[j], e_y / e_r])
                    else:
                        t.append([yis, yie])
                        e_ratio.append(e_y / e_r)
                        j = j+1
                else:
                    t.append([yis, yie])
                    e_ratio.append(e_y / e_r)
                    
        # (more than) enough pauses detected?
        if len(t) >= opt['n']:
            break
        e_rel = e_rel + 0.1
        
    if opt['margin'] == 0 or len(t) == 0:
        return np.array(t), np.array(e_ratio)

    # shorten pauses by margins
    mar = int(opt['margin'] * opt['fs'])
    tm, erm = [], []

    for i in utils.idx_a(len(t)):
        
        # only slim non-init and -fin pauses
        if i > 0:
            ts = t[i][0] + mar
        else:
            ts = t[i][0]
        if i < len(t) - 1:
            te = t[i][1] - mar
        else:
            te = t[i][1]

        # pause disappeared
        if te <= ts:
            
            # ... but needs to be kept
            if opt['n'] > 0:
                tm.append(t[i])
                erm.append(e_ratio[i])
            continue
        
        # pause still there
        tm.append([ts, te])
        erm.append(e_ratio[i])

    return np.array(tm), np.array(erm)


def pau_detector_red(t, e_ratio, opt):

    ''' remove pauses with highes e_ratio
    (if number of pause restriction applies)

    Args:
    t: (np.array) [[on, off], ...]
    e_ratio: (np.array) of energy ratios
    opt: (dict)

    Returns:
    t: (np.array)
    e_ratio: (np.array)

    '''
    
    # keep boundary pauses
    if opt['fbnd'] == True:
        n = opt['n'] - 2
        bp = np.concatenate((np.array([t[0,]]), np.array([t[-1,]])), axis=0)
        ii = np.arange(1, len(t)-1, 1)
        t = t[ii,]
        e_ratio = e_ratio[ii]
    else:
        n = opt['n']
        bp = np.array([])

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


def pau_detector_merge(t, e, opt):

    '''
    merging pauses across too short chunks
    merging chunks across too small pauses
    
    Args:
      t: (np.array) [[on off]...] of pauses
      e: (np.array) [e_rat ...]
    
    Returns:
      t [[on off]...] merged
      e [e_rat ...] merged (simply mean of merged segments taken)
    '''

    # min pause and chunk length in samples
    mpl = utils.sec2smp(opt['min_pau_l'], opt['fs'])
    mcl = utils.sec2smp(opt['min_chunk_l'], opt['fs'])

    # merging chunks across short pauses
    tm = []
    em = []
    for i in utils.idx_a(len(t)):
        if ((t[i, 1]-t[i, 0] >= mpl) or
                (opt['fbnd'] == True and (i == 0 or i == len(t)-1))):
            tm.append(list(t[i, :]))
            em.append(e[i])

    # nothing done in previous step?
    if len(tm) == 0:
        tm = cp.deepcopy(t)
        em = cp.deepcopy(e)
    if len(tm) == 0:
        return t, e

    tm = np.array(tm)
    em = np.array(em)
    
    # merging pauses across short chunks
    tn = list([tm[0, :]])
    en = [em[0]]
    if (tn[0][0] < mcl):
        tn[0][0] = 0.0
    for i in np.arange(1, len(tm), 1):
        if (tm[i, 0] - tn[-1][1] < mcl):
            tn[-1][1] = tm[i, 1]
            en[-1] = np.mean([en[-1], em[i]])
        else:
            tn.append(list(tm[i, :]))
            en.append(em[i])

    return np.array(tn), np.array(en)


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
        return np.array([[0, l - 1]])
    if t[0, 0] > 0:
        tc = np.array([[0, t[0, 0] - 1]])
    else:
        tc = np.array([])
    for i in np.arange(0, len(t) - 1, 1):
        if t[i, 1] < t[i + 1, 0] - 1:
            tc = utils.push(tc, [t[i, 1] + 1, t[i + 1, 0] - 1])
    if t[-1, 1] < l - 1:
        tc = utils.push(tc, [t[-1, 1] + 1, l - 1])
        
    return tc


def syl_ncl(s, opt={}):

    '''
    syllable nucleus detection
    
    Args:
      s: (np.array) mono signal
      opt: (dict)
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
      ncl: (dict)
        ncl['t'] - vector of syl ncl time stamps (in sec)
           ['ti'] - corresponding vector idx in s
           ['e_ratio'] - corresponding energy ratios (analysisWindow/referenceWindow)
      bnd: (dict)
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
        t = np.asarray([round(len(s) / 2 + opt['ons'])])
        ncl = {'ti': t, 't': utils.idx2sec(t, opt['fs']), 'e_ratio': [0]}
        bnd = cp.deepcopy(ncl)
        return ncl, bnd

    # reference window length
    rws = math.floor(opt['l_ref'] * opt['fs'])
    
    # energy win length
    ml = math.floor(opt['l']*opt['fs'])

    # stepsize
    sts = max([1, math.floor(0.03*opt['fs'])])

    # minimum distance between subsequent nuclei
    # (indices)
    md = math.floor(opt['d_min'] * opt['fs'])

    # bandpass filtering
    flt = fu_filt(s, opt['flt'])
    y = flt['y']
    
    # signal length
    ls = len(y)

    # minimum energy as proportion of maximum energy found
    e_y = []
    i_steps = np.arange(1, ls, sts)

    for i in i_steps:
        yi = np.arange(i, min([ls, i+ml-1]), 1)
        e_y.append(utils.rmsd(y[yi]))

    e_y = np.array(e_y)
    if bool(opt['center']):
        e_y -= np.mean(e_y)

    e_min = opt['e_min'] * max(e_y)
    # output vector collecting nucleus sample indices
    all_i = []
    all_e = []
    all_r = []

    # energy calculation in analysis and reference windows
    wopt_en = {'win': ml, 'rng': [0, ls]}
    wopt_ref = {'win': rws, 'rng': [0, ls]}
    for i in i_steps:
        yi = utils.windowing_idx(i, wopt_en)
        ys = y[yi]
        e_y = utils.rmsd(ys)
        ri = utils.windowing_idx(i, wopt_ref)
        rs = y[ri]
        e_rw = utils.rmsd(rs)
        all_i.append(i)
        all_e.append(e_y)
        all_r.append(e_rw)

    all_i = np.array(all_i)
    all_e = np.array(all_e)
    all_r = np.array(all_r)
        
    # local energy maxima
    # (do not use min duration md for order option, since local
    #  maximum might be obscured already by energy increase
    #  towards neighboring peak further away than md, and not only by
    #  closer than md peaks)
    idx = sis.argrelmax(all_e, order=1)

    # plot_sylncl(all_e,idx) #!v
    
    # maxima related to syl ncl
    # a) energy constraints
    # timestamps (idx)
    tx = []

    # energy ratios
    e_ratiox = []

    # idx in all_i
    tix = []

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
                if all_e[i] > ie[0]:
                    all_e[tix[-1]] = all_e[i]
                    tx[-1] = all_i[i]
                    tix[-1] = i
                    e_ratiox[-1] = all_e[i] / all_r[i]

                continue

        if ((all_e[i] >= all_r[i] * opt['e_rel']) and (all_e[i] > e_min)):
            tx.append(all_i[i])
            tix.append(i)
            e_ratiox.append(all_e[i] / all_r[i])

    if len(tx) == 0:
        dflt = {'ti': np.array([]),
                't': np.array([]),
                'e_ratio': np.array([])}
        return dflt, dflt

    tx = np.array(tx)
    tix = np.array(tix, dtype=int)
    e_ratiox = np.array(e_ratiox)
    
    # plot_sylncl(all_e,tix) #!v
    # b) min duration constraints
    # init by first found ncl

    t = [tx[0]]
    e_ratio = [e_ratiox[0]]
    # idx in all_i
    ti = [tix[0]]

    for i in range(1, len(tx)):
        # ncl too close
        if np.abs(tx[i] - t[-1]) < md:
            # current ncl with higher energy: replace last stored one
            if e_ratiox[i] > e_ratio[-1]:
                t[-1] = tx[i]
                ti[-1] = tix[i]
                e_ratio[-1] = e_ratiox[i]
        else:
            t.append(tx[i])
            ti.append(tix[i])
            e_ratio.append(e_ratiox[i])


    t = np.array(t)
    ti = np.array(ti, dtype=int)
    e_ratio = np.array(e_ratio)
    
    # plot_sylncl(all_e,ti)
    
    # minima related to syl bnd
    tb = []
    e_ratio_b = []

    if len(t) > 1:
        for i in range(len(ti) - 1):
            j = utils.idx_seg(ti[i], ti[i+1])
            j_min = np.argmin(all_e[j])
            
            # bnd idx
            bj = j[0] + j_min
            tb.append(all_i[bj])
            e_ratio_b.append(all_e[bj] / all_r[bj])
            
    tb = np.array(tb, dtype=int)

    # add onset
    t = t + opt['ons']
    tb = tb + opt['ons']

    # output dict,
    # incl idx to seconds
    ncl = {'ti': t, 't': utils.idx2sec(t, opt['fs']), 'e_ratio': e_ratio}
    bnd = {'ti': tb, 't': utils.idx2sec(tb, opt['fs']), 'e_ratio': e_ratio_b}

    return ncl, bnd


def syl_ncl_trouble(s, opt):
    if len(s) / opt['fs'] < 0.1:
        return True
    return False


def fu_filt(y, opt):

    '''
    wrapper around Butter filter
    
    Args:
      y: (np.array) 1-dim vector
      opt: (dict)
        opt['fs'] - sample rate
           ['f']  - scalar (high/low) or 2-element vector (band) of boundary freqs
           ['order'] - order
           ['btype'] - band|low|high; all other values: signal returned as is
    
    Returns:
      flt: (dict)
        flt['y'] - filtered signal
           ['b'] - coefs
           ['a']
    '''

    # do nothing
    if not re.search(r'^(high|low|band)$', opt['btype']):
        return {'y': y, 'b': np.array([]), 'a': np.array([])}

    # check f < fs / 2
    if (opt['btype'] == 'low' and opt['f'] >= opt['fs'] / 2):
        opt['f'] = opt['fs'] / 2 - 100
    elif (opt['btype'] == 'band' and opt['f'][1] >= opt['fs'] / 2):
        opt['f'][1] = opt['fs'] / 2 - 100
    fn = opt['f'] / (opt['fs'] / 2)
    b, a = sis.butter(opt['ord'], fn, btype=opt['btype'])
    yf = sis.filtfilt(b, a, y)
    
    return {'y': yf, 'b': b, 'a': a}


def plot_sylncl(y, idx):

    '''
    plot extracted syllable nuclei (can be plotted before pruning, too)
    
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

