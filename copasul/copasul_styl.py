import copy as cp
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy import interpolate
import sys
from tqdm import tqdm

import copasul.copasul_plot as copl
import copasul.copasul_utils as utils
import copasul.copasul_sigproc as cosp


def styl_gnl(copa, typ, f_log_in=None, silent=False):

    '''
    general f0 and en features (mean and sd)

    Args:
      copa (dict)
      typ (str) 'f0'|'en', pitch or energy
      f_log (file handle) log file
      silent: (boolean) if True, tqdm bar is supressed
    
    Returns: (ii=fileIdx, i=channelIdx, j=tierIdx, k=segmentIdx)
      + ['data'][ii][i]['gnl_f0|en_file']; see styl_std_feat()
      + ['data'][ii][i]['gnl_f0|en'][j][k]['std'][*]; see styl_std_feat()
      ['gnl_en']...['std'] additionally contains ...['sb'] for spectral balance,
              and ['r_en_f0'] for correlation with f0
    '''

    global f_log
    f_log = f_log_in

    myLog(f"DOING: styl gnl {typ} ...")

    fld = f"gnl_{typ}"
    utils.check_var({'var': copa, 'env': 'copasul',
                     'spec': {'step': 'styl', 'dom': fld}})

    if fld in copa['config']['styl']:
        opt = cp.deepcopy(copa['config']['styl'][fld])
    else:
        opt = {}
    opt['type'] = typ
    utils.check_var({'var': copa, 'env': 'copasul',
                     'spec': {'step': 'styl', 'dom': fld}})

    # over files
    file_ii = utils.sorted_keys(copa['data'])
    if silent:
        for ii in file_ii:
            copa = styl_gnl_file(copa, ii, fld, opt)
    else:
        for ii in tqdm(file_ii , desc=f"styl {fld}"):
            copa = styl_gnl_file(copa, ii, fld, opt)
            
    return copa


def styl_gnl_file(copa, ii, fld, opt):

    '''
    
    GNL feature exraction per file

    Args:
    copa: (dict)
    ii: (int) file index
    fld: (str) key in copa
    opt: (dict)

    Returns:
    copa: (dict) updated

    case typ='en':
      y_raw_in, t_raw refer to raw signal
      y_raw - resp channel of signal
      y, t refer to energy contour
    myFs: sample freq for F0 or energy contour (derived by opt...['sts']); not for raw signal
    '''

    # wav of f0 input (wav to be loaded once right here)
    if opt['type'] == 'en':

        # mean subtraction? (default; translates to do_preproc)
        if (('centering' not in opt) or opt['centering']):
            meanSub = True
        else:
            meanSub = False
        iii = utils.sorted_keys(copa['data'][ii])
        fp = copa['data'][ii][iii[0]]['fsys']['aud']
        f = f"{fp['dir']}/{fp['stm']}.{fp['ext']}"
        y_raw_in, fs_sig = cosp.wavread(f, {'do_preproc': meanSub})
    else:
        myFs = copa['config']['fs']

    # over channels
    for i in utils.sorted_keys(copa['data'][ii]):

        fstm = copa['data'][ii][i]['fsys']['aud']['stm']
        myLog(f"\tfile {fstm}, channel {i+1}")

        if opt['type'] == 'f0':
            t = copa['data'][ii][i]['f0']['t']
            y = copa['data'][ii][i]['f0']['y']
            opt['fs'] = myFs
            y_f0i, r_ef = None, None
        else:

            # for f0-energy correlation
            t_f0 = copa['data'][ii][i]['f0']['t']
            y_f0 = copa['data'][ii][i]['f0']['y']
            if np.ndim(y_raw_in) > 1:
                y_raw = y_raw_in[:, i]
            else:
                y_raw = y_raw_in

            # signal fs for energy extraction
            opt['fs'] = fs_sig

            # energy
            y = cosp.sig_energy(y_raw, opt)
            myFs = int(1 / opt['sts'])
            
            t = utils.smp2sec(utils.idx_seg(1, len(y)), myFs, 0)

            # energy fs for standard feature extraction
            opt['fs'] = myFs

            # sync f0 contour to energy contour
            interp = interpolate.interp1d(t_f0, y_f0, kind="linear",
                                          fill_value=(y_f0[0], y_f0[-1]))
            y_f0i = interp(t)

            # correlation of energy and f0 contour
            r_ef = utils.robust_corrcoef(y, y_f0i)

        # file wide
        yz = f"{fld}_file"
        copa['data'][ii][i][yz] = styl_std_feat(y, opt)
        copa['data'][ii][i][yz] = styl_std_quot(
            y, opt, copa['data'][ii][i][yz])
        if r_ef is not None:
            copa['data'][ii][i][yz]["r_en_f0"] = r_ef

        # over tiers
        for j in utils.sorted_keys(copa['data'][ii][i][fld]):

            # over segments (+ normalization ['*_nrm'])
            nk = utils.sorted_keys(copa['data'][ii][i][fld][j])
            for k in nk:
                
                # analysis window
                yi = styl_yi(copa['data'][ii][i][fld][j][k]['t'], myFs, y)
                
                # for interval tier input; for styl_std_feat duration calculation
                if len(copa['data'][ii][i][fld][j][k]['to']) > 1:
                    to = copa['data'][ii][i][fld][j][k]['to'][0:2]
                else:
                    to = np.array([])
                tn = copa['data'][ii][i][fld][j][k]['tn']
                sf = styl_std_feat(y[yi], opt, to)
                
                # nrm window
                yin = styl_yi(tn, myFs, y)

                # add normalization _nrm
                sf = styl_std_nrm(sf, y[yin], opt, tn)

                # quotients/shape
                sf = styl_std_quot(y[yi], opt, sf)

                # add spectral balance, rmsd, and r_en_f0 for en
                if (opt['type'] == 'en'):
                    yi_raw = styl_yi(copa['data'][ii][i]
                                     [fld][j][k]['t'], fs_sig, y_raw)
                    yin_raw = styl_yi(tn, fs_sig, y_raw)
                    rms_y = utils.rmsd(y_raw[yi_raw])
                    rms_yn = utils.rmsd(y_raw[yin_raw])
                    sb = cosp.splh_spl(y_raw[yi_raw], fs_sig, opt['sb'])
                    if rms_yn == 0:
                        rms_yn == 1
                    sf['sb'] = sb
                    sf['rms'] = rms_y
                    sf['rms_nrm'] = rms_y / rms_yn
                    r_ef = utils.robust_corrcoef(y[yi], y_f0i[yi])
                    sf['r_en_f0'] = r_ef

                copa['data'][ii][i][fld][j][k]['std'] = sf
                
    return copa


def styl_std_quot(y, opt, r={}):

    '''
    calculates quotients and 2nd order shape coefs for f0/energy contours
    
    Args:
      y: (np.array) f0 or energy contout
      opt: (dict) config['styl']
      r: (dict) <{}> of stylization results
    
    Returns:
      r: (dict) output + keys
        qi mean_init/mean_nonInit
        qf mean_fin/mean_nonFin
        qb mean_init/mean_fin
        qm mean_max(initFin)/mean_nonMax
        c0 offset
        c1 slope
        c2 shape
    '''

    # backward compatibility
    if 'gnl' not in opt:
        opt['gnl'] = {'win': 0.3}

    # init
    for x in ['qi', 'qf', 'qm', 'qb', 'c0', 'c1', 'c2']:
        r[x] = np.nan

    if len(y) == 0:
        return r

    yl = len(y)

    # window length (smpl)
    wl = min(yl, int(opt['fs'] * opt['gnl']['win']))

    # initial and final segment
    y_ini = y[0:wl],
    y_nin = y[wl:yl]
    y_fin = y[yl-wl:yl]
    y_nfi = y[0:yl-wl]

    # robustness: non-empty slices
    if len(y_ini) == 0:
        y_ini = [y[0]]
    if len(y_nin) == 0:
        y_nin = [y[-1]]
    if len(y_fin) == 0:
        y_fin = [y[-1]]
    if len(y_nfi) == 0:
        y_nfi = [y[0]]

    # means
    mi = np.mean(y_ini)
    mni = np.mean(y_nin)
    mf = np.mean(y_fin)
    mnf = np.mean(y_nfi)

    # quotients
    if mni > 0:
        r['qi'] = mi / mni
    if mnf > 0:
        r['qf'] = mf / mnf
    if mf > 0:
        r['qb'] = mi / mf

    # max quot
    if np.isnan(r['qi']):
        r['qm'] = r['qf']
    elif np.isnan(r['qf']):
        r['qm'] = r['qi']
    else:
        r['qm'] = max(r['qi'], r['qf'])

    # polyfit
    t = utils.nrm_vec(utils.idx_a(len(y)), {'mtd': 'minmax', 'rng': [0, 1]})
    c = styl_polyfit(t, y, 2)
    r['c0'] = c[2]
    r['c1'] = c[1]
    r['c2'] = c[0]

    return r


def styl_std_nrm(sf, ys, opt, tn=[]):

    '''
    adds normalized feature values to std feat dict
    
    Args:
       sf: (dict) std feat dict
       ys: (np.array) vector f0/energy in normalization window
       opt: (dict) copa['config']['styl']
       tn: (list) <[]> t_on, t_off of normalization interval
    
    Returns:
       sf: (dict) + '*_nrm' entries
    '''

    sfn = styl_std_feat(ys, opt, tn)
    nrm_add = {}
    for x in sf:
        if sfn[x] == 0:
            sfn[x] = 1
        nrm_add[x] = sf[x] / sfn[x]
    for x in nrm_add:
        sf[f"{x}_nrm"] = nrm_add[x]
    return sf


def styl_yi(iv, fs, y=[]):

    '''
    transforms time interval to index array
    
    Args:
      iv: (np.array) 2-element array, time interval
      fs: (int) sample rate
      y: (list) vector (needed, since upper bound sometimes >=len(y))
    
    Returns:
      i: (np.array) indices between on and offset defined by iv
    '''

    j = utils.sec2idx(np.array(iv), fs)
    if len(y) > 0:
        return utils.idx_seg(j[0], min(j[1], len(y)-1))
    else:
        return utils.idx_seg(j[0], j[1])

    
def styl_voice(copa, f_log_in=None):

    '''
    
    Args:
      copa: (dict)
      f_log_in: (handle) of log file
    
    Returns:
      copa: (dict)
      + .data...[voice|voice_file]
             .shim
                .v: shimmer
                .c: 3rd order polycoefs
                .v_nrm: normalized shimmer
                .m: mean amplitude
                .m_nrm: file-normalized mean amplitude
                .sd amplitude sd
                .sd_nrm: file-normalized amplitude sd
             .jit
                .v: jitter (relative)
                .v_abs: jitter (absolute)
                .v_nrm: normalized jitter (relative)
                .m: mean period length (in sec)
                .m_nrm: normalized mean period length
                .sd: period length sd
                .sd_nrm: normalized period length sd
                .c: 3rd order polycoefs
    (*_nrm keys only for "voice" subdict, not for "voice_file")
    '''

    global f_log
    f_log = f_log_in

    myLog("DOING: styl voice")

    fld = 'voice'
    if fld in copa['config']['styl']:
        opt = cp.deepcopy(copa['config']['styl'][fld])
    else:
        opt = {}
    opt = utils.opt_default(opt, {'shim': {}, 'jit': {}})

    # over files
    file_ii = utils.sorted_keys(copa['data'])
    for ii in tqdm(file_ii , desc=f"styl {fld}"):
        copa = styl_voice_file(copa, ii, fld, opt)

    return copa


def styl_voice_file(copa, ii, fld, opt):

    '''
    styl_voice() for single file

    Args:
    copa: (dict)
    ii: (int) file index
    fld: (str) key in copa
    opt: (dict)

    Returns:
    copa: (dict) updated

    '''

    chan_i = utils.sorted_keys(copa['data'][ii])
    fp = copa['data'][ii][chan_i[0]]['fsys']['pulse']
    fa = copa['data'][ii][chan_i[0]]['fsys']['aud']
    f = f"{fp['dir']}/{fp['stm']}.{fp['ext']}"
    fw = f"{fa['dir']}/{fa['stm']}.{fa['ext']}"

    # pulses as 2-dim list, one column per channel
    p = utils.input_wrapper(f, 'lol', {'colvec': True})

    # signal
    sig_all, fs_sig = cosp.wavread(fw)
    opt['fs_sig'] = fs_sig

    # over channels
    for i in chan_i:

        fstm = copa['data'][ii][i]['fsys']['aud']['stm']
        myLog(f"\tfile {fstm}, channel {i+1}")

        # jitter and shimmer file wide
        fld_f = f"{fld}_file"

        # channel-related column
        if np.ndim(sig_all) > 1:
            sig = sig_all[:, i]
        else:
            sig = sig_all

        copa['data'][ii][i][fld_f] = styl_voice_feat(p[:, i], sig, opt)

        # normalization facors for segment-level voice quality
        voi_nrm = copa['data'][ii][i][fld_f]

        # over tiers
        for j in utils.sorted_keys(copa['data'][ii][i][fld]):
            nk = utils.sorted_keys(copa['data'][ii][i][fld][j])
            # over segments
            for k in nk:
                # resp. segments in pulse sequence and signal
                lb = copa['data'][ii][i][fld][j][k]['t'][0]
                ub = copa['data'][ii][i][fld][j][k]['t'][1]
                pul_i = np.where((p[:, i] >= lb) & (p[:, i] <= ub))[0]                
                voi = styl_voice_feat(p[pul_i, i], sig, opt, voi_nrm)
                for x in voi:
                    copa['data'][ii][i][fld][j][k][x] = voi[x]

    return copa


def styl_voice_feat(pul, sig, opt, nrm=None):

    '''
    
    Args:
      pul: (np.array) pulse time stamps
      sig: (sp.array) signal
      opt: (dict)
        'fs_sig' required
        'shim': <not yet used> {}
        'jit':
           't_max' max distance in sec of subsequent pulses to be part of
                   same voiced segment <0.02> (=50 Hz; praat: Period ceiling)
           't_min' min distance in sec <0.0001> (Period floor)
           'fac_max' factor of max adjacent period difference <1.3>
                   (Maximum period factor)
     nrm: (dict)
        'jit' and 'shim' subdicts generated by styl_voice_feat() over entire file
    
    Returns:
      (dict)
          'jit': see styl_voice_jit()
          'shim': see styl_voice_shim()
    '''

    # remove -1 padding from input file
    pul = pul[pul > 0]
    opt = opt_voice_default(opt)
    if nrm is not None:
        nrm_jit, nrm_shim = nrm['jit'], nrm['shim']
    else:
        nrm_jit, nrm_shim = None, None
    jit = styl_voice_jit(pul, opt['jit'], nrm_jit)
    shim = styl_voice_shim(pul, sig, opt['shim'], nrm_shim)
    return {'jit': jit, 'shim': shim}


def opt_voice_default(opt=None):

    '''
    returns default options for voice analyses
    '''

    opt = utils.opt_default(opt, {'shim': {}, 'jit': {}})
    opt['shim'] = utils.opt_default(opt['shim'], {'fs_sig': opt['fs_sig']})
    opt['jit'] = utils.opt_default(opt['jit'], {'fs_sig': opt['fs_sig'],
                                                't_max': 0.02, 't_min': 0.0001,
                                                'fac_max': 1.3})
    return opt


def styl_voice_shim(pul, sig, opt, nrm):

    '''
    shimmer: average of abs diff between amplitudes of subsequent pulses
             divided by mean amplitude
    
    Args:
      pul: (np.array) vector of pulse sequence
      sig: (np.array) signal
      opt: (dict) ('fs_sig': sample rate)
      nrm: (dict) output of styl_voice_shim() applied on file level
    
    Returns:
      (dict)
        v: shimmer
        c: 3rd order polycoefs describing changes of shimmer over normalized time
        m: mean amplitude
        sd: std amplitude
        {v|m|sd}_nrm: divided by file-level values
    '''

    # get amplitude values at pulse time stamps
    a = []
    # time stamps
    t = []
    
    # robust return
    robRet = {'v': np.nan, 'c': np.nan * np.ones(4),
              'm': np.nan, 'sd': np.nan, 'v_nrm': np.nan,
              'm_nrm': np.nan, 'sd_nrm': np.nan}

    for i in utils.idx(pul):
        j = utils.sec2idx(pul[i], opt['fs_sig'])
        if j >= len(sig):
            break
        a.append(np.abs(sig[j]))
        t.append(pul[i])
        
    if len(a) == 0:
        return robRet

    a = np.array(a)
    t = np.array(t)
    
    d = np.abs(np.diff(a))
    ma = np.mean(a)
    sda = np.std(a)
    # shimmer
    s = np.mean(d) / ma

    # 3rd order polynomial fit through normalized amplitude diffs
    # time points between compared periods
    # normalized to [-1 1] by range of [pul[0] pul[-1]]
    if ma > 0:
        d = d / ma

    if len(t) < 2:
        c = robRet['c']
    else:
        x = utils.nrm_vec(t[1:len(t)], {'mtd': 'minmax', 'rng': [-1, 1]})
        c = styl_polyfit(x, d, 3)

    ret = {'v': s, 'c': c, 'm': ma, 'sd': sda}

    # add normalized values
    return voice_nrm(ret, nrm)


def styl_voice_jit(pul, opt, nrm):

    '''
    
    Args:
      pul: (np.array) vector of pulse sequence
      opt: (dict) options
      nrm: (dict) for normalization
    
    Returns:
      (dict)
        v: jitter (relative)
        v_abs: jitter (absolute)
        m: mean period length (in sec)
        sd: standard deviation
        {v|v_abs|m|sd}_nrm: divided by file-level values
        c: 3rd order polycoefs describing changes of jitter over normalized time
    '''

    # all periods
    ta = np.diff(pul)
    
    # abs diffs of adjacent periods
    d = []

    # indices of these periods
    ui = [] # np.array([]).astype(int)
    i = 0
    while i < len(ta)-1:
        ec = skip_period(ta, i, opt)
        # skip first segment
        if ec > 0:
            i += ec
            continue
        d.append(np.abs(ta[i]-ta[i+1]))
        ui.append(i)
        i += 1

    d = np.array(d)
    ui = np.array(ui, dtype=int)
        
    # not enough values
    if len(ui) == 0:
        v = np.nan
        return {'v': v, 'v_abs': v, 'm': v, 'sd': v,
                'v_nrm': v, 'v_abs_nrm': v, 'm_nrm': v,
                'sd_nrm': v, 'c': np.array([v, v, v, v])}

    # all used periods
    periods = ta[ui]

    # mean and std period
    mp = np.mean(periods)
    sdp = np.std(periods)

    # jitter absolute
    jit_abs = np.sum(d) / len(d)

    # jitter relative
    jit = jit_abs / mp

    # 3rd order polynomial fit through all normalized period diffs
    # time points between compared periods
    # normalized to [-1 1] by range of [pul[0] pul[-1]]
    if mp > 0:
        d = d / mp
    x = utils.nrm_vec(pul[ui+1], {'mtd': 'minmax', 'rng': [-1, 1]})
    c = styl_polyfit(x, d, 3)

    ret = {'v': jit, 'v_abs': jit_abs, 'm': mp, 'sd': sdp, 'c': c}

    # add normalized values
    return voice_nrm(ret, nrm)


def voice_nrm(ret, nrm):

    '''
    add normalized voice quality values to ret
    for keys *_nrm.
    If nrm is None: ret[x_nrm] = ret[x]
    '''

    ret_nrm = cp.deepcopy(ret)

    for x in ret:
        
        # skip coefs
        if x == "c":
            continue
        if nrm is None:
            v = ret[x]
        else:
            v = ret[x] / nrm[x]

        ret_nrm[f"{x}_nrm"] = v

    return ret_nrm


def skip_period(d, i, opt):

    '''
    returns error code which is increment in jitter calculation
    0: all fine
    '''

    # 1st period problem
    if d[i] < opt['t_min'] or d[i] > opt['t_max']:
        return 1

    # 2nd period problem
    if d[i+1] < opt['t_min'] or d[i+1] > opt['t_max']:
        return 2

    # transition problem
    if max(d[i], d[i+1]) > opt['fac_max']*min(d[i], d[i+1]):
        return 1
    return 0


def styl_glob(copa, f_log_in=None, silent=False):

    '''
    global segments
  
    Args:
      copa: (dict)
      f_log_in: (file handle)
      silent: (bool) if True, tqdm is suppressed
    
    Returns:  (ii=fileIdx, i=channelIdx, j=segmentIdx)
      +['data'][ii][i]['glob']['c']  coefs
      +['clst']['glob']['c']     coefs
                       ['ij']    link to [fileIdx, channelIdx, segmentIdx]
      +['data'][ii][i]['glob'][j]['decl'][bl|ml|tl|rng]['r']   reset
      +['data'][ii][i]['f0']['r']    f0 residual
    '''

    global f_log
    f_log = f_log_in

    myLog("DOING: styl glob")

    utils.check_var({'var': copa, 'env': 'copasul',
                     'spec': {'step': 'styl_glob', 'dom': 'glob'}})

    # re-empty (needed if styl is repatedly applied)
    copa['clst']['glob'] = {'c': [], 'ij': []}
    
    opt = copa['config']['styl']['glob']
    reg = copa['config']['styl']['register']
    if reg is None:
        # 'none' needed as key later
        reg = 'none'
    err_sum, N = 0, 0

    # over files
    file_ii = utils.sorted_keys(copa['data'])
    if silent:
        for ii in file_ii:
            copa, err_sum, N = styl_glob_file(copa, ii, opt, reg, err_sum, N)
    else:
        for ii in tqdm(file_ii , desc=f"styl glob"):
            copa, err_sum, N = styl_glob_file(copa, ii, opt, reg, err_sum, N)

    copa['clst']['glob']['c'] = np.array(copa['clst']['glob']['c'])
    copa['clst']['glob']['ij'] = np.array(copa['clst']['glob']['ij'])
            
    # error report
    if N > 0:
        copa['val']['styl']['glob']['err_prop'] = err_sum / N
    else:
        copa['val']['styl']['glob']['err_prop'] = np.nan
    return copa


def styl_glob_file(copa, ii, opt, reg, err_sum, N):

    ''' glob stylization for single file '''

    myFs = copa['config']['fs']
    
    # over channels
    for i in utils.sorted_keys(copa['data'][ii]):
        t = copa['data'][ii][i]['f0']['t']
        y = copa['data'][ii][i]['f0']['y']

        fstm = copa['data'][ii][i]['fsys']['aud']['stm']
        myLog(f"\tfile {fstm}, channel {i+1}")

        # residual
        r = np.zeros(len(y))

        # [[bl ml tl]...] medians over complete F0 contour (same length as y)
        med = styl_reg_med(y, opt)

        # over glob segments
        for j in utils.sorted_keys(copa['data'][ii][i]['glob']):
            gt = copa['data'][ii][i]['glob'][j]['t']

            # to for std_feat dur calculation; for interval tier input only
            # tor: time for declination rate calculation
            if len(copa['data'][ii][i]['glob'][j]['to']) > 1:
                to = copa['data'][ii][i]['glob'][j]['to'][0:2]
                tor = to
            else:
                to = np.array([])
                tor = gt[[0, 1]]
                
            yi = styl_yi(gt, myFs, y)
            ys = y[yi]
            df = styl_decl_fit(ys, opt, med[yi, :], tor)

            # utterance end prediction parameter set
            # consisting of line crossing and rate*duration parameters
            #       (the latter measuring actually observed declination)
            eou = styl_df_eou(df, myFs)

            # plot (if specified)
            copl.plot_main({'call': 'browse', 'state': 'online',
                            'fit': df, 'type': 'glob', 'set': 'decl', 'y': ys,
                            'infx': f"{ii}-{i}-{j}"}, copa['config'])

            # coefs + fitted line
            copa['data'][ii][i]['glob'][j]['decl'] = df

            # end-of-utterance prediction features derived from df
            copa['data'][ii][i]['glob'][j]['eou'] = eou

            # standard features
            copa['data'][ii][i]['glob'][j]['gnl'] = styl_std_feat(ys, opt, to)

            # coefs without intersection for clustering
            cc = list(df[reg]['c'][0:len(df[reg]['c'])-1])
            copa['clst']['glob']['c'].append(cc)
            
            # reference indices: [file, channel, segment]
            copa['clst']['glob']['ij'].append([ii, i, j])

            # bl|ml|tl|rng reset
            for x in utils.lists():
                if j == 0:
                    # first regline value ok, since basevalue subtracted before
                    py = 0
                else:
                    py = copa['data'][ii][i]['glob'][j-1]['decl'][x]['y'][-1]

                copa['data'][ii][i]['glob'][j]['decl'][x]['r'] = \
                            copa['data'][ii][i]['glob'][j]['decl'][x]['y'][0] - py

            # residual
            r[yi] = styl_residual(y[yi], df, reg)

            # error
            err_sum += df['err']
            N += 1

        # entire residual
        copa['data'][ii][i]['f0']['r'] = r
        
    return copa, err_sum, N


def styl_std_feat(y, opt, t=[]):

    '''
    standard f0 features mean, std
    
    Args:
      y: (np.array) f0 segment
      opt
      t = [] time on/offset (for original time values instead of sample counting)
    
    Returns:
      v['m']   arithmetic mean
       ['sd']  standard dev
       ['med'] median
       ['iqr'] inter quartile range
       ['max'] max
       ['maxpos'] relative position of maximum (normalized to [0 1]
       ['min'] min
       ['dur'] duration in sec
    '''

    if len(y) == 0:
        y = np.array([0])
    elif not utils.listType(y):
        y = np.array(y)

    if len(t) > 0:
        d = t[1] - t[0]
    else:
        d = utils.smp2sec(len(y), opt['fs'])
        
    return {'m': np.mean(y), 'sd': np.std(y), 'med': np.median(y),
            'iqr': np.percentile(y, 75) - np.percentile(y, 25),
            'max': np.max(y), 'min': np.min(y),
            'maxpos': (np.argmax(y) + 1) / len(y),
            'dur': d}


def styl_discont(t, y, a, b, opt, plotDict={}, med=np.array([]), caller='bnd'):

    '''

    discontinuity stylization
    for most variables: higher discontinuity is expressed by higher values
    exceptions: reset 'r', slope differences 'sd_*', for which later on abs values
       might be taken
    Args:
    t   time seq of whole file
    y   f0 seq of whole file
    a   decl dict of first segment returned by styl_decl_fit()
    b   decl dict of second segment
    opt: copa['config']['styl']['bnd']
    plotDict {'copa'-copa, 'infx' key} <{}>
    med: [[bl ml tl]...] medians
    caller: for evtl. later plotting
    
    Returns:
    bnd['p'] - pause length in sec
     ['t_on'] - onset time of post-boundary segment (evtl. end of pause)
     ['t_off'] - offset time of pre-boundary segment (evtl. start of pause)
     [myRegister][myDiscontinuity]
    myRegister :=
      bl - baseline
      ml - midline
      tl - topline
      rng - range
    myDiscontinuity :=
      r - reset: b[0] - a[-1]
      rms - rmsd(a.b, a+b)
      rms_pre - rmsd(A.b,a)
      rms_post - rmsd(a.B.b)
      sd_prepost - slope(b)-slope(a)
      sd_pre - slope(a.b)-slope(a)
      sd_post - slope(a.b)-slope(b)
      corrD - (1-corr(a+b, a.b))/2; correlation based distance, the higher the more discont;
                                ranging from 0 to 1
      corrD_pre - (1-corr(a, A.b))/2
      corrD_post - (1-corr(b, a.B))/2
      rmsR - rmse ratio rmse(a.b)/rmse(a+b); the higher the more discont
      rmsR_pre - rmse(A.b)/rmse(a)
      rmsR_post - rmse(a.B)/rmse(b)
      aicI - Akaike information criterion AIC increase of joint vs separate fit,
              AIC(a.b)-AIC(a+b); the higher the more discont
      aicI_pre - AIC(A.b)-AIC(a)
      aicI_post - AIC(a.B)-AIC(b)
      d_o - onset difference: a[0]-b[0]
      d_m - diff of mean values: mean(a)-mean(b)

    Notation:
    a, b: fits on pre-, post-boundary segments
    a.b: joint fit over segments a and b
    a+b: separate fits over segments a and b
    A.b: first part of a.b (a segment)
    a.B: second part of a.b (b segment)
    AIC for least squares calculated as explained in:
    https://en.wikipedia.org/wiki/Akaike_information_criterion#Comparison_with_least_squares
    3 model parameters per linear fit: intercept, slope, variance of (Gaussian) noise
    
    variables:
       wab: original f0 contour for segments a and b
       wa: original f0 contour for segment a
       wb: original f0 contour for segment b
       xab: register fit input (median sequence) for a+b
       xa: register fit input for a
       xb: register fit input for b
       yab: values of adjacent lines fitted separately on segments a+b
       ya: line fitted on segment a
       yb: line fitted on segment b
       zab: joint a.b fitted line values
       za: part of zab corresponding to segment a; A.b
       zb: part of zab corresponding to segment b; a.B

    '''

    # non-processable input marked by empty t
    if len(t) == 0:
        return discont_nan()

    # pause length, time
    bnd = {'p': b['to'][0] - a['to'][-1],
           't_off': a['to'][-1], 't_on': b['to'][0]}

    # joint segment
    ta = a['t']
    ia = styl_yi(ta, opt['fs'])
    tb = b['t']
    ib = styl_yi(tb, opt['fs'])

    # robust y-lim of indices
    ia, ib = sdw_robust(ia, ib, len(y)-1)

    # removing overlap
    if ta[-1] == tb[0]:
        ia = ia[0:len(ia)-1]

    # f0 segment
    ys = np.concatenate((y[ia], y[ib]))

    # corresponding median segment (only if available)
    if len(med) == len(y):
        meds = np.concatenate(
            (utils.lol(med[ia, :]), utils.lol(med[ib, :])), 0)
    else:
        meds = np.array([])

    # decl fit of joint segment
    df = styl_decl_fit(ys, opt, meds)

    # discontinuities for all register representations
    for x in utils.lists():

        # f0 arrays as introduced above (cf 'variables:')
        wa, wb, wab, xa, xb, xab, ya, yb, yab, za, zb, zab = bnd_segs(
            a, b, df, y, ys, ta, tb, ia, ib, x)

        # reset
        bnd[x] = {}
        bnd[x]['r'] = b['decl'][x]['y'][0] - a['decl'][x]['y'][-1]

        # onsets and means difference (downstep)
        bnd[x]['d_o'] = a['decl'][x]['y'][0] - b['decl'][x]['y'][0]
        bnd[x]['d_m'] = np.mean(a['decl'][x]['y']) - np.mean(b['decl'][x]['y'])

        # RMS between fitted lines
        bnd[x]['rms'] = utils.rmsd(yab, zab)
        bnd[x]['rms_pre'] = utils.rmsd(ya, za)
        bnd[x]['rms_post'] = utils.rmsd(yb, zb)

        # slope diffs
        sab, sa, sb = df[x]['c'][0], a['decl'][x]['c'][0], b['decl'][x]['c'][0]
        bnd[x]['sd_prepost'] = sa - sb
        bnd[x]['sd_pre'] = sab - sa
        bnd[x]['sd_post'] = sab - sb

        # distance derived from person r
        corrD = bnd_corrD(ya, yb, yab, za, zb, zab)
        for fld in corrD:
            bnd[x][fld] = corrD[fld]

        # fitting error ratios (in terms of RMSE) a.b / a+b
        rmsR = bnd_rmsR(xa, xb, xab, ya, yb, yab, za, zb, zab)
        for fld in rmsR:
            bnd[x][fld] = rmsR[fld]

        # AIC increases
        aicI = bnd_aicI(xa, xb, xab, ya, yb, yab, za, zb, zab)
        for fld in aicI:
            bnd[x][fld] = aicI[fld]

    # generate plot subdict for final plotting with copl.plot_main()
    #   .fit|y|t
    bnd['plot'] = bnd_plotObj(y, ia, ib, ta, tb, a, b, df, opt)

    # online plot
    if 'copa' in plotDict:
        copl.plot_main({'call': 'browse', 'state': 'online',
                        'type': 'complex', 'set': caller,
                        'fit': bnd['plot']['fit'],
                        'y': bnd['plot']['y'],
                        't': bnd['plot']['t'],
                        'infx': plotDict['infx']},
                       plotDict['copa']['config'])

    return bnd


def bnd_corrD(ya, yb, yab, za, zb, zab):

    '''
    distance d from pearson r ranging from 0 to 1
    d = (1 - r) / 2
    
    Args:
      all segments
    
    Returns:
      dict with dcorr, dcorr_pre, dcorr_post
    '''

    if len(yab) <= 1:
        corrD = 0.0
    elif np.min(yab) == np.max(yab) or np.min(zab) == np.max(zab):
        if np.min(yab) == np.max(yab) and np.min(zab) == np.max(zab):
            corrD = 0.0
        else:
            corrD = 1.0
    else:
        corrD = (1 - np.corrcoef(yab, zab)[0, 1]) / 2

    if len(ya) <= 1:
        corrD_pre = 0.0
    elif np.min(ya) == np.max(ya) or np.min(za) == np.max(za):
        if np.min(ya) == np.max(ya) and np.min(za) == np.max(za):
            corrD_pre = 0.0
        else:
            corrD_pre = 1.0
    else:
        corrD_pre = (1 - np.corrcoef(ya, za)[0, 1]) / 2

    if len(yb) <= 1:
        corrD_post = 0.0
    elif np.min(yb) == np.max(yb) or np.min(zb) == np.max(zb):
        if np.min(yb) == np.max(yb) and np.min(zb) == np.max(zb):
            corrD_post = 0.0
        else:
            corrD_post = 1.0
    else:
        corrD_post = (1 - np.corrcoef(yb, zb)[0, 1]) / 2
        
    return {'corrD': corrD,
            'corrD_pre': corrD_pre,
            'corrD_post': corrD_post}


def bnd_rmsR(xa, xb, xab, ya, yb, yab, za, zb, zab):

    '''
    fitting error ratios joint vs sep segments (in terms of RMSE)
    the higher the worse the fit on a.b compared to a+b
    
    Args:
      all segments
    
    Returns:
      dict with rmsR, rmsR_pre, rmsR_post
    '''

    rms_zab = utils.rmsd(zab, xab)
    rms_za = utils.rmsd(za, xa)
    rms_zb = utils.rmsd(zb, xb)
    rms_yab = utils.rmsd(yab, xab)
    rms_ya = utils.rmsd(ya, xa)
    rms_yb = utils.rmsd(yb, xb)
    return {'rmsR': utils.robust_div(rms_zab, rms_yab),
            'rmsR_pre': utils.robust_div(rms_za, rms_ya),
            'rmsR_post': utils.robust_div(rms_zb, rms_yb)}


def bnd_aicI(xa, xb, xab, ya, yb, yab, za, zb, zab):

    '''
    AIC increase joint a.b vs sep a+b segments
    the higher, the more discontinuity
    
    Args:
      all segments
    
    Returns:
      dict with aicR, aicR_pre, aicR_post
    '''

    # number of parameters per line fit:
    # intercept, slope, noise variance
    k = 3
    aic_zab = utils.aic_ls(xab, zab, k)
    aic_za = utils.aic_ls(xa, za, k)
    aic_zb = utils.aic_ls(xb, zb, k)
    aic_yab = utils.aic_ls(xab, yab, k * 2)
    aic_ya = utils.aic_ls(xa, ya, k)
    aic_yb = utils.aic_ls(xb, yb, k)

    return {'aicI': aic_zab - aic_yab,
            'aicI_pre': aic_za - aic_ya,
            'aicI_post': aic_zb - aic_yb}


def bnd_segs(a, b, df, y, ys, ta, tb, ia, ib, x):

    '''
    returns orig f0 and lines around boundaries for pre/post/joint segment
    see above for variable introduction
    
    Args:
       a: decl dict of pre-bnd seg
       b: decl dict of post-bnd seg
       df: decl dict of joint seg
       x: register id
    
    Returns:
       f0 segments xa, xb, xab ...
    '''

    # concat without overlap
    if ta[-1] == tb[0]:
        ya = a['decl'][x]['y'][0:len(a['decl'][x]['y'])-1]
        xa = a['decl'][x]['x'][0:len(a['decl'][x]['y'])-1]
    else:
        ya = a['decl'][x]['y']
        xa = a['decl'][x]['x']
    yb = b['decl'][x]['y']
    xb = b['decl'][x]['x']

    # joint segments' portions for rmsd, corr (pre, post, total)
    zab = df[x]['y']
    za = df[x]['y'][0:len(ya)]
    zb = df[x]['y'][len(ya):len(df[x]['y'])]

    yab = np.concatenate((ya, yb))
    xab = np.concatenate((xa, xb))

    # robust length adjustment
    yab, zab = utils.hal(yab, zab)
    ya, za = utils.hal(ya, za)
    yb, zb = utils.hal(yb, zb)
    # ... same for stylization input (adjust to length of z*)
    xab = utils.halx(xab, len(zab))
    xa = utils.halx(xa, len(za))
    xb = utils.halx(xb, len(zb))
    # ... same for original f0 contours (adjust to length of z*)
    wab = utils.halx(ys, len(zab))
    wa = utils.halx(y[ia], len(za))
    wb = utils.halx(y[ib], len(zb))
    
    return wa, wb, wab, xa, xb, xab, ya, yb, yab, za, zb, zab


def bnd_plotObj(y, ia, ib, ta, tb, a, b, df, opt):

    '''
    generates 'plot' subdict for bnd dict which can be used later for plotting
    '''
    
    sts = 1 / opt['fs']

    # pause zeros
    zz = []
    tz = ta[1] + sts

    while tz < tb[0]:
        zz.append(0)
        tz += sts
    zz = np.array(zz)
        
    yab = np.concatenate((y[ia], zz, y[ib]))
    tab = np.linspace(ta[0], tb[1], len(yab))
    ya = y[ia]
    yb = y[ib]
    ta = np.linspace(ta[0], ta[1], len(ya))
    tb = np.linspace(tb[0], tb[1], len(yb))

    return {'fit': {'a': a['decl'], 'b': b['decl'], 'ab': df},
            'y': {'a': ya, 'b': yb, 'ab': yab},
            't': {'a': ta, 'b': tb, 'ab': tab}}


def discont_nan():

    '''
    return all-NaN dict if discont input cannot be processed
    '''

    bnd = {}
    for x in ['p', 't_on', 't_off']:
        bnd[x] = np.nan
    for x in utils.lists('register'):
        bnd[x] = {}
        for y in utils.lists('bndfeat'):
            bnd[x][y] = np.nan
    return bnd


def styl_contains_nan(typ, z):

    '''
    returns list of paths through x to field with NaN
    
    Args:
      typ: subdict type
      z:   copa-subdict
    '''

    nanl = []
    if typ == 'bnd':
        for x in utils.lists('register'):
            for y in utils.lists('bndfeat'):
                if np.isnan(z[x][y]):
                    nanl.append(f"{x}.{y}")
    return nanl

def styl_residual(y, df, r):

    '''
    removal of global component from f0
    
    Args:
      y: (np.array) f0 contour
      df: (dict) of decl_fit
      r: (str) register type 'bl'|'ml'|'tl'|'rng'|'none'
    
    Returns:
      y-residual
    '''

    y = cp.deepcopy(y)

    # do nothing
    if r is None or r == 'none':
        return y

    # level subtraction of bl, ml, or tl
    elif r != 'rng':
        y = y - df[r]['y']

    # range normalization
    else:
        opt = {'mtd': 'minmax', 'rng': [0, 1]}
        for i in utils.idx(y):
            opt['min'] = df['bl']['y'][i]
            opt['max'] = df['tl']['y'][i]
            yo = y[i]
            y[i] = utils.nrm(y[i], opt)
    return y



def styl_df_eou(df, fs):

    '''
    calculate x, y coordinates of bl, ml, tl crossings
    
    Args:
      df: dict returned by styl_decl_fit()
    
    Returns:
      eou
        .tl_ml_cross_f0|t
        .tl_bl_cross_f0|t
        .ml_bl_cross_f0|t
        .{tl|ml|bl|rng}_drop
    '''

    dur = len(df['tn']) / fs
    eou = {}
    
    # drop parameters
    for r in utils.lists("register"):
        eou[f"{r}_drop"] = df[r]['rate'] * dur

    # line crossing coordinates
    #   (time and value automatically normalized according
    #    to ST and timenorm specs)
    for ra in ['tl', 'ml']:
        for rb in ['ml', 'bl']:
            if ra == rb:
                continue
            x, y = line_intersect(df[ra]['c'], df[rb]['c'])
            eou[f"{ra}_{rb}_cross_t"] = x
            eou[f"{ra}_{rb}_cross_f0"] = y

    return eou


def line_intersect(c1, c2):

    '''
    x and y coordinates of line intersection
    
    Args:
       c1: [slope intercept] of line 1
       c2: [slope intercept] of line 2
    
    Returns:
       x: x value (np.nan if lines are parallel)
       y: y value (np.nan if lines are parallel)
    '''

    a, c = c1[0], c1[1]
    b, d = c2[0], c2[1]
    if a == b:
        return np.nan, np.nan
    x = (d - c) / (a - b)
    y = a * x + c
    return x, y


def styl_decl_fit(y, opt, med=np.array([]), t=np.array([])):

    '''
    fit register level and range
    
    Args:
      y: (1-dim np.array) f0
      opt['decl_win'] window length for median calculation in sec
         ['prct']['bl']  <10>  value range for median calc
                 ['tl']  <90>
         ['nrm']['mtd']   normalization method
                ['range']
      med <[]> median values (in some contexts pre-calculated)
      t   <[]> time [on off] not provided in discontinuity context
    
    Returns:
      df['tn']   [normalizedTime]
        [myRegister]['c']   base/mid/topline/range coefs [slope intercept]
                    ['y']   line
                    ['rate'] rate (ST per sec, only if input t is not empty)
                    ['drop'] rate * len(tn); actual drop of f0 in segment
                    ['m']   mean value of line (resp. line dist)
           myRegister :=
             'bl' - baseline
             'ml' - midline
             'tl' - topline
             'rng' - range
        ['err'] - 1 if any line pair crossing, else 0
    '''


    med = utils.lol(med)

    # medians
    if len(med) != len(y):
        med = styl_reg_med(y, opt)

    # normalized time
    tn = utils.nrm_vec(utils.idx_a(len(y)), opt['nrm'])

    # fit midline
    mc = styl_polyfit(tn, med[:, 1], 1)
    mv = np.polyval(mc, tn)

    # interpolate over midline-crossing bl and tl medians
    med[:, 0] = styl_ml_cross(med[:, 0], mv, 'bl')
    med[:, 2] = styl_ml_cross(med[:, 2], mv, 'tl')

    # return dict
    df = {}
    for x in utils.lists():
        df[x] = {}
    df['tn'] = tn
    df['ml']['c'] = mc
    df['ml']['y'] = mv
    df['ml']['m'] = np.mean(df['ml']['y'])

    # get c for 'none' register
    df['none'] = {'c': styl_polyfit(tn, y, 1)}

    # fit base and topline
    df['bl']['c'] = styl_polyfit(tn, med[:, 0], 1)
    df['bl']['y'] = np.polyval(df['bl']['c'], tn)
    df['bl']['m'] = np.mean(df['bl']['y'])
    df['tl']['c'] = styl_polyfit(tn, med[:, 2], 1)
    df['tl']['y'] = np.polyval(df['tl']['c'], tn)
    df['tl']['m'] = np.mean(df['tl']['y'])

    # fit range (pointwise distance)
    df['rng']['c'] = styl_polyfit(tn, med[:, 2] - med[:, 0], 1)
    df['rng']['y'] = np.polyval(df['rng']['c'], tn)
    df['rng']['m'] = max(0.0, np.mean(df['rng']['y']))

    # declination rates
    if len(t) == 2:
        for x in ['bl', 'ml', 'tl', 'rng']:
            if len(df[x]['y']) == 0:
                continue
            if t[1] <= t[0]:
                df[x]['rate'] = 0
            else:
                df[x]['rate'] = (df[x]['y'][-1] - df[x]['y'][0]) / (t[1] - t[0])

    # erroneous line crossing
    if ((min(df['tl']['y'] - df['ml']['y']) < 0) or
        (min(df['tl']['y'] - df['bl']['y']) < 0) or
        (min(df['ml']['y'] - df['bl']['y']) < 0)):
        df['err'] = 1
    else:
        df['err'] = 0

    # styl input for later error calculation
    df['bl']['x'] = med[:, 0]
    df['ml']['x'] = med[:, 1]
    df['tl']['x'] = med[:, 2]
    df['rng']['x'] = med[:, 2] - med[:, 0]

    return df


def styl_reg_med(y, opt):

    '''
    returns base/mid/topline medians
    
    Args:
      y: n x 1 f0 vector
      opt: config['styl']['glob']
    
    Returns:
      med: n x 3 median sequence array
    '''

    # window [on off] on F0 indices
    dw = round(opt['decl_win'] * opt['fs'])

    # window alignment <center>|left|right
    if "align" in opt:
        al = opt["align"]
    else:
        al = "center"

    yw = utils.seq_windowing(win=dw, rng=[0, len(y)], align=al)

    # median sequence for base/mid/topline
    # med [[med_bl med_ml med_tl]...]
    med = []
    for i in range(len(yw)):
        ys = np.round(y[yw[i, 0]:yw[i, 1]], 8)
        qb, qt = np.percentile(ys, [opt['prct']['bl'], opt['prct']['tl']])
        qb = np.round(qb, 8)
        qt = np.round(qt, 8)
        if len(ys) <= 2:
            ybl = ys
            ytl = ys
        else:
            ybl = ys[ys <= qb]
            ytl = ys[ys >= qt]

        # fallbacks
        if len(ybl) == 0:
            qm = np.round(np.percentile(ys, 50), 8)
            ybl = ys[ys <= qm]
        if len(ytl) == 0:
            qm = np.round(np.percentile(ys, 50), 8)
            ytl = ys[ys >= qm]

        if len(ybl) > 0:
            med_bl = np.median(ybl)
        else:
            med_bl = np.nan
        if len(ytl) > 0:
            med_tl = np.median(ytl)
        else:
            med_tl = np.nan
        med_ml = np.median(ys)
        med.append([med_bl, med_ml, med_tl])

    med = np.array(med)
        
    # interpolate over nan
    for ii in ([0, 2]):
        xi = np.where(np.isnan(med[:, ii]))[0]
        if len(xi) > 0:
            xp = np.where(np.isfinite(med[:, ii]))[0]
            yp = med[xp, ii]
            yi = np.interp(xi, xp, yp)
            med[xi, ii] = yi
            
    return med


def styl_ml_cross(l, ml, typ):

    '''
    prevent from declination line crossing
    interpolate over medians for topline/baseline fit
    that are below/above the already fitted midline
    
    Args:
      l median sequence
      ml fitted midline
      typ: 'bl'|'tl' for base/topline
    
    Returns:
      l median sequence with linear interpolations
    '''

    if typ == 'bl':
        xi = np.where(ml - l <= 0)[0]
    else:
        xi = np.where(l - ml <= 0)[0]
    if len(xi) > 0:
        if typ == 'bl':
            xp = np.where(ml - l > 0)[0]
        else:
            xp = np.where(l - ml > 0)[0]
        if len(xp) > 0:
            yp = l[xp]
            yi = np.interp(xi, xp, yp)
            l[xi] = yi

    return l


def styl_loc(copa, f_log_in=None, silent=False):

    '''
    local segments
    
    Args:
      copa: (dict)
      f_log_in: (file handle)
      silent: (boolean) if True, tqdm bar is supressed
    
    Returns: (ii=fileIdx, i=channelIdx, j=segmentIdx)
      + ['data'][ii][i]['loc'][j]['acc'][*], see styl_loc_fit()
                                 ['gnl'][*], see styl_std_feat()
      + ['clst']['loc']['c']  coefs
                       ['ij'] link to [fileIdx, channelIdx, segmentIdx]
    '''

    global f_log
    f_log = f_log_in

    myLog("DOING: styl loc")

    utils.check_var({'var': copa, 'env': 'copasul',
                     'spec': {'step': 'styl', 'dom': 'loc'}})
    opt = copa['config']['styl']['loc']

    # re-empty (needed if styl is repatedly applied)
    copa['clst']['loc'] = {'c': [], 'ij': []}

    # rmse
    rms_sum = 0
    N = 0

    # over files
    file_ii = utils.sorted_keys(copa['data'])
    if silent:
        for ii in file_ii:
            copa, rms_sum, N = styl_loc_file(copa, ii, opt, rms_sum, N)
    else:
        for ii in tqdm(file_ii , desc=f"styl loc"):
            copa, rms_sum, N = styl_loc_file(copa, ii, opt, rms_sum, N)

    copa['clst']['loc']['c'] = np.array(copa['clst']['loc']['c'])
    copa['clst']['loc']['ij'] = np.array(copa['clst']['loc']['ij'])   
    copa['val']['styl']['loc']['rms_mean'] = rms_sum / N
    return copa


def styl_loc_file(copa, ii, opt, rms_sum, N):
    myFs = copa['config']['fs']

    # over channels
    for i in utils.sorted_keys(copa['data'][ii]):

        fstm = copa['data'][ii][i]['fsys']['aud']['stm']
        myLog(f"\tfile {fstm}, channel {i+1}")

        t = copa['data'][ii][i]['f0']['t']

        # add residual vector if not provided yet by styl_glob()
        if 'r' not in copa['data'][ii][i]['f0']:
            copa['data'][ii][i]['f0']['r'] = cp.deepcopy(
                copa['data'][ii][i]['f0']['y'])
        y = copa['data'][ii][i]['f0']['r']

        # over loc segments
        jj = utils.sorted_keys(copa['data'][ii][i]['loc'])
        for j in jj:
            lt = copa['data'][ii][i]['loc'][j]['t']
            if len(copa['data'][ii][i]['loc'][j]['to']) > 1:
                to = copa['data'][ii][i]['loc'][j]['to'][0:2]
            else:
                to = np.array([])
            tn = copa['data'][ii][i]['loc'][j]['tn']
            yi = styl_yi(lt[[0, 1]], myFs, y)
            ys = y[yi]
            lf = styl_loc_fit(lt, ys, opt)
            copa['data'][ii][i]['loc'][j]['acc'] = lf
            sf = styl_std_feat(ys, opt, to)
            
            # + nrm window
            yin = styl_yi(copa['data'][ii][i]['loc'][j]['tn'], myFs, y)

            # add normalization _nrm
            sf = styl_std_nrm(sf, y[yin], opt, tn)
            copa['data'][ii][i]['loc'][j]['gnl'] = sf

            cc = list(lf['c'])
            copa['clst']['loc']['c'].append(cc)
            copa['clst']['loc']['ij'].append([ii, i, j])

            # online plot: polyfit
            copl.plot_main({'call': 'browse', 'state': 'online', 'fit': lf, 'type': 'loc', 'set': 'acc',
                            'y': ys, 'infx': f"{ii}-{i}-{j}"}, copa['config'])

            # online plot: superpos (after having processed all locsegs in globseg)
            if ((j+1 not in jj) or
                (copa['data'][ii][i]['loc'][j+1]['ri'] >
                 copa['data'][ii][i]['loc'][j]['ri'])):
                ri = copa['data'][ii][i]['loc'][j]['ri']
                copl.plot_main({'call': 'browse', 'state': 'online', 'fit': copa, 'type': 'complex',
                                'set': 'superpos', 'i': [ii, i, ri], 'infx': f"{ii}-{i}-{ri}"},
                               copa['config'])

            # summed rmse
            rms_sum += utils.rmsd(lf['y'], ys)
            N += 1
            
    return copa, rms_sum, N


def styl_loc_fit(t, y, opt):

    '''
    polyfit of local segment
    
    Args:
      t: (list) time [onset offset nucleus]
      y: (np.array) f0
      opt: (dict)
    
    Returns:
      f: (dict)
       ['c']  polycoefs (descending order)
       ['tn'] normalized time
       ['y'] stylized f0
       ['rmsd'] root mean squared deviation (area) under polynomial contour
               (in case midline register was subtracted this is the area
                between contour and midline)
       ['rms'] root mean squared error between original and resynthesized contour
              for each polynomial order till opt['ord']; descending as for ['c']
    '''

    tt = np.linspace(t[0], t[1], len(y))

    # time normalization with zero placement
    t0 = np.round(t[2], 2)
    tn = utils.nrm_zero_set(tt, t0=t0, rng=opt['nrm']['rng'])
    f = {'tn': tn}
    
    f['c'] = styl_polyfit(tn, y, opt['ord'])
    f['y'] = np.polyval(f['c'], tn)
    f['rmsd'] = utils.rmsd(f['y'])

    # errors from 0..opt['ord'] stylization
    # (same order as in f['c'], i.e. descending)
    f['rms'] = [utils.rmsd(f['y'], y)]
    for o in np.linspace(opt['ord']-1, 0, opt['ord']):
        c = styl_polyfit(tn, y, o)
        r = np.polyval(c, tn)
        f['rms'].append(utils.rmsd(r, y))

    return f


def styl_loc_ext(copa, f_log_in=None, silent=False):

    '''
    extended local segment feature set
    decl and gestalt features
    
    Args:
      copa: (dict)
      f_log_in: (file handle)
      silent: (bool) if True, tqdm is suppressed
    
    Returns: (ii=fileIdx, i=channelIdx, j=segmentIdx)
      +[i]['loc'][j]['decl'][*], see styl_decl_fit()
                    ['gst'][*], see styl_gestalt()
    '''

    global f_log
    f_log = f_log_in

    myLog("DOING: styl loc ext")

    utils.check_var({'var': copa, 'env': 'copasul',
                     'spec': {'step': 'styl', 'dom': 'loc_ext', 'req': False,
                              'dep': ['loc', 'glob']}})

    gopt = copa['config']['styl']['glob']
    lopt = copa['config']['styl']['loc']

    # over files
    file_ii = utils.sorted_keys(copa['data'])
    if silent:
        for ii in file_ii:
            copa = styl_loc_ext_file(copa, ii, gopt, lopt)
    else:
        for ii in tqdm(file_ii , desc=f"styl loc_ext"):
            copa = styl_loc_ext_file(copa, ii, gopt, lopt)

    return copa


def styl_loc_ext_file(copa, ii, gopt, lopt):

    myFs = copa['config']['fs']

    # over channels
    for i in utils.sorted_keys(copa['data'][ii]):

        fstm = copa['data'][ii][i]['fsys']['aud']['stm']
        myLog(f"\tfile {fstm}, channel {i+1}")

        t = copa['data'][ii][i]['f0']['t']
        y = copa['data'][ii][i]['f0']['y']

        # [[bl ml tl]...] medians over complete F0 contour (same length as y)
        med = styl_reg_med(y, gopt)

        # decl, gestalt
        # over glob segments
        for j in utils.sorted_keys(copa['data'][ii][i]['glob']):
            dfg = copa['data'][ii][i]['glob'][j]['decl']
            gto = copa['data'][ii][i]['glob'][j]['t']

            # on:0.01:off
            gt = np.arange(gto[0], gto[1]+0.01, 0.01)

            # praat's f0 time starts with 0.1
            if gt[0] == 0:
                gt = gt[1:]
            lsi = copa['data'][ii][i]['glob'][j]['ri']
            
            # over contained loc segments
            for k in lsi:
                # t and y of local segment
                lt = copa['data'][ii][i]['loc'][k]['t']

                # indices in entire utterance
                yi = styl_yi(lt[[0, 1]], myFs)

                # orig time for rate calculation
                if len(copa['data'][ii][i]['loc'][k]['to']) > 1:
                    to = copa['data'][ii][i]['loc'][k]['to'][0:2]
                else:
                    to = lt[[0, 1]]

                # idx in copa['data'][ii][i]['glob'][j]['decl']['bl|ml|tl'][y]...
                ygi = utils.find_interval(gt, lt[[0, 1]])

                # adjust lengths
                yi, ygi = utils.hal(yi, ygi)

                if len(yi) < 4:
                    myLog(f"WARNING! styl_loc_ext_file(): file num {ii}, " \
                          f"channel num {i}, local segment time interval " \
                          f"{lt[0]} {lt[1]} too short for polynomial fitting.")

                ys = y[yi]
                dfl = styl_decl_fit(ys, gopt, med[yi, :], to)
                copa['data'][ii][i]['loc'][k]['decl'] = dfl
                copa['data'][ii][i]['loc'][k]['gst'] = styl_gestalt(
                    {'dfl': dfl, 'dfg': dfg, 'idx': ygi, 'opt': lopt, 'y': ys})
                copl.plot_main({'call': 'browse', 'state': 'online', 'fit': dfl, 'type': 'loc',
                                'set': 'decl', 'y': ys, 'infx': f"{ii}-{i}-{j}"},
                               copa['config'])
    return copa


def styl_gestalt(obj):

    '''
    measures deviation of locseg declination from corresponding stretch of globseg
    
    Args:
      obj['dfl'] decl_fit dict of locseg
         ['dfg'] decl_fit dict of globseg
         ['idx'] y indices in globseg spanned by locseg
         ['y']   orig f0 values in locseg
         ['opt']['nrm']['mtd'|'rng'] normalization specs
                ['ord']   polyorder for residual
    
    Returns:
      gst[myRegister]['rms'] RMSD between corresponding lines in globseg and locseg
                     ['d'] mean distance between corresponding lines in globseg and locseg. Directed as opposed to 'rms'
                     ['sd']  slope diff
                     ['d_init'] y[0] diff
                     ['d_fin']  y[-1] diff
         ['residual'][myRegister]['c'] polycoefs of f0 residual

    myRegister := bl|ml|tl|rng
    residual: ml, bl, tl subtraction, pointwise rng [0 1] normalization

    REMARKS:
      - all diffs: locseg-globseg
      - sd is taken from (y_off-y_on)/lng since poly coefs
        of globseg and locseg decl are derived from different
        time normalizations. That is, slopediff actually is a rate difference!
      - no alignment in residual stylization (i.e. no defined center := 0)
    '''

    (dfl, dfg_in, idx, opt, y) = (
        obj['dfl'], obj['dfg'], obj['idx'], obj['opt'], obj['y'])
    dcl = {}

    # preps for residual calculation
    dfg = {}
    for x in utils.lists():
        dfg[x] = {}
        dfg[x]['y'] = dfg_in[x]['y'][idx]
    l = len(idx)
    dcl['residual'] = {}

    # gestalt features
    for x in utils.lists():
        dcl[x] = {}
        yl = dfl[x]['y']
        yg = dfg[x]['y']

        # rms
        dcl[x]['rms'] = utils.rmsd(yl, yg)

        # mean distance d
        dcl[x]['d'] = np.mean(yl - yg)

        # slope diff sd
        dcl[x]['sd'] = ((yl[-1] - yl[0]) - (yg[-1] - yg[0])) / l

        # d_init, d_fin
        dcl[x]['d_init'] = yl[0] - yg[0]
        dcl[x]['d_fin'] = yl[-1] - yg[-1]

        # residual
        r = styl_residual(y, dfg, x)
        t = utils.nrm_vec(utils.idx_a(l), opt['nrm'])
        dcl['residual'][x] = {}
        dcl['residual'][x]['c'] = styl_polyfit(t, r, opt['ord'])

    return dcl


def styl_polyfit(x, y, o):

    '''
    robust polyfit
    
    Args:
    x: (np.array)
    y: (np.array)
    o: (int) order

    Returns:
    c: (np.array) of coefs, highest power first

    '''

    o = int(o)
     
    if len(x) == 0:
        return np.zeros(o + 1)
    if len(x) <= o:
        c = np.append(np.zeros(o), np.mean(y))
        return c
    try:
        c = np.polyfit(x, y, o)
    except:
        c = np.zeros(o+1)
    return c


def styl_bnd(copa, f_log_in=None):

    '''
    boundary stylization
    
    Args:
      copa
    
    Returns: (ii=fileIdx, i=channelIdx, j=tierIdx, k=segmentIdx)
      + ['data'][ii][i]['bnd'][j][k]['decl']; see styl_decl_fit()
      + ['data'][ii][i]['bnd'][j][k][myWindowing]; see styl_discount()
             for non-final segments only

    myWindowing :=
      std - standard, i.e. neighboring segments
      win - symmetric window around segment boundary
            (using 'tn': norm window limited by current chunk, see pp_t2i())
      trend - from file onset to boundary, and from boundary to file offset
            (using 'tt': trend window [chunkOn timePoint chunkOff], see p_t2i())
    '''

    global f_log
    f_log = f_log_in

    myLog("DOING: styl bnd")

    utils.check_var({'var': copa, 'env': 'copasul',
                     'spec': {'step': 'styl', 'dom': 'bnd'}})

    navi = copa['config']['navigate']
    opt = copa['config']['styl']['bnd']

    # over files
    file_ii = utils.sorted_keys(copa['data'])
    for ii in tqdm(file_ii , desc=f"styl bnd"):
        copa = styl_bnd_file(copa, ii, navi, opt)
    return copa


def styl_bnd_file(copa, ii, navi, opt):

    '''
    called by styl_bnd for file-wise processing
    
    Args:
      copa
      ii: fileIdx
      navi: navigation dict
      opt: copa['config']['styl']['bnd']
    
    Returns:
      copa
        +['bnd'][j][k]['decl']
                      ['std']
                      ['win']
                      ['trend']
            j - tier idx, k - segment idx
            bnd features refer to boundary that FOLLOWS the segment k
    '''


    myFs = copa['config']['fs']

    # over channels
    for i in utils.sorted_keys(copa['data'][ii]):
        t = copa['data'][ii][i]['f0']['t']

        fstm = copa['data'][ii][i]['fsys']['aud']['stm']

        myLog(f"\tfile {fstm}, channel {i+1}")

        # which f0 contour to take (+/- residual)
        if opt['residual']:
            y = copa['data'][ii][i]['f0']['r']
        else:
            y = copa['data'][ii][i]['f0']['y']
            
        # [[bl ml tl]...] medians over complete F0 contour (same length as y)
        med = styl_reg_med(y, opt)

        # over tiers
        for j in utils.sorted_keys(copa['data'][ii][i]['bnd']):

            # over segments
            nk = utils.sorted_keys(copa['data'][ii][i]['bnd'][j])
            for k in nk:
                gt = copa['data'][ii][i]['bnd'][j][k]['t']
                yi = styl_yi(gt, myFs, y)
                ys = y[yi]

                # decl fit
                df = styl_decl_fit(ys, opt, med[yi, :])
                copa['data'][ii][i]['bnd'][j][k]['decl'] = df
                if k < 1:
                    continue
                a = copa['data'][ii][i]['bnd'][j][k-1]
                b = copa['data'][ii][i]['bnd'][j][k]

                # plotting dict
                tn = copa['data'][ii][i]['bnd'][j][k]['tier']
                po = {'copa': copa, 'infx': f"{ii}-{i}-{tn}-{k-1}-std"}

                # discont
                copa['data'][ii][i]['bnd'][j][k-1]['std'] = styl_discont(
                    t, y, a, b, opt, po, med, 'bnd')
                
                # alternative bnd windows
                # - trend
                if navi['do_styl_bnd_trend'] == True:
                    tn = copa['data'][ii][i]['bnd'][j][k]['tier']
                    po['infx'] = f"{ii}-{i}-{tn}-{k-1}-trend"
                    copa['data'][ii][i]['bnd'][j][k-1]['trend'] = styl_discont_wrapper(
                        copa['data'][ii][i]['bnd'][j][k]['tt'], t, y, opt, po, med, 'bnd_trend')

                # - fixed length window
                if navi['do_styl_bnd_win'] == True:
                    tn = copa['data'][ii][i]['bnd'][j][k]['tier']
                    po['infx'] = f"{ii}-{i}-{tn}-{k-1}-win"
                    copa['data'][ii][i]['bnd'][j][k-1]['win'] = styl_discont_wrapper(
                        copa['data'][ii][i]['bnd'][j][k]['tn'], t, y, opt, po, med, 'bnd_win')

    return copa


def styl_discont_wrapper(tw, t, y, opt, po={}, med=np.array([]), caller='bnd'):

    '''
    wrapper around styl_discont()
    
    Args:
      tw: time array [on joint1 (joint2) off]
          if joint2 is missing, i.e. 3-element array: joint2=joint1
          discont is calculated between [on joint1] and [joint2 off]
      t: all time array
      y: all f0 array
      opt: copa['config']
      po: plotting options
      med: [[bl ml tl]...], same length as y (will be calculated in this function if not provided)
      caller: infx for evtl. plotting
    
    Returns:
      dict from styl_discont()
    
    REMARKS:
       tw: ['bnd'][j]['tn'|'tt'] can be passed on as is ('win', 'trend' discont)
           ['bnd'][j]['t'] + ['bnd'][j+1]['t'] needs to be concatenated ('std' discont)
    '''


    myFs = opt['fs']

    # called with entire opt dict
    if 'styl' in opt:
        opt = cp.deepcopy(opt['styl']['bnd'])

    # f0 medians [[bl ml tl]...], same length as y
    if len(med) > 0:
        med = utils.lol(med)

    if len(med) != len(y):
        med = styl_reg_med(y, opt)

    # segments
    if len(tw) == 4:
        on1, off1, on2, off2 = tw[0], tw[1], tw[2], tw[3]
    else:
        on1, off1, on2, off2 = tw[0], tw[1], tw[1], tw[2]

    # segments too short or missing
    # (e.g. if cross_chunk is False, and seg 1 ends at or seg2 starts at chunk bnd)
    if on1 == off1 or on2 == off2:
        return styl_discont([], [], {}, {}, {}, {}, [], '')

    i1 = styl_yi([on1, off1], myFs)
    i2 = styl_yi([on2, off2], myFs)

    # robust extension
    i1, i2 = sdw_robust(i1, i2, len(y)-1)
    ys1 = y[i1]
    ys2 = y[i2]
    med1 = med[i1, :]
    med2 = med[i2, :]

    # decl fits
    a = {'decl': styl_decl_fit(ys1, opt, med1),
         't': [on1, off1], 'to': [on1, off1]}
    b = {'decl': styl_decl_fit(ys2, opt, med2),
         't': [on2, off2], 'to': [on2, off2]}

    # styl discontinuity
    return styl_discont(t, y, a, b, opt, po, med, caller)


def sdw_robust(i1, i2, u):

    '''
    for discont styl, padds indices so that index arrays have minlength 2
    
    Args:
      i1, i2: (np.arrays) index arrays of adjacent segments
      u: (index) upper index limit (incl; lower is 0)

    Returns:
      i1, i2: (np.arrays) corrected indices
    '''

    i1 = i1[i1 <= u]
    i2 = i2[i2 <= u]
    
    if len(i1) == 0:
        if len(i2) == 0:
            i1 = [0]
        else:
            i1 = np.array([max(0, i2[0] - 2)])

    if len(i2) == 0:
        i2 = np.array([i1[-1]])

    while len(i1) < 2 and i1[-1] < u:
        i1 = np.append(i1, i1[-1] + 1)

    while len(i2) < 2 and i2[-1] < u:
        i2 = np.append(i2, i2[-1] + 1)

    return i1, i2


def styl_rhy(copa, typ, f_log_in=None):

    '''
    speech rhythm
    
    Args:
      copa
      typ: 'en'|'f0'
    
    Returns:
      +['rhy'], see output of styl_speech_rhythm()
    '''

    global f_log
    f_log = f_log_in

    myLog(f"DOING: styl rhy {typ}")

    fld = f"rhy_{typ}"
    utils.check_var({'var': copa, 'env': 'copasul',
                     'spec': {'step': 'styl', 'dom': fld}})
    opt = cp.deepcopy(copa['config']['styl'][fld])
    opt['type'] = typ
    opt['plot'] = copa['config']['plot']
    opt['fsys'] = copa['config']['fsys']
    opt['navigate'] = copa['config']['navigate']

    # over files
    file_ii = utils.sorted_keys(copa['data'])
    for ii in tqdm(file_ii , desc=f"styl {fld}"):
        copa = styl_rhy_file(copa, ii, fld, opt)

    return copa


def styl_rhy_file(copa, ii, fld, opt):

    '''
    
    Args:
      copa
      ii fileIdx
      fld in copa 'rhy_en|f0'
      opt augmented deepcopy of copa[config]
    '''


    # over channels
    for i in utils.sorted_keys(copa['data'][ii]):

        fstm = copa['data'][ii][i]['fsys']['aud']['stm']
        myLog(f"\tfile {fstm}, channel {i+1}")

        if opt['type'] == 'f0':
            # f0
            t = copa['data'][ii][i]['f0']['t']
            y = copa['data'][ii][i]['f0']['y']
            myFs = copa['config']['fs']
        else:
            # energy
            fp = copa['data'][ii][i]['fsys']['aud']
            f = f"{fp['dir']}/{fp['stm']}.{fp['ext']}"
            y, fs_sig = cosp.wavread(f)

            if np.ndim(y) > 1:
                # select channel
                y = y[:, i]
                
            opt['fs'] = fs_sig
            t = utils.smp2sec(utils.idx_seg(1, len(y)), fs_sig, 0)
            myFs = fs_sig

            # rescale to max to make different recordings comparable
            if opt['sig']['scale']:
                amax = max(abs(y))
                fac = max(1, (1 / amax) - 1)
                y *= fac

        # file wide
        r = copa['data'][ii][i]['rate']
        
        # for plotting
        opt['infx'] = f"{ii}-{i}-file"
        copa['data'][ii][i][f"{fld}_file"] = styl_speech_rhythm(
            y, r, opt, copa['config'])
        
        # over tiers
        for j in utils.sorted_keys(copa['data'][ii][i][fld]):

            # over segments
            nk = utils.sorted_keys(copa['data'][ii][i][fld][j])
            for k in nk:
                gt = copa['data'][ii][i][fld][j][k]['t']
                yi = styl_yi(gt, myFs, y)
                ys = y[yi]
                r = copa['data'][ii][i][fld][j][k]['rate']

                # for plotting
                tn = copa['data'][ii][i][fld][j][k]['tier']
                opt['infx'] = f"{ii}-{i}-{tn}-{k}"
                copa['data'][ii][i][fld][j][k]['rhy'] = styl_speech_rhythm(
                    ys, r, opt, copa['config'])
                
    return copa



def styl_speech_rhythm(y, r={}, opt=None, copaConfig=None):

    '''
    DCT-related features on energy or f0 contours
    
    Args:
    y :(np.array) f0 or amplitude sequence (raw signal, NOT energy contour)
    r: (dict)
      r[myDomain] <{}> rate of myDomain (e.g. 'syl', 'ag', 'ip')]
    opt: (dict)
      opt['type'] - signal type: 'f0' or 'en'
      opt['fs']  - sample frequency
      opt['sig']           - options to extract energy contour
                             (relevant only for opt['type']=='en')
                ['wintyp'] - <'none'>, any type supported by
                             scipy.signal.get_window()
                ['winparam'] - <''> additionally needed window parameters,
                             scalar, string, list ...
                ['sts']    - stepsize of moving window
                ['win']    - window length
         ['rhy']           - DCT options
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
                ['rb'] - <1> frequency catch band to measure influence
                         of rate of events in rate dict R in DCT
                         e.g. r['syl']=4, opt['rb']=1
                         'syl' influence on DCT: summed abs coefs of 3,4,5 Hz
    copaConfig: (dict)
    
    Returns:
    rhy: (dict)
      rhy['c_orig'] all coefs
         ['f_orig'] all freq
         ['c'] coefs with freq between lb and ub
         ['f'] freq between lb and ub
         ['i'] indices of 'c' in 'c_orig'
         ['sm'] ndarray spectral moments
         ['m'] weighted coef mean
         ['sd'] weighted coef std
         ['cbin'] ndarray, summed abs coefs in freq bins between lb and ub
         ['fbin'] ndarray, corresponding frequencies
         ['wgt'][myDomain]['mae'] - mean absolute error between
                                    IDCT of coefs around resp rate
                                    and IDCT of coefs between 'lb' and 'ub'
                          ['prop'] - proportion of coefs around
                                    resp rate relative to coef sum
                          ['rate'] - rate in analysed domain
                          ['dgm'] dist to glob max in dct !
                          ['dlm'] dist to loc max in dct !
         ['dur'] - segment duration (in sec)
         ['f_max'] - freq of max amplitude
         ['n_peak'] - number of peaks in DCT spectrum

    '''

    if opt is None:
        opt = {}
    if copaConfig is None:
        copaConfig = {}
        
    err = 0
    dflt = {}
    dflt['sig'] = {'sts': 0.01, 'win': 0.05,
                   'wintyp': 'hamming', 'winparam': None}
    dflt['rhy'] = {'wintyp': 'kaiser', 'winparam': 1, 'nsm': 3,
                   'rmo': True, 'lb': 0, 'ub': 0, 'wgt': {'rb': 1}}
    for x in list(dflt.keys()):
        if type(dflt[x]) is not dict:
            if x not in opt:
                opt[x] = dflt[x]
        else:
            opt[x] = utils.opt_default(opt[x], dflt[x])

    for x in ['type', 'fs']:
        if x not in opt:
            myLog(f'ERROR! speech_rhythm(): opt must contain {x}')
            err = 1
        else:
            opt['sig'][x] = opt[x]
            opt['rhy'][x] = opt[x]
    if err == 1:
        myLog('ERROR! in speech rhythm extraction', True)

    # adjust sample rate for type='en' (energy values per sec)
    if opt['type'] == 'en':
        opt['rhy']['fs'] = int(1 / opt['sig']['sts'])
        
    # energy contour
    if opt['type'] == 'en':
        y = cosp.sig_energy(y, opt['sig'])

    # dct features
    rhy = cosp.dct_wrapper(y, opt['rhy'])

    # number of local maxima
    rhy['n_peak'] = len(rhy['f_lmax'])

    # duration
    rhy['dur'] = utils.smp2sec(len(y), opt['rhy']['fs'])

    # domain weight features + ['wgt'][myDomain]['prop'|'mae'|'rate']
    rhy = rhy_sub(y, r, rhy, opt)

    copl.plot_main({'call': 'browse', 'state': 'online', 'fit': rhy,
                    'type': f"rhy_{opt['type']}", 'set': 'rhy',
                    'infx': opt['infx']}, copaConfig)

    return rhy

    
def rhy_sub(y, r, rhy, opt):

    '''
    quantifying influence of events with specific rates on DCT
    
    Args:
      y - ndarray contour
      r - dict {myEventId:myRate}
      rhy - output dict of dct_wrapper
      opt - dict, see speech_rhythm()
    
    Returns:
      rhy
        +['wgt'][myDomain]['mae']
                          ['prop']
                          ['rate']
                          ['dgm'] dist to glob max in dct
                          ['dlm'] dist to loc max in dct
    '''

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
    yr = cosp.idct_bp(rhy['c_orig'], rhy['i'])

    rhy['wgt'] = {}

    # over rate keys
    for x in r:

        # distances to global and nearest local peak
        dg = r[x] - gpf
        dl = np.exp(10)
        for z in lpf:
            dd = r[x] - z
            if abs(dd) < dl:
                dl = dd

        # define catch band around respective event rate of x
        lb = r[x] - rb
        ub = r[x] + rb

        # 'prop': DCT coef abs ampl around rate
        if len(ac) == 0:
            j = np.array([])
            prp = 0
        else:
            j = np.where((rhy['f'] >= lb) & (rhy['f'] <= ub))[0]
            prp = sum(ac[j]) / sac

        # 'mae': mean abs error between rhy[c] IDCT and IDCT of
        #        coefs between event-dependent lb and ub
        yrx = cosp.idct_bp(rhy['c_orig'], j)
        ae = utils.mae(yr, yrx)

        rhy['wgt'][x] = {'mae': ae, 'prop': prp, 'rate': r[x],
                         'dlm': dl, 'dgm': dg}

    return rhy


def myLog(msg, e=False):

    '''
    log file output (if filehandle), else terminal output
    
    Args:
      msg: (str) log message
      e: (boolean) if True, do exit
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
       
