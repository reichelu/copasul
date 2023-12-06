import copy as cp
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import scipy.fftpack as sf
import scipy.io.wavfile as sio
import scipy.signal as sis
import shutil as sh
import sklearn.cluster as sc
import sklearn.preprocessing as sp
import sys
from tqdm import tqdm

import copasul.copasul_clst as cocl
import copasul.copasul_init as coin
import copasul.copasul_preproc as copp
import copasul.copasul_sigproc as cosp
import copasul.copasul_styl as cost
import copasul.copasul_utils as utils


'''
 common pipeline for filewise and batch processing for IP and accent extraction
 aug_glob/aug_batch('glob',...):
 1. feature matrix, weights, seed vectors, time info
   fv, wgt, tc, is0, is1, i_nan, t, to, pt, pto, fto = \
                aug_glob_fv(ty, y, annot, opt, i, fstm, f, lng, spec)
 2. clustering
   c, cntr, wgt = aug_cntr_wrapper('glob', fv, tc, wgt, opt, is0, is1, i_nan, meas)
 3. segmentation based on class vector c
   ip = aug_glob_seg(c, tc, t, to, pt, pto, fto, opt)
 
 aug_loc/aug_batch('loc',...):
 1. feature matrix, weights, seed vectors, time info
   fv, wgt, tc, is0, is1, i_nan, to, ato, do_postsel = \
                aug_loc_fv(ty, y, bv, annot, opt, i, fstm, f, lng, add_mat, spec)
 2. clustering
   c, cntr, wgt = aug_cntr_wrapper('loc', fv, tc, wgt, opt, is0, is1, i_nan, meas)
 3. accentuation based on class vector c, and further selection within accent groups/words
   acc = aug_loc_acc(c, fv, wgt, to, ato, do_postsel, cntr)

 augments TextGrids by tiers: myChunkTierName_myChannelIdx and
  mySylTierName_myChannelIdx

 not stored in copa dict, since relation to f0 file index which is
 the key in subdict copa['data'] is unknown
''' 

def aug_main(copa, f_log_in):

    ''' augmentation main function

    Args:
    copa: (nested dict)
    f_log_in: (handle) of log file
    
    outputs/modifies TextGrids
    
    '''

    global f_log
    f_log = f_log_in
    opt = copa['config']

    myLog("DOING: augment annotations ...")

    # wav files
    ff_wav = utils.file_collector(opt['fsys']['aud']['dir'],
                                  opt['fsys']['aud']['ext'])
    # annot files
    ff_annot = utils.file_collector(opt['fsys']['annot']['dir'],
                                    opt['fsys']['annot']['ext'])
    # f0 files
    ff_f0 = utils.file_collector(opt['fsys']['f0']['dir'],
                                 opt['fsys']['f0']['ext'])

    # additional material for acc extraction
    add_mat = {'ff': copp.pp_file_collector(opt)}

    # over wav files
    timeStamp = utils.isotime()
    timeStamp = re.sub(r"[:\.]", "-", timeStamp)
    for i in tqdm(utils.idx_a(len(ff_wav)), desc="augmentation"):

        # add to existing file or generation from scratch
        annot, fstm, fo = aug_annot_init(ff_annot, ff_wav, i, opt, timeStamp)
        add_mat['fi'] = i

        # extending annotation by chunk, syl tiers f.a. channel
        # if unit=='file' glob and loc tiers are also added
        annot = aug_sub(ff_wav[i], ff_f0[i], annot, fstm, opt, add_mat)

        utils.output_wrapper(annot, fo, opt['fsys']['annot']['typ'])

    # boundaries batch clustering
    if opt['navigate']['do_augment_glob'] and opt['augment']['glob']['unit'] == 'batch':
        myLog("DOING: boundaries batch clustering ...")
        aug_batch('glob', ff_wav, ff_annot, ff_f0, opt, add_mat)

    # accents batch clustering
    if opt['navigate']['do_augment_loc'] and opt['augment']['loc']['unit'] == 'batch':
        myLog("DOING: accents batch clustering")
        aug_batch('loc', ff_wav, ff_annot, ff_f0, opt, add_mat)

    return


def aug_batch(dom, ff_wav, ff_annot, ff_f0, opt, add_mat):

    '''

    batch clustering
    
    Args:
    dom: (str) domain 'glob'|'loc'
    ff_wav: (list) of wav file names
    ff_annot: (list) of annotation file names
    ff_f0: (list) of f0 file names
    opt: (dict) copa['config']
    
    
    adds tiers to annotation files
    '''
    
    print(f'Batch clustering {dom}')
    
    # feature vectors for entire corpus: ['batch'] subdict
    #    and for each file: ['file'] subdict
    ps = aug_batch_ps(dom, ff_wav, ff_annot, ff_f0, opt, add_mat)

    # clustering over all data
    d = ps['batch']

    c, cntr, wgt = aug_cntr_wrapper(dom, d['fv'], d['tc'], d['wgt'], opt,
                                    d['is0'], d['is1'], d['i_nan'],
                                    opt['augment'][dom]['measure'])

    # export featvecs and time stamps to sto files, e.g. for supervised learning
    if 'export' in opt['fsys']['augment'][dom] and len(opt['fsys']['augment'][dom]['export']) > 0:
        utils.output_wrapper(
            d, f"{opt['fsys']['export']['dir']}/{opt['fsys']['augment'][dom]['export']}.pickle",
            'pickle')

    # distributing c to files
    ps = aug_batch_distrib(ps, c, wgt)
    for ii in utils.sorted_keys(ps['file']):
        annot, fstm, fo = aug_annot_init(ff_annot, ff_wav, ii, opt)
        for i in utils.sorted_keys(ps['file'][ii]):
            psf = ps['file'][ii][i]
            if dom == 'glob':
                d = aug_glob_seg(psf['c'], psf['tc'], psf['t'], psf['to'],
                                 psf['pt'], psf['pto'], psf['fto'], opt)
            else:
                d = aug_loc_acc(psf['c'], psf['fv'], psf['wgt'], psf['to'], psf['ato'],
                                psf['do_postsel'], cntr)
                if len(d) == 0 and opt["augment"]["loc"]["force"]:
                    d = force_acc(ff_wav[ii])
            annot = aug_annot_upd(dom, annot, {'t': d}, i, opt, psf['lng'])
        utils.output_wrapper(annot, fo, opt['fsys']['annot']['typ'])
    return


def aug_batch_distrib(ps, c, wgt):

    '''
    distributes batch class vector over files/channels

    Args:
    ps: (dict) with prosStruct
    c: (list) class vector
    wgt: (list) feature weights

    Returns:
    ps: (dict)
      ps['file'] + ['c']: class-subvector for file/channel
                 + ['wgt']: feat weight vector
                 + ['fv']-replacement (since pho-features might have been added)
                     filewise fv are needed for accent post-selection

    '''
    
    ps['batch']['wgt'] = wgt
    fv = ps['batch']['fv']
    for ii in utils.sorted_keys(ps['file']):
        for i in utils.sorted_keys(ps['file'][ii]):
            ps['file'][ii][i]['c'] = c[ps['file'][ii][i]['ri']]
            ps['file'][ii][i]['fv'] = fv[ps['file'][ii][i]['ri']]
            ps['file'][ii][i]['wgt'] = wgt
    return ps


def aug_batch_ps(dom, ff_wav, ff_annot, ff_f0, opt, add_mat):

    '''
    returns prosodic struct dict containing batch and file-related info

    Args:
      dom: (str) domain 'glob'|'loc'
      ff_wav: (list) wav file list
      ff_annot: (list) annotation file list
      ff_f0: (list) f0 file list
      opt: (dict) copa['config']
      add_mat: (various) misc additional material, customized online

    Returns:
      ps: (dict)
        ps['batch']['fv'|'tc'|'is0'|'is1'|'ij']   matrix or vectors
                                 in sync: fv, tc, ij [fileIdx, channelIdx]
          ['file']['fi']['ci']  fileIdx x channelIdx: rowIdx in data['fv']
                              ['is0'] set of 0-seed indices
                              ['is1'] set of 1-seed indices
                              ['tc'] time stamps of boundaries or accent candidates
                              ['pt'] parent segments [[on off]...]
                              ['pto'] pt unrounded
                              ['t'] for acc only
                              ['to']  t unrounded
                              ['lng'] signal length
                              ['ri'] row indices in featmat
    '''

    # batch prosstruct container
    ps = aug_ps_init()

    # onset idx (featmatRow-file/channel assignment)
    oi = 0

    add_pho, opt, wgt_pho = aug_do_add_pho(dom, opt)
    pho = {'dur': [], 'lab': []}

    # over f0 files
    for ii in utils.idx_a(len(ff_f0)):
        
        f = ff_wav[ii]
        aug_spec, fs, s, lng, f0_dat, ncol = aug_sig(f, ff_f0[ii], opt)

        # existing file or generation from scratch
        annot, fstm, fo = aug_annot_init(ff_annot, ff_wav, ii, opt)
        add_mat['fi'] = ii
        
        # over channels
        for i in range(ncol):
            
            # Signal preprocessing
            # if dom=='loc':
            #    if ncol>1:
            #        y = utils.sig_preproc(s[:,i])
            #    else:
            #        y = utils.sig_preproc(s)

            # F0 preprocessing
            f0, f0_ut = copp.pp_read_f0(f0_dat, opt['fsys']['f0'], i)
            f0, t, z, bv = copp.pp_f0_preproc(f0, np.round(lng, 2), opt)

            # phrase/accent extraction
            if dom == 'glob':
                fv, wgt, tc, is0, is1, i_nan, t, to, pt, pto, fto = aug_glob_fv(t, z, annot, opt,
                                                                                i, fstm, f, lng,
                                                                                aug_spec['glob'])
                # domain dependent input
                psf = {'t': t, 'pt': pt, 'pto': pto, 'to': to, 'fto': fto}
            else:
                fv, wgt, tc, is0, is1, i_nan, to, ato, do_postsel = aug_loc_fv(t, z, bv, annot, opt,
                                                                               i, fstm, f, lng,
                                                                               add_mat, aug_spec['loc'])
                psf = {'do_postsel': do_postsel, 'ato': ato, 'to': to}
                
            if add_pho:
                pho = aug_pho(dom, pho, annot, tc, i, fstm, opt)
                
            ps = aug_batch_ps_upd(ps, dom, ii, i, fv, wgt,
                                  tc, is0, is1, i_nan, lng, psf, oi)
            
            oi = len(ps['batch']['fv'])
            
    # normalize vowel durations
    if add_pho:
        
        ndur = aug_pho_nrm(pho)

        # merge pho and fv, add weight
        # reinsert weight
        opt['augment'][dom]['wgt']['pho'] = wgt_pho
        ps = aug_ps_pho_mrg(ps, ndur, opt['augment'][dom])
        
    return ps


def aug_pho(dom, pho, annot, tc, ci, stm, opt):

    '''
    extract vowel length
    Args:
     dom: (str) 'glob'|'loc'
     pho: (dict) phonem dict to be updated, initialized in aug_batch()
     annot: (dict) annotation dict
     tc: (np.array) potential prosodic event time stamps where to
         look for vowel length
     ci: (int) channel index
     stm: (str) annotation file stem
     opt: (dict) copa['config']
    Returns:
     pho: (dict) updated by duration, label at tc stamps
          .lab: label array
          .dur: duration array
    '''

    # vowel pattern
    vow = r"{}".format(opt['fsys']['pho']['vow'])

    # pho-tier(s) in list
    tn = copp.pp_tiernames(opt['fsys'], 'pho', 'tier', ci)

    # relevant options
    t, to, lab = copp.pp_read(annot, opt['fsys']['annot'], tn[0], stm)

    # segment input needed
    if np.ndim(t) < 2:
        return pho

    # closest/previous vowels to tc stamps
    for i in range(len(tc)):

        # default setting if no vowel found
        vlab, vdur = 'x', 1

        for j in utils.idx_a(len(t)):
            if t[j, 0] >= tc[i] or j+1 >= len(t):
                pho['lab'].append(vlab)
                pho['dur'].append(vdur)
                break
            
            if re.search(vow, lab[j]):
                vlab = lab[j]
                vdur = t[j, 1]-t[j, 0]

    return pho


def aug_do_add_pho(dom, opt):

    '''
    add normalized vowel duration feature (normalization by phoneme-related mean)
    IP: last vowel in front of boundary candidate
    ACC: nearest vowel to accent candidate
    wgt['pho'] will be removed from opt since it would require special treatment
    at several feature extraction steps. PHO features will be extracted separately
    from the others.  wgt['pho'] will be re-inserted to opt at the end of processing
    
    Args:
     dom: (str) 'glob'|'loc'
     opt: (dict) copa['config']
    
    Returns:
     boolean
     opt: (dict)
     wgt_pho: (dict)
    '''

    if (('pho' in opt['augment'][dom]['wgt']) and
        ('pho' in opt['fsys']) and ('tier' in opt['fsys']['pho']) and
            (len(opt['fsys']['pho']['tier']) > 0)):
        opt = cp.deepcopy(opt)
        wgt_pho = opt['augment'][dom]['wgt']['pho']
        del opt['augment'][dom]['wgt']['pho']
        return True, opt, wgt_pho

    # needs to be deleted in any case, otherwise more weights than features
    if 'pho' in opt['augment'][dom]['wgt']:
        del opt['augment'][dom]['wgt']['pho']

    return False, opt, 0


def aug_pho_nrm(pho):

    '''
    normalize vowel durations
    
    Args:
      pho: (dict) filled by aug_pho()
    
    Returns:
      ndur: (np.array) of normalized durations
    '''

    # collect duration values per vowel
    # myPhone -> [myDur ...]
    dur = {}

    # over vowel labels
    for i in utils.idx(pho['lab']):
        v = pho['lab'][i]
        if v not in dur:
            dur[v] = []
        dur[v].append(pho['dur'][i])
        
    # Scaler for each vowel vowel
    cs = {}
    for x in dur:
        dur[x] = np.array(dur[x])
        dur[x] = dur[x].reshape(-1, 1)
        cs[x] = sp.RobustScaler().fit(dur[x])

    # array of transformed durations
    ndur = []
    for i in range(len(pho['lab'])):
        v = pho['lab'][i]
        nd = cs[v].transform([[pho['dur'][i]]])
        ndur.append(nd[0, 0])

    return np.array(ndur)


def aug_ps_pho_mrg(ps, ndur, opt):

    '''

    adds phoneme duration to feature matrix

    Args:
      ps: (dict) pros structure
      ndur: (np.array) of normalized phoneme durations
      opt: (dict) copa['config']['augment'][dom]
    
    Returns:
      ps: with appended duration column and weight
    '''

    if len(ndur) != len(ps['batch']['fv']):
        return ps

    ps['batch']['fv'] = np.column_stack((ps['batch']['fv'], ndur))
    ps['batch']['wgt'] = np.append(ps['batch']['wgt'], opt['wgt']['pho'])
    return ps


def aug_batch_ps_upd(ps, dom, ii, i, fv, wgt, tc, is0, is1, i_nan, lng, psf, oi):

    '''
    update ps dict in batch augmentation

    Args:
      ps: (dict) as generated so far (aug_batch_ps_init() for its structure)
      dom: (str) 'glob'|'loc'
      ii: (int) file index
      i: (int) channel index
      fv: (np.array) feature matrix for [ii][i]
      wgt: (np.array) feature weights
      tc: (np.array) time stamps vector
      is0: (set) of 0 class seed indices
      is1: (set) of 1 class seed indices
      i_nan: (np.array) row idx with nans
      lng: (int) signal length
      psf: (dict) domain-related fields to be passed to ps
      oi: (int) onset index

    Returns:
      ps: (dict) updated

    '''

    # file specs
    if ii not in ps['file']:
        ps['file'][ii] = {}
    ps['file'][ii][i] = {'fv': fv, 'is0': is0, 'is1': is1,
                         'i_nan': i_nan, 'tc': tc, 'lng': lng}
    for x in psf:
        ps['file'][ii][i][x] = psf[x]

    ps['file'][ii][i]['ri'] = utils.idx_a(len(fv)) + oi

    # batch
    # add onset to candidate indices
    is0 = set(np.array(list(is0)).astype(int) + oi)
    is1 = set(np.array(list(is1)).astype(int) + oi)
    i_nan = set(np.array(list(i_nan)).astype(int) + oi)

    ps['batch']['fv'] = utils.cmat(ps['batch']['fv'], fv)
    ps['batch']['is0'] = ps['batch']['is0'].union(is0)
    ps['batch']['is1'] = ps['batch']['is1'].union(is1)
    ps['batch']['i_nan'] = ps['batch']['i_nan'].union(i_nan)
    if len(ps['batch']['wgt']) == 0:
        ps['batch']['wgt'] = wgt

    # file/channel assignment
    fi = (np.ones((len(fv), 1)) * ii).astype(int)
    ci = (np.ones((len(fv), 1)) * i).astype(int)
    ij_add = utils.cmat(fi, ci, 1)
    ps['batch']['ij'] = utils.cmat(ps['batch']['ij'], ij_add, 0)

    # time stamps (onset adding not needed, since file/channel assignment prevents
    # from comparisons across files/channels)
    ps['batch']['tc'] = np.append(ps['batch']['tc'], tc)

    return ps


def aug_ps_init():

    '''
    initialize prosodic structure dict

    Returns:
    ps: (dict)
      ps['batch']['fv'|'t'|'is0'|'is1'|'ij'|'wgt']   matrix or vectors
                               in sync: fv, t, ij [fileIdx, channelIdx]
                               is0, is1: index sets
                               wgt: 1dim featweight vector
        ['file']['fi']['ci']  fileIdx x channelIdx: rowIdx in data['fv']
                            ['is0'] set of 0-seed indices
                            ['is1'] set of 1-seed indices
                            ['i_nan'] set of nan row idx
                            ['tc']
                            ['pt']
                            ['pto']
                            ['t']
                            ['to']
                            ['lng']
                            ['ri'] row indices in featmat
    
    Feature vectors in ['clst']['fv'] will be clustered.
    Classes will be assigned to time points in ['data']['fi']['ci']['t']
    '''

    ps = {'batch': {}, 'file': {}}
    for x in ['fv', 'tc', 'i1', 'i2', 'ij', 'wgt']:
        ps['batch'][x] = np.array([])
    for x in ['is0', 'is1', 'i_nan']:
        ps['batch'][x] = set()

    return ps


def aug_annot_init(ff_annot, ff_wav, i, opt, infx=None):

    '''
    initalizes annotation dict

    Args:
      ff_annot: (list) of annotation files
      ff_wav: (list) of wav files (file name taken in case ff_annot is empty since to be generated
              from scratch)
      i: (int) current index in wav file list
      opt (dict)
      infx: (str) for secure copy of annotation file <''> (if empty string, no copy made)
    Returns:
      annot: annotation dictionary (empty if no annot file available)
      fstm: file stem for error messages
      fo: output annotation file (input file will be overwritten)
    '''

    if len(ff_annot) > i:

        pth, f = os.path.split(f"{ff_annot[i]}")
        fo = os.path.join(pth, f"{f}_{infx}")
                
        # secure copy
        if infx is not None:
            sh.copy(ff_annot[i], fo)

        # annotation file content
        annot = utils.input_wrapper(ff_annot[i], opt['fsys']['annot']['typ'])
        
        # file stem for error msg
        fstm = utils.stm(ff_annot[i])

        # output file
        fo = ff_annot[i]
    else:

        # empty annotation
        annot = {}
        fstm = ''        
        fo = os.path.join(opt['fsys']['annot']['dir'],
                          f"{utils.stm(ff_wav[i])}.{opt['fsys']['annot']['ext']}")

    return annot, fstm, fo


def aug_sub(f, f_f0, annot, fstm, opt, add_mat):

    '''
    pause/chunk/syllable/IP/AG extraction
    
    Args:
      f: (str) audioFileName
      f_f0: (str) f0 fileName
      annot: (dict) of annotations (empty if augmentation from scratch)
      fstm: (str) annotation file stem for error msg (empty if augmentation from scratch)
      fi: (int) file idx
      copa: (dict) after copa_init()
      opt: (dict) config
      add_mat: (dict) additional material dict for accent augmentation
          'copa': copa, 'fi': fileIdx, 'ff': outputDictOfPp_file_collector()
    
    Returns:
      annot: (dict) with added tiers
    '''

    aug_spec, fs, s, lng, f0_dat, ncol = aug_sig(f, f_f0, opt)
    
    # adjust opts
    opt_chunk = cp.deepcopy(opt['augment']['chunk'])
    opt_chunk['fs'] = fs
    opt_chunk['force_chunk'] = True

    # over channels
    for i in range(ncol):
        
        myLog(f"\tfile {utils.stm(f)}, channel {i+1}")

        # Signal preprocessing
        if ncol > 1:
            y = utils.sig_preproc(s[:, i])
        else:
            y = utils.sig_preproc(s)

        # F0 preprocessing
        f0, f0_ut = copp.pp_read_f0(f0_dat, opt['fsys']['f0'], i)
        f0, t, z, bv = copp.pp_f0_preproc(f0, np.round(lng, 2), opt)

        # Chunking
        if opt['navigate']['do_augment_chunk']:
            chunk = cosp.pau_detector(y, opt_chunk)
            annot = aug_annot_upd(
                'chunk', annot, {'t': chunk['tc']}, i, opt, lng)

        # Syllable nucleus + boundary detection
        if opt['navigate']['do_augment_syl']:
            d, sb = aug_syl(y, annot, opt, fs, i, fstm,
                            f, lng, aug_spec['syl'])
            annot = aug_annot_upd('syl', annot, {'t': d}, i, opt, lng)
            if 'ncl_only' not in opt['fsys']['augment']['syl']:
                annot = aug_annot_upd(
                    'syl', annot, {'t': sb}, i, opt, lng, 'bnd')

        # Global segments
        if opt['navigate']['do_augment_glob'] and opt['augment']['glob']['unit'] == 'file':
            d = aug_glob(t, z, annot, opt, i, fstm, f, lng, aug_spec['glob'])
            annot = aug_annot_upd('glob', annot, {'t': d}, i, opt, lng)

        # Accents
        if opt['navigate']['do_augment_loc'] and opt['augment']['loc']['unit'] == 'file':
            d = aug_loc(t, z, bv, annot, opt, i, fstm,
                        f, lng, add_mat, aug_spec['loc'])
            annot = aug_annot_upd('loc', annot, {'t': d}, i, opt, lng)

    return annot


def aug_sig(f, f_f0, opt):

    ''' returns signal and and f0

    Args:
    f: (str) audio file name
    f_f0: (str) f0 file name
    opt: (dict)

    Returns:
    aug_spec: (dict) augmentation specs (see aug_init())
    fs: (int) sampling rate
    s: (np.array) audio signal
    lng: (float) signal length in sec
    f0_dat: (np.array) f0
    ncol: (int) number of channels
    '''
    
    # aug constraints config
    aug_spec = aug_init(opt)

    # signal input
    fs, s_in = sio.read(f)

    # int -> float
    s = utils.wav_int2float(s_in)

    # signal length (in sec)
    lng = len(s) / fs

    # f0 input
    f0_dat = utils.input_wrapper(f_f0, opt['fsys']['f0']['typ'])

    # num of channels
    ncol = len(s.shape)

    return aug_spec, fs, s, lng, f0_dat, ncol


def aug_init(opt):

    '''
    aug spec init
    Args:
      opt: (dict) copa['config']
    Returns:
      s: (dict)
        ['chunk|syl|glob|loc']
                   ['tier_parent']
                   ['tier'|'tier_acc']  (_acc, _ag for 'loc' only)
                   ['tier_ag']
                           ['candidates']
                               - list of tier candidates
                                 (in order of been checked for existence,
                               - '' (usually in final position) indicates
                                 that file on off is used as only segment
                           ['constraints']
                                 ['ncol'] - 2 for 'segment', 1 for 'event'
                                 ['obligatory'] - True|False
    '''

    s = {}
    ua = opt['fsys']['augment']
    sd = {'candidates': [], 'constraints': {}}
    xx = ['chunk', 'syl', 'glob', 'loc']
    for x in xx:
        s[x] = {}
        s[x]['tier_parent'] = cp.deepcopy(sd)
        s[x]['tier_parent']['constraints']['ncol'] = 2
        s[x]['tier_parent']['constraints']['obligatory'] = True
        if x != 'loc':
            s[x]['tier'] = cp.deepcopy(sd)
            s[x]['tier']['constraints']['obligatory'] = True
        else:
            s[x]['tier_ag'] = cp.deepcopy(sd)
            s[x]['tier_acc'] = cp.deepcopy(sd)
            s[x]['tier_ag']['constraints']['ncol'] = 2
            s[x]['tier_ag']['constraints']['obligatory'] = False
            s[x]['tier_acc']['constraints']['ncol'] = 1
            s[x]['tier_acc']['constraints']['obligatory'] = True

    # loosen constraints
    s['syl']['tier']['constraints']['obligatory'] = False
    s['chunk']['tier']['constraints']['obligatory'] = False

    # parent tier candidates
    par = {'chunk': [], 'syl': ['chunk'], 'glob': ['chunk'],
           'loc': ['glob', 'chunk']}
    for x in xx:
        s[x]['tier_parent']['candidates'] = aug_init_parents(ua, x, par)

    # analysis tiers (empty lists if not specified here)
    # glob: default - syllable boundaries
    if utils.non_empty_val(ua['glob'], 'tier'):
        for y in utils.aslist(ua['glob']['tier']):
            s['glob']['tier']['candidates'].append(y)
    dflt = f"{ua['syl']['tier_out_stm']}_bnd"
    if dflt not in s['glob']['tier']['candidates']:
        s['glob']['tier']['candidates'].append(dflt)
        
    # loc
    if utils.non_empty_val(ua['loc'], 'tier_acc'):
        for y in utils.aslist(ua['loc']['tier_acc']):
            s['loc']['tier_acc']['candidates'].append(y)
    if ua['syl']['tier_out_stm'] not in s['loc']['tier_acc']['candidates']:
        s['loc']['tier_acc']['candidates'].append(ua['syl']['tier_out_stm'])
    if utils.non_empty_val(ua['loc'], 'tier_ag'):
        for y in utils.aslist(ua['loc']['tier_ag']):
            s['loc']['tier_ag']['candidates'].append(y)

    return s


def aug_init_parents(ua, x, par):

    '''
    returns list of parent tiers for x
    
    Args:
     ua: (dict) ['config']['augment']
     x: (str) domain 'chunk','syl'...
     par: (dict) parent dict defined in aug_init()
    
    Returns:
     z: (list) of parent tiers
    '''

    # output list, predefined order, but each element just once
    z = []
    added = {}

    # user defined parent tier
    if 'tier_parent' in ua[x]:
        y = cp.deepcopy(ua[x]['tier_parent'])
        for y in utils.aslist(ua[x]['tier_parent']):
            z.append(y)
            added[z[-1]] = True
            
    # fallback parent tiers from
    for y in par[x]:
        if ((y in ua) and ('tier_out_stm' in ua[y]) and
            (ua[y]['tier_out_stm'] not in added)):
            z.append(ua[y]['tier_out_stm'])
            added[z[-1]] = True
    z.append('FILE')
    return z


def aug_annot_upd(dom, annot, aug, i, opt, lng, infx=''):

    '''
    updates annotation dict by augmentation dict
    
    Args:
      dom: (str) domain, chunk|syl|glob|loc
      annot: (dict) annotation dict (derived from xml or TextGrid)
      aug: (dict) augmentation dict
           't':   [[on off]...], [[timeStamp]...]
           can be extended if needed
      i: (int) channel idx
      opt: (dict) copa['config']
      lng: (float) signal length (in sec, for TextGrid tier spec)
      infx: (str) infix string for syl boundary output
    
    Returns:
      annot: (dict) updated
    '''

    # do not return but output empty tier
    if len(aug['t']) == 0:
        myLog("WARNING! Augmentation: no event extracted. Empty tier output.")

    #    return annot
    # tier name
    if len(infx) == 0:
        tn = f"{opt['fsys']['augment'][dom]['tier_out_stm']}_{int(i+1)}"
        # standard label (for chunk and syl)
        lab = "x"
    else:
        tn = f"{opt['fsys']['augment'][dom]['tier_out_stm']}_{infx}_{int(i+1)}"
        lab = "x"

    # extracting time info from aug
    if opt['fsys']['annot']['typ'] == 'xml':
        return aug_annot_upd_xml(dom, annot, aug, i, opt, lng, tn, lab)
    else:
        return aug_annot_upd_tg(dom, annot, aug, i, opt, lng, tn, lab)


def aug_annot_upd_xml(dom, annot, aug, i, opt, lng, tn, lab):

    '''
    add augmentation tier to xml annotation
    
    Args:
    dom: (str) glob, loc
    annot: (dict) annotations
    aug: (dict) augmentations
    i: (int) channel index
    opt: (dict)
    lng: (float) signal length (in sec)
    tn: (str) tier name
    lab: (list) labels

    Returns:
    annot: (dict) + key "tn"

    '''

    oa = opt['fsys']['augment']

    # tier type
    if re.search(r'(chunk|glob)$', dom):
        ttyp = 'segment'
    else:
        ttyp = 'event'

    # add new tier (incl. channelIdx)
    # or overwrite existing tier with same name
    tn = f"{oa[dom]['tier_out_stm']}_{int(i+1)}"
    annot[tn] = {'type': ttyp, 'items': {}}

    for j in utils.idx_a(len(aug['t'])):
        if ttyp == 'segment':
            annot[tn]['items'][j] = {'label': lab,
                                     't_start': aug['t'][j, 0],
                                     't_end': aug['t'][j, 1]}
        else:
            annot[tn]['items'][j] = {'label': lab,
                                     't': aug['t'][j, 0]}

    return annot



def aug_annot_upd_tg(dom, annot, aug, i, opt, lng, tn, lab):

    '''
    add augmentation tier to TextGrid annotation
    
    Args:
    dom: (str) glob, loc
    annot: (dict) annotations
    aug: (dict) augmentations
    i: (int) channel index
    opt: (dict)
    lng: (float) signal length (in sec)
    tn: (str) tier name
    lab: (list) labels

    Returns:
    annot: (dict) + tier "tn"

    '''

    oa = opt['fsys']['augment']
    
    # from scratch
    if len(annot.keys()) == 0:
        annot = {'format': 'long', 'name': '', 'item': {}, 'item_name': {},
                 'head': {'xmin': 0, 'xmax': lng, 'size': 0, 'type': 'ooTextFile'}}

    specs = {'lab_pau': oa['lab_pau'], 'name': tn, 'xmax': lng}
    labs = []
    for j in utils.idx_a(len(aug['t'])):
        labs.append(lab)

    tt = utils.tg_tab2tier(aug['t'], labs, specs)
    annot = utils.tg_add(annot, tt, {'repl': True})

    return annot


def aug_syl(y, annot, opt, fs, i, fstm, f, lng, spec):

    '''
    returns syllable nuclus time stamps
    
    Args:
      y: (np.array) signal
      annot: (dict) annotation dict
      opt: (dict) copa['config']
      fs: (int) sample rate
      i: (int) channel idx
      fstm: (str) annot file stem for error msg
      f: (str) signal file name for error msg
      lng: (float) signal file length (in sec)
      spec: (dict) tier spec alternatives and constraints by aug_init()
    
    Returns:
      syl: (np.array) syllable nuclus time stamps
    '''
    
    # adjust opt
    opt_syl = cp.deepcopy(opt['augment']['syl'])
    opt_syl['fs'] = fs
    msg = aug_msg('syl', f)

    # parent tier
    tn = aug_tn('tier_parent', spec, annot, opt, i, 'syl')
    if tn == 'FILE':
        myLog(msg[1])
        pt_trunc, pt, lab_pt = t_file_parent(lng)
    elif len(tn) == 0:
        myLog(msg[6], True)
    else:
        pt_trunc, pt, lab_pt = copp.pp_read(
            annot, opt['fsys']['annot'], tn, fstm)

    # exit due to size violations
    em = {'zero': msg[2], 'ncol': msg[3]}
    err = aug_viol(pt, spec, 'tier_parent')
    if err in em:
        myLog(em[err], True)

    # time to idx
    pt = utils.sec2idx(pt, fs)

    # time stamp array for nuclei and boundaries
    syl, bnd = [], []
    
    # over parent units
    for j in utils.idx_a(len(pt)):

        # signal indices in chunk
        ii = utils.idx_seg(pt[j, 0], min(pt[j, 1], len(y)-1))

        # signal segment
        cs = y[ii]
        opt_syl['ons'] = ii[0]

        # syllable time stamps within parent unit
        s, b = cosp.syl_ncl(cs, opt_syl)

        # force at least 1 syllable to be in each parent tier unit
        if opt["augment"]["syl"]["force"] and len(s['t']) == 0:
            s['t'] = force_syl(cs, fs) + opt_syl['ons']/fs

        # add time stamps
        syl.extend(list(s['t']))
        bnd.extend(list(b['t']))
        
        # add parent-tier final time stamp
        ptf = utils.idx2sec(pt[j, 1], fs)
        if len(bnd) == 0 or bnd[-1] < ptf:
            bnd.append(ptf)

    return np.array(syl), np.array(bnd)


def force_syl(s, fs):

    '''
    called if list of found syllables is empty and
    at least one syllable needs to be detected in file
    
    Args:
      s: (np.array) signal segment (e.g. chunk)
      fs: (int) signal sample rate
    Returns:
      d: (np.array) 1 element with time point (in sec) of energy max
    '''

    y = cosp.wrapper_energy(s, {}, fs)
    ima = np.argmax(y[:, 1])
    
    return np.array([y[ima, 0]])


def aug_tn(tq, spec, annot, opt, ci, fld):

    '''
    returns first tier name in list which fullfills all following criteria:
       (1) matches opt['fsys']['channel'] (i.e. is returned by copp.pp_tiernames())
       (2) is in annotation file
       (3) fullfills segment|event constraint (if FILE, then needs to be segment)

    Args:
       tq: (str) tier type in question, i.e. 'tier_parent', 'tier', etc
       spec: (dict) fld-related subdict of aug_init()
       annot: (dict) annot file content
       opt: (dict) copa['config']
       ci: (int) channelIdx
       fld: (str) 'syl'|'glob'|'loc'
    
    Returns:
       xx (str) tier name
    '''

    atyp = opt['fsys']['annot']['typ']
    if 'ncol' in spec[tq]['constraints']:
        ncol = spec[tq]['constraints']['ncol']
    else:
        ncol = 0
        
    for x in spec[tq]['candidates']:
        xc = f"{x}_{int(ci+1)}"

        # return x if fullname, else x+channelIdx
        xx = x
        if x == 'FILE':
            # constraint (3)
            if ncol == 1:
                myLog(f"WARNING! {fld} augmentation. {tq} - {x} parent tier not usable, " \
                      "since parent needs to be event tier")
                continue
            else:
                return x

        # constraint (1)
        tn = copp.pp_tiernames(opt['fsys']['augment'], fld, tq, ci)
        if x not in tn:
            if xc not in tn:
                myLog("INFO! {fld} augmentation. {tq} candidate {x} not usable. " \
                      "Trying fallbacks")
                continue
            else:
                # stem+channelIdx fullfils constraint, continue with this string
                myLog(f"INFO! {fld} augmentation. {tq} candidate {x} found. " \
                      "Checking tier class ...")
                xx = xc
                
        # constraint (2)
        if not copp.pp_tier_in_annot_strict(xx, annot, atyp):
            if not copp.pp_tier_in_annot_strict(xc, annot, atyp):
                myLog(f"INFO! {fld} augmentation. {tq} candidate {x} not in annotation. " \
                      "Trying fallbacks")
                continue
            else:
                myLog(f"INFO! {fld} augmentation. {tq} candidate {x} found. " \
                      "Checking tier class ...")
                xx = xc
                
        # constraint (3)
        tc = copp.pp_tier_class(xx, annot, atyp)
        if (tc is not None and ((tc == 'segment' and ncol == 1) or
                                (tc == 'event' and ncol == 2))):
            myLog(f"INFO! {fld} augmentation. {tq} candidate {x} does not match required " \
                  "tier class. Trying fallbacks")
            continue
        
        myLog(f"INFO! {fld} augmentation. {tq} candidate {x}: tier class match")
        return xx

    # no matching candidate
    return ''


def aug_viol(t, spec, tq):

    '''
    returns error type if tier content T does not match requirements in specs
    
    Args:
      t: (np.array) tier content (1- or 2-dim array)
      spec: (dict) domain subdict of aug_init()
      tq: (str) type of tier in question: e.g. 'tier_parent', 'tier'...
    Returns:
      errorType: (str)
             '' no violation
             'zero': empty table
             'ncol': wrong number of columns
    '''

    if spec[tq]['constraints']['obligatory'] and len(tq) == 0:
        return 'zero'

    if ('ncol' in spec[tq]['constraints'] and
        spec[tq]['constraints']['ncol'] != utils.ncol(t)):
        return 'ncol'
    
    return ''


def aug_glob(ty, y, annot, opt, i, fstm, f, lng, spec):

    '''
    automatically extracts glob segment boundaries
    
    Args:
      ty: (np.array) time
      y: (np.array) f0
      annot: (dict) annotation
      opt: (dict) copa['config']
      i: (int) channel idx
      fstm: (str) annot file stem for error msg
      f: (str) signal file name for error msg
      lng: (float) signal file length (in sec)
      spec: (dict) tier spec alternatives and constraints by aug_init()
    
    Returns:
      glob: (np.array) global segments [[on off], ...]
    '''

    # feature matrix and weights
    fv, wgt, tc, is0, is1, i_nan, t, to, pt, pto, fto = aug_glob_fv(
        ty, y, annot, opt, i, fstm, f, lng, spec)

    if len(tc) > 0:
        # nearest centroid classification (+/- prosodic boundary)
        c, cntr, wgt = aug_cntr_wrapper('glob', fv, tc, wgt, opt, is0, is1, i_nan,
                                        opt['augment']['glob']['measure'])

    # return global segment time intervals
    return aug_glob_seg(c, tc, t, to, pt, pto, fto, opt)


def aug_glob_fv(ty, y, annot, opt, i, fstm, f, lng, spec):

    '''
    returns featvecs and feature weights for global segment bnd classification
    
    Args:
     ty: (np.array) time
     y: (np.array) f0
     annot: (dict) annotation
     opt: (dict) copa['config']
     i: (int) channelIdx
     fstm: (str) fileStem
     f: (str) fileName
     lng: (float) signal length (in smpl)
     spec: (dict) spec['glob'] alternatives and constraints by aug_init()
    
    Returns:
     fv: (np.array) m x n feat matrix
     wgt: (np.array) n x 1 feature weights
     tc: (np.array) m x 1 time stamps for feature extraction
     is0: (set) of class 0 (no boundary) seed indices
     is1: (set) of class 1 (boundary) seed indices
     i_nan: (set) of feattab row idx initially containing NaNs
     t: (np.array) [[on off]...] of inter-boundary candidate segments
     to: (np.array) t unrounded
     pt: (np.array) [[on off]...] parent tier segments
     pto: (np.array) pt unrounded
     fto: (np.array) file [on off] as segmentation fallback
    '''

    msg = aug_msg('glob', f)

    # need to force cross_chunk==True
    # (otherwise for trend features pauses at chunk boundaries
    #  are neglected)
    opt = cp.deepcopy(opt)
    opt['styl']['bnd']['cross_chunk'] = True
    gopt = opt['augment']['glob']

    # final fallback: file bnd
    ft, fto, lab_ft = t_file_parent(lng)

    # parent tier
    ptn = aug_tn('tier_parent', spec, annot, opt, i, 'glob')
    if ptn == 'FILE':
        myLog(msg[1])
        pt, pto, lab_pt = ft, fto, lab_ft
    else:
        pt, pto, lab_pt = copp.pp_read(annot, opt['fsys']['annot'], ptn, fstm)

    # analysis tier
    tn = aug_tn('tier', spec, annot, opt, i, 'glob')
    if len(tn) == 0:
        myLog(msg[5], True)
    t, to, lab = copp.pp_read(annot, opt['fsys']['glob'], tn, fstm)
    if len(t) == 0:
        myLog(msg[5], True)

    # constrain t items by pt intervals
    # t now contains segments for any input (incl. events)
    to, to_nrm, to_trend = copp.pp_t2i(to, 'bnd', opt, pto, True)
    t = np.round(to, 2)
    t_nrm = np.round(to_nrm, 2)
    t_trend = np.round(to_trend, 2)

    # r: input dict for bnd feature extraction
    #    .myIdx: over boundary candidates
    r = {}
    for j in range(len(lab)):
        
        # fallback: file nrm
        if copp.too_short('glob', to[j,], fstm):
            to2 = utils.two_dim_array(to[j, :])
            bto, bto_nrm, bto_trend = copp.pp_t2i(to2, 'bnd', opt, fto, True)
            bt = np.round(bto, 2)
            bt_nrm = np.round(bto_nrm, 2)
            bt_trend = np.round(bto_trend, 2)
            r[j] = {'t': bt[0, :], 'to': bto[0, :], 'tn': bt_nrm[0, :],
                    'tt': bt_trend[0, :], 'lab': lab[j]}
        else:
            r[j] = {'t': t[j, :], 'to': to[j, :], 'tn': t_nrm[j, :],
                    'tt': t_trend[j, :], 'lab': lab[j]}

    
    
    # time offsets (for pruning too close boundaries)
    tc = []

    # extract boundary features for each segment ...

    # weights of selected features
    wglob = gopt['wgt']

    # over cue sets: feat and wgt for each set
    fx, wx = {}, {}

    for x in wglob:

        # add phoneme features later
        if x == 'pho':
            continue
        
        fx[x] = []
        wx[x] = []
        
    # collect all indices for which pause length > 0
    #   -> certain boundaries
    is1 = set()

    # all indices in vicinity
    #   -> certainly no boundaries (min length assumption)
    is0 = set()

    # feature table row idx initially containing an NaN or Inf
    i_nan = set()

    # f0 medians [[bl ml tl]...], same length as y
    med = cost.styl_reg_med(y, opt['styl']['glob'])

    # feature matrix for seed_minmax method
    fmat = []

    # over segments in candidate tier
    # last segment is skipped for all feature subsets
    # std: fv added for j-1
    # trend: j just increased till jj[-1]
    jj = utils.sorted_keys(r)
    for j in jj:
        
        # featvec row to be added to fmat (if seed_minmax)
        fvec = []
        
        if j == jj[-1]:
            break

        # over feature sets
        for x in wglob:
            
            if x == 'std':
                tw = np.concatenate((r[j]['t'], r[j+1]['t']))
            elif x == 'trend':
                tw = r[j]['tt']
            elif x == 'win':
                tw = r[j]['tn']
            else:
                # no other featsets supported (except of pho
                # which is added below)
                continue
            
            d = cost.styl_discont_wrapper(tw, ty, y, opt, {}, med)

            # certain boundary (pause, etc.)
            if aug_certain_bnd(r, j):
                is1.add(j)
                
            # heuristics to identify cetrain non-boundaries
            if 'heuristics' in opt['augment']['glob']:
                if opt['augment']['glob']['heuristics'] == 'ORT':
                    # no boundary following articles etc.
                    if np.diff(r[j]['t']) < 0.1:
                        if j not in is1:
                            is0.add(j)

            # feature vector, user-defined weights, time stamps
            #   (time needed since not for all time stamps all
            #   features can be extracted, e.g.: syllable nucleus
            v, w = aug_glob_fv_sub(x, d, opt)

            # add to feature matrix of respective set
            fx[x].append(v)
            
            # add features to current matrix row
            fvec.extend(v)
            
            # feature weights for set x
            if len(wx[x]) == 0:
                wx[x] = w

        fmat.append(fvec)
        tc.append(r[j]['to'][1])

    fmat = np.array(fmat)
    tc = np.array(tc)
    for x in fx:
        fx[x] = np.array(fx[x])
        
    # add phoneme duration features
    add_pho, opt, wgt_pho = aug_do_add_pho('glob', opt)

    if add_pho:
        pho = {'dur': [], 'lab': []}
        pho = aug_pho('glob', pho, annot, tc, i, fstm, opt)
        ndur = aug_pho_nrm(pho)
        zz = list(fx.keys())
        if len(zz) == 0 or len(ndur) == len(fx[zz[0]]):
            fx['pho'] = utils.lol(ndur).T
            wx['pho'] = np.asarray([wgt_pho])

    # get is1 and is0 seeds from seed_minmax
    # (e.g. if no word boundaries given)
    if gopt['cntr_mtd'] == 'seed_minmax':
        is0, is1 = aug_seed_minmax(fmat, gopt)

    # get is0 seeds from min_l next to is1 boundaries
    elif len(is1) > 0 and re.search(r'^seed', gopt['cntr_mtd']):
        is0 = aug_is0(tc, is0, is1, gopt['min_l'])

    # merge featset-related featmats and weights to single matrices
    # transform to delta values depending on opt['measure']
    # weight vector is doubled in case of abs+delta
    if len(tc) > 0:
        fv, wgt, i_nan = aug_fv_mrg(fx, wx)
        # abs, delta, abs+delta
        fv, wgt = aug_fv_measure(fv, wgt, gopt['measure'])
        
    return fv, wgt, tc, is0, is1, i_nan, t, to, pt, pto, fto


def aug_glob_seg(c, tc, t, to, pt, pto, fto, opt):

    '''
    global segment intervals by centroid-based classification

    Args:
     c: (np.array) class vector 0|1
     tc: (np.array) boundary candidate time stamps
     t: (np.array) [[on off]...] of inter-boundary candidate segments
     to: (np.array) t unrounded
     pt: (np.array) [[on off] ...] parent tier segments
     pto: pt unrounded
     fto: (np.array) file on off as segment fallback
     opt: (dict) copa['config']
    Returns:
     d: (np.array) [[on off]...] on- and offsets of global segments (in sec)
    '''

    opt = cp.deepcopy(opt)
    opt['styl']['bnd']['cross_chunk'] = True
    gopt = opt['augment']['glob']

    # no global segment yet -> add file end
    #  (happens in 1-word utterances if word boundaries are IP candidates)
    if len(c) == 0:
        c = np.array([1], astype=int)
        tc = np.append(tc, fto[0, 1])

    # getting glob segments
    d = []
    ons = to[0, 0]
    for j in range(len(c)):

        # classified as boundary
        if c[j] == 1:
            d.append([ons, to[j, 1]])
            if j+1 < len(t)-1:
                ons = to[j+1, 0]
                
        # parent tier bnd (use nk2 for comparison)
        else:
            pti = np.where(pt[:, 1] == t[j, 1])[0]
            if len(pti) > 0:
                d.append([ons, to[j, 1]])
                if pti + 1 < len(pt) - 1:
                    ons = pto[pti[0] + 1, 0]
                elif j + 1 < len(t) - 1:
                    ons = to[j + 1, 0]

    # + adding pto boundaries
    if len(d) == 0:
        d.append([pto[0, 0], pto[-1, 1]])
    elif d[-1][1] < pto[-1, 1]:
        if pto[-1, 1] - d[-1][1] < gopt['min_l']:
            d[-1][1] = pto[-1, 1]
        else:
            d.append([d[-1][1], pto[-1, 1]])
            
    return np.array(d)


def rm_is1(is1, i_nan, m):

    '''
    remove is1 indices if corresp featvecs contained NaNs (meanwhile replaced by 0)
    
    Args:
      is1: (set) of is1 row indices in featmat
      i_nan: (set) of row indices in featmat that contained NaNs
      m: (str) 'abs'|'delta'|'abs+delta'; if incl. delta i_nan+1 added to i_nan
    
    Returns:
      is1 (set) after removal of i_nan-s
    '''

    isDelta = False
    if re.search('delta', m):
        isDelta = True

    for i in i_nan:
        is1.discard(i)
        if isDelta:
            is1.discard(i + 1)
    
    return is1


def aug_seed_minmax(x, o, is0=set(), is1=set()):

    '''

    calculates for each featvec it's summed abs error to 0
    The items with the highest values are put to set 1
    (since high values indicate high prominence (loc) or
    boundary strength (glob), respectively)

    Args:
      x: (np.array) feature matrix, one row per item
      o: (dict) with 'minmax_prct' key
         below 10 -> set 0, above 90 -> set 1
      is0: (set) of indices of class 0 items
      is1: (set) of indices of class 1 items

    Returns:
      is0: (set) 0 updated
      is1: (set) 1 updated
    '''

    if 'minmax_prct' not in o:
        mmp = [10, 90]
    else:
        mmp = o['minmax_prct']

    cs = sp.RobustScaler()
    x = cs.fit_transform(abs(x))
    y = np.sum(x, 1)
    p = np.percentile(y, mmp)
    is0 = set(np.where(y <= p[0])[0])
    is1 = set(np.where(y >= p[1])[0])

    return is0, is1


def aug_certain_bnd(r, j):

    '''
    returns True if pause between adjacent segments in r[j|j+1]['t']

    Args:
    r: (dict) segments time info
    j: (int) index in r

    Returns:
    boolean
    '''

    if r[j]['t'][1] < r[j+1]['t'][0]:
        return True

    return False


def aug_loc(ty, y, bv, annot, opt, i, fstm, f, lng, add_mat, spec):

    '''
    extract accented syllable time stamps
    Args:
      ty: (np.array) time for f0
      y: (np.array) f0
      bv: (float) f0 base value
      annot: (dict) annotation
      opt: (dict) copa['config']
      i: (int) channel idx
      fstm: (str) annot file stem for error msg
      f: (str) signal file name
      lng: (float) signal file length (in sec)
      add_mat: (dict) with additional material
          ['copa']: dict after copa_init()
          ['fi']: fileIdx
          ['f0']: original f0 data of file
      spec: (dict) tier spec alternatives and constraints by aug_init()
    Returns:
      loc: (np.array) with accent time stamps
    '''

    fv, wgt, tc, is0, is1, i_nan, to, ato, do_postsel = aug_loc_fv(ty, y, bv, annot, opt, i,
                                                                   fstm, f, lng, add_mat, spec)
    
    # all accents to be returned
    if len(fv) == 0:
        return to

    # centroid-based classification
    c, cntr, wgt = aug_cntr_wrapper('loc', fv, tc, wgt, opt, is0, is1, i_nan,
                                    opt['augment']['loc']['measure'])

    
    # accent time stamps
    d = aug_loc_acc(c, fv, wgt, to, ato, do_postsel, cntr)

    # force one accent
    if len(d) == 0 and opt["augment"]["loc"]["force"]:
        d = force_acc(f)
        
    return d


def force_acc(f):

    '''
    called if list of found pitch accents is empty and
    at least one accent needs to be detected in file
    
    Args:
      f: (str) signal file
    Returns:
      d: (np.array) 1 element vector with time point (in sec) of energy max
    '''

    y = cosp.wrapper_energy(f)
    ima = np.argmax(y[:, 1])
    return np.array([y[ima, 0]])


def aug_loc_acc(c, fv, wgt, to, ato, do_postsel, cntr):

    '''
    returns accent time stamps

    Args:
     c: (np.array) m x 1 binary 0|1 class vector
     fv: (np.array) m x n feature matrix
     wgt: (np.array) m x 1 feat weight vector
     to: (np.array) 1-dim time stamps
     ato: (np.array) 2-dim accent group time intervals
     do_postsel: (bool), if True, multiple accents per AG are removed
     cntr: (dict) of feature centroid vectors for class 0 and 1

    Returns:
     d: (np.array) 1-dim accent time stamps
    '''

    # accents
    d, fva = [], []
    
    for j in range(len(c)):
        if c[j] == 1:
            d.append(to[j])
            fva.append(list(fv[j, :]))

    d = np.array(d)
    fva = np.array(fva)
            
    # selecting most prominent accent in analysis tier segments
    if do_postsel:
        d, loc_i = aug_loc_postsel(d, fva, wgt, ato, cntr)

    return d


def aug_loc_fv(ty, y, bv, annot, opt, i, fstm, f, lng, add_mat, spec):

    '''
    generate accent feature matrix
    
    Args:
      ty: (np.array) time of f0
      y: (np.array) f0
      bv: (float) base value
      annot: (dict) annotation
      opt: (dict) copa['config']
      i: (int) channel index
      fstm: (str) fileStem for error messages
      f: (str) fileName
      lng: (float) signal length in sec
      add_mat: (dict), see aug_loc
      spec: (dict) 'loc' subdict by aug_init()
    
    Returns:
     fv: (np.array) feature matrix
     wgt: (np.array) feature weight vector (as specified by user,
          no data-driven estimation yet, this is done in aug_cntr_wrapper())
     tc: (np.array) accent time stamps
     is0: (set) of class 0 seed indices
     is1: (set) of class 1 seed indices
     i_nan: (set) of feature table row indices containing NaNs
     to: (np.array) 1-dim time stamp of accent candidates
     ato: (np.array) 2-dim [[on off]...] of accent group
     do_postsel: (bool) if True select most prominent syllable only from accent candidates
    '''


    global f_log
    msg = aug_msg('loc', f)

    # opt
    lopt = opt['augment']['loc']

    # file idx
    ii = add_mat['fi']

    # acc tier
    tn = aug_tn('tier_acc', spec, annot, opt, i, 'loc')

    if len(tn) == 0:
        myLog(msg[5], True)

    # time stamps and labels of accent candidates
    t, to, lab = copp.pp_read(annot, opt['fsys']['loc'], tn, fstm)

    if len(t) == 0:
        myLog(msg[5], True)

    # AG tier (if available)
    # to reduce syllable candidates
    atn = aug_tn('tier_ag', spec, annot, opt, i, 'loc')

    # time intervals and labels of AG
    if atn == 'FILE':
        myLog(msg[1])
        at, ato, lab_pt = t_file_parent(lng)
    elif len(atn) > 0:
        at, ato, alab = copp.pp_read(annot, opt['fsys']['loc'], atn, fstm)
    else:
        at, ato, alab = copp.pp_read_empty()

    # syllable pre-selection or copy from analysis point tier
    # - if analysis segment tier and criterion='left'|'right'
    # - if analysis point tier
    t, to, lab_t, do_postsel = aug_loc_preselect(
        t, to, lab, at, ato, alab, opt)

    # keep all accents if:
    # - analysis tier is available and consists of segments
    # - all AGs are to be selected
    # - do_postsel is False, i.e. accents are assigned to
    #   left/rightmost syl in analysis tier segments
    if (lopt['ag_select'] == 'all' and
            do_postsel == False and len(at) > 0 and utils.ncol(at) == 2):
        return [], [], [], [], [], [], to, [], do_postsel

    # fallback: file segments
    ft, fto, flab = t_file_parent(lng)

    # parent tier
    ptn = aug_tn('tier_parent', spec, annot, opt, i, 'loc')
    if ptn == 'FILE':
        myLog(msg[1])
        pt, pto, plab = ft, fto, flab
    else:
        pt, pto, plab = copp.pp_read(annot, opt['fsys']['annot'], ptn, fstm)

    # Prepare copa for stylizations
    # Update T and TO since some too short segments might be dropped
    #    during copa generation.
    # Drop all time stamps not within a parent tier are dropped.
    # (otherwise gestalt and gnl feature matrices would not have same number
    #  of rows, since the former is restricted to parent tier segments, the latter
    #  is not)
    add_mat['lng'] = lng
    add_mat['y'] = y
    add_mat['t'] = ty
    add_mat['bv'] = bv
    copa, t_upd, to_upd = aug_prep_copy(t, to, pto, annot, i, opt, add_mat)
    wloc = lopt['wgt']

    # parent tier declination
    copa = cost.styl_glob(copa, f_log, silent=True)

    # fallback: file-level
    if 0 not in copa['data'][ii][i]['glob']:
        copa, t_upd, to_upd = aug_prep_copy(t, to, fto, annot, i, opt, add_mat)
        copa = cost.styl_glob(copa, f_log, silent=True)
        if 0 not in copa['data'][ii][i]['glob']:
            return [], [], [], [], [], [], to, [], do_postsel

    t = t_upd
    to = to_upd

    # all feature vectors and weights
    # keys: featsets (wloc.keys())
    # e.g. fx['gnl_en'] = [[featsOfSeg1], [featsOfSeg2], ...]
    #      wx['gnl_en'] = [myFeatWeights], same len as ncol([featsOfSegX])
    fx, wx = {}, {}
    for x in wloc:

        if x == 'pho':
            # add pho duration later
            continue

        if x == 'acc':
            copa = cost.styl_loc(copa, f_log, silent=True)
        elif re.search(r'(gst|decl)', x):
            copa = cost.styl_loc_ext(copa, f_log, silent=True)
        elif x == 'gnl_f0':
            copa = cost.styl_gnl(copa, 'f0', f_log, silent=True)
        elif x == 'gnl_en':
            copa = cost.styl_gnl(copa, 'en', f_log, silent=True)
            
        v, w, tc, tco = aug_loc_feat(x, copa, ii, i)
        fx[x] = v
        wx[x] = w

    # add phoneme durations
    add_pho, opt, wgt_pho = aug_do_add_pho('loc', opt)
    if add_pho:
        pho = {'dur': [], 'lab': []}
        pho = aug_pho('loc', pho, annot, tc, i, fstm, opt)
        ndur = aug_pho_nrm(pho)
        zz = list(fx.keys())
        if len(zz) == 0 or len(ndur) == len(fx[zz[0]]):
            fx['pho'] = utils.lol(ndur).T
            wx['pho'] = np.asarray([wgt_pho])

    # merge all featset-related featmats and weights to single matrices
    fv, wgt, i_nan = aug_fv_mrg(fx, wx)
    
    if len(fv) == 0:
        return [], [], [], [], [], [], to, [], do_postsel

    # get seed candidates for stressed and unstressed words
    # if centroid method is seed_{split|kmeans} and there is an AG
    # tier below the file level which is assumed to contain word segments
    # (heuristics ORT)
    if lopt['cntr_mtd'] == 'seed_minmax':
        is0, is1 = aug_seed_minmax(fv, lopt)
    elif (re.search('seed', lopt['cntr_mtd']) and
          len(atn) > 0 and atn != 'FILE' and ('heuristics' in lopt) and
          lopt['heuristics'] == 'ORT'):
        is0, is1 = aug_loc_seeds(
            tco, ato, fv, wgt, lopt['min_l_a'], lopt['max_l_na'])
    else:
        is0, is1 = set([]), set([])

    # get is0 seeds from min_l next to is1 candidates
    if len(is1) > 0 and re.search(r'^seed', lopt['cntr_mtd']):
        is0 = aug_is0(tc, is0, is1, lopt['min_l'])

    # abs, delta, abs+delta
    # weight vector doubled in case of abs+delta
    fv, wgt = aug_fv_measure(fv, wgt, lopt['measure'])

    return fv, wgt, tc, is0, is1, i_nan, to, ato, do_postsel


def aug_loc_seeds(to, ato, fv, wgt, min_l_a, max_l_na):

    '''
    Get locseg indices of accent/no accent seed candidates based on length of AG.
    Underlying assumption: ORT heuristics, i.e. AG tier is assumed to contain word segments.
    
    Args:
      to: (np.array) [[t_sylncl] ...]
      ato: (np.array) [[t_on, t_off] ...] of word segments
      fv: (np.array) feature matrix
      wgt: (np.array) feature weight vector
      min_l_a: (float) min word length for accented seeds in sec
      max_l_na: (float) max word length for not-accented seeds in sec

    Returns:
      is0: (set) of non-accented seed indices
      is1: (set) of accented seed indices
    '''

    is0, is1 = set([]), set([])

    # fill is0 and is1
    for i in range(len(ato)):
        d = ato[i, 1] - ato[i, 0]
        if d <= max_l_na:
            v = 0
        elif d >= min_l_a:
            v = 1
        else:
            continue
        
        j = utils.find_interval(to, [ato[i, 0], ato[i, 1]])

        if len(j) == 0:
            continue
        
        if v == 0:
            for x in j:
                is0.add(x)
        elif len(j) == 1:
            is1.add(j[0])
        else:
            
            # only most prominent syllable in word as 1-candidate
            to1, fv1 = [], []
            for x in j:
                to1.append(to[x])
                fv1.append(list(fv[x, :]))

            to1 = np.array(to1)
            fv1 = np.array(fv1)
            to1, to_i = aug_loc_postsel(to1, fv1, wgt, ato)
            is1.add(j[to_i[0]])

    return is0, is1


def aug_fv_mrg(fx, wx):

    '''
    merge feature matrices and weights across feature sets
   
    Args:
      fx: (dict) feature matrix by feature set
      wx: (dict) weight vector by feature set

    Returns:
      fv: (np.array) m x n, feat matrix, col-concat over featSets;
          zero-weight features removed
      wgt: (np.array) n x 1, weight vector, col-concat
      i_nan: (set) of row indices that contain NaNs or Infs
    '''

    # m x n feat matrix, n x 1 feature weight vector
    fv, wgt = [], []

    # feature sets
    xx = sorted(fx.keys())

    if len(xx) == 0:
        return np.array(fv), np.array(wgt)

    # over segments
    for i in range(len(fx[xx[0]])):
        
        err = False

        # n x 1 feat vector for current segment
        v = []

        # over featsets
        for x in xx:

            # col concat
            if len(fx[x]) < i+1:
                err = True
                if i == 0:
                    wgt = []
                break

            # non-zero weights
            w1 = np.where(wx[x] > 0)[0]
            v.extend(fx[x][i, w1])
            
            # create wgt only once
            if i > 0 and len(wgt) > 0:
                continue
            wgt.extend(wx[x][w1])

        if err:
            continue

        fv.append(v)

        if err:
            break

    fv = np.array(fv)
    wgt = np.array(wgt)
        
    # nan rows
    i_nan = set()
    for i in utils.idx(fv):
        l_nan = np.where(np.isnan(fv[i]))[0]
        l_inf = np.where(~np.isfinite(fv[i]))[0]
        if len(l_nan) + len(l_inf) > 0:
            i_nan.add(i)
            
    # replace all NaN by 0
    # this works, since for all selected features holds:
    #   the higher (positive) its value the more likely a boundary/accent candidate
    fv = np.nan_to_num(fv)
    wgt = np.nan_to_num(wgt)

    return fv, wgt, i_nan


def t_file_parent(lng):

    '''
    returns t, to, lab for file-wide default segment

    Args:
    lng: (float) signal length in sec
    
    Returns:
    t: (np.array) [on, off] rounded
    to: (np.array) unrounded
    lab: (list) of labels
    '''

    to = np.array([0.0, lng])
    t = np.round(to, 2)
    lab = ['file']
    
    return t, to, lab


def aug_loc_postsel(loc, fv, wgt, ato, cntr={}, criterion="max"):

    '''
    selecting max prominent element in analysis tier segments,
    e.g. to allow only for a single accent within a word

    Args:
      loc: (np.array) timeStamps of accents
      fv: (np.array) corresponding featvecs
      ato: (np.array) analysis tier intervals [[on off], ...]
      cntr: (dict) dict with feat centroid vectors for classes 0 and 1
      criterion: (str)
         "max": select row with highest weighted mean of feature values
         "cntr": select row closest to 1-centroid

    Returns:
      ret: (np.array) timeStamps
      ret_i: (np.array) their indices in loc
    '''
    
    ret = []
    ret_i = []
    
    for x in ato:

        # index of segment of current accent
        i = utils.find_interval(loc, x)
        if len(i) == 0:
            continue
        
        if len(i) == 1:
            
            # only 1 accent, no postselection needed
            ret.append(loc[i[0]])
            ret_i.append(i[0])
        else:
            
            # several candidates
            if criterion == "max":

                # selected candidate max weighted mean
                s_max, j_max = -np.inf, -1
                for j in i:
                    s = utils.wgt_mean(fv[j, :], wgt)
                    if s > s_max:
                        s_max, j_max = s, j
                        
                ret.append(loc[j_max])
                ret_i.append(j_max)
                
            else:

                # select candidate closest to class 1 centroid
                s_min, j_min = np.inf, -1
                for j in i:
                    s = utils.dist_eucl(fv[j, :], cntr[1], wgt)
                    if s < s_min:
                        s_min, j_min = s, j
                        
                ret.append(loc[j_min])
                ret_i.append(j_min)

    ret = np.array(ret)
    ret_i = np.array(ret_i, dtype=int)
                
    return ret, ret_i


def aug_loc_feat(typ, copa, ii, i):

    '''
    returns feature matrix and weight vector
    for featset of type TYP (e.g. 'acc', 'gst', 'gnl_en',...)
    
    Args:
      typ: (str) feature set name
      copa: (dict)
      ii: (int) file index
      i: (int) channel index

    Returns:
      fv: (np.array) m x n feature matrix
      w: (np.array) 1 x n weight vector (with 0 for features to be dropped later)
      tc: (np.array) 1 x m time stamp vector
      tco: (np.array) tc unrounded
    '''

    fv, wgt, tc, tco = [], [], [], []

    # feature to weight mapping
    sel = copa['config']['augment']['loc']['wgt'][typ]

    # do not use absolute values of coefs (only recommended if creating
    # supervised ml input, not for clustering)
    noAbs = utils.ext_true(copa['config']['augment']['loc'], 'c_no_abs')

    if typ in copa['data'][ii][i]['loc'][0]:

        # acc, gst, decl
        c = copa['data'][ii][i]['loc']
        
        # to adress feature subset below segment idx
        fld = typ

    elif ((typ in copa['data'][ii][i]) and
          (0 in copa['data'][ii][i][typ])):

        # gnl_f0|en, only one analysis tier given, addressed by [0]
        c = copa['data'][ii][i][typ][0]

        # to adress feature subset below segment idx
        fld = 'std'

    else:
        return np.array(fv), np.array(wgt), np.array(tc), np.array(tco)

    do_wgt = True

    # over segments
    for j in utils.sorted_keys(c):
        if fld not in c[j]:
            continue
        
        # lists of segment-related row in feat matrix and its weights
        v, w = aug_loc_feat_sub([], [], c[j][fld], sel, noAbs)

        if len(v) == 0:
            continue

        fv.append(v)

        if do_wgt:
            wgt = w
            do_wgt = False

        # add syl ncl time stamp
        if len(c[j]['t']) > 2:
            tc.append(c[j]['t'][2])
            tco.append(c[j]['to'][0])
        else:
            tc.append(np.round(np.mean(c[j]['t']), 2))
            tco.append(np.mean(c[j]['to']))

    fv = np.array(fv)
    wgt = np.array(wgt)
    tc = np.array(tc)
    tco = np.array(tco)
    
    return fv, wgt, tc, tco


def aug_loc_feat_sub(fv, wgt, c, sel, noAbs=False):

    '''
    recursively adds features and weights for branch in copa dict
    if value is list/array: all elements are added with the same weight
    dicts are recursively processed
    
    Args:
      fv: (list) featvec of current segment (init: [])
      wgt: (list) weight vector (init: [])
      c: (dict) current (sub-)dict in copa['data']...[mySegmentIdx]...
           e.g. 'gst', 'ml', 'rms' (recursively stepped through until
                                    non-dict value is reached)
      sel: (dict) corresponding dict in copa['augmentation']...['wgt']...
      noAbs: (boolean) <False> do not use abs values for coefs (as recommended
                for bootstrap clustering)

    Returns:
      fv: (list) updated featvec
      wgt: (list) updated weigths

    PROCESS:
      iteration over keys (same in c and sel):
        if value is dict -> recursive processing
        if value is list -> collect values, uniform or element-wise weighting
        if value is scalar-> collect value, weighting
    '''

    # robust termination
    if type(c) is not dict:
        return fv, wgt
    
    # over feature(subsets)
    for x in sel:

        if x not in c:
            continue

        if type(sel[x]) is dict:

            # dict: recursive processing
            fv, wgt = aug_loc_feat_sub(fv, wgt, c[x], sel[x], noAbs)
            
        elif utils.of_list_type(c[x]):
            
            # array: e.g. coef vector

            for i in range(len(c[x])):

                # for polycoefs use abs values
                if x == 'c' and not noAbs:
                    fv.append(np.abs(c[x][i]))
                else:
                    fv.append(c[x][i])

                # weight each list item differently or uniformly
                if type(sel[x]) is list:
                    wgt.append(sel[x][i])
                else:
                    wgt.append(sel[x])

        else:
            # scalar: (most features)
            fv.append(c[x])
            wgt.append(sel[x])

    return fv, wgt


def aug_prep_copy(t, to, pto, annot, i, opt, add_mat):

    '''
    Copy + adjust annot, opt, copa for accent feature extraction.
    Needed for uniform call of functions in copasul_preproc and _styl
    
    Args:
      t: (np.array) rounded syllable time stamps
      to: (np.array) original syllable time stamps
      pto: (np.array) parent tier on- and offsets
      annot: (dict) tg or xml at current state of augmentation
      i: (int) channel index
      opt: (dict) copa['config']
      lng: (float) file length in sec (for tg)
      add_mat (dict)
          keys: 'fi'|'lng'|'ff'|'y'|'t'|'bv'
    Returns:
      copa: (dict) data subdict contains only one file and channel key, resp
      t: (np.array) updated
      to: (np.array) updated
    '''

    global f_log

    # adjust config
    opt = cp.deepcopy(opt)
    
    # time-sync gnl_* and loc(_ext) time stamps
    # both to be within global segment
    # (otherwise this is only checked for loc(_ext) time stamps
    #  in pp_channel())
    opt['preproc']['loc_sync'] = False
    time_key = 'loc'
    for x in ['acc', 'decl', 'gst', 'class']:
        if x in opt['augment']['loc']['wgt']:
            opt['preproc']['loc_sync'] = True
            break

    # get subdict with time info if not provided by loc
    if not opt['preproc']['loc_sync']:
        for x in ['gnl_f0', 'gnl_en', 'rhy_f0', 'rhy_en']:
            if x in opt['augment']['loc']['wgt']:
                time_key = x
                break

    tnc = copp.pp_tiernames(opt['fsys'], 'chunk', 'tier', i)
    tna = copp.pp_tiernames(opt['fsys']['augment'], 'chunk', 'tier', i)
    if (len(tnc) == 0 and len(tna) > 0):
        opt['fsys']['chunk'] = tna

    # temporary tiers for superpos analyses
    tn_loc = 'augloc_loc'
    tn_glob = 'augloc_glob'
    opt['fsys']['augment']['glob']['tier_out_stm'] = tn_glob
    opt['fsys']['augment']['loc']['tier_out_stm'] = tn_loc
    opt['fsys']['glob']['tier'] = tn_glob
    opt['fsys']['loc']['tier_ag'] = []
    opt['fsys']['loc']['tier_acc'] = tn_loc
    opt['fsys']['gnl_f0']['tier'] = [tn_loc]
    opt['fsys']['gnl_en']['tier'] = [tn_loc]
    opt['fsys']['bnd']['tier'] = []
    opt['fsys']['rhy_f0']['tier'] = []
    opt['fsys']['rhy_en']['tier'] = []
    for x in tna:
        opt['fsys']['channel'][x] = i

    opt['fsys']['channel'][f"{tn_loc}_{int(i+1)}"] = i
    opt['fsys']['channel'][f"{tn_glob}_{int(i+1)}"] = i

    # f0 is not to be preprocessed again in pp_channel()
    opt['skip_f0'] = True

    # just inform about missing tiers not needed for augmentation
    # (might be added by augmentation and will be re-read before analyses)
    opt['sloppy'] = True

    # adjust annotation
    annot = cp.deepcopy(annot)
    annot = aug_annot_upd('glob', annot, {'t': pto}, i, opt, add_mat['lng'])
    annot = aug_annot_upd('loc', annot, {'t': to}, i, opt, add_mat['lng'])

    # re-init copa (wih selected f0 and time stamps)
    copa = coin.copa_init(opt)

    # add preprocessed f0 to copa
    ii = add_mat['fi']
    copa['data'] = {ii: {i: {'f0': {}}}}
    for x in ['t', 'y', 'bv']:
        copa['data'][ii][i]['f0'][x] = add_mat[x]
    copa = copp.pp_channel(copa, opt, add_mat['fi'], i, np.array([]),
                           annot, add_mat['ff'], f_log)

    # update time stamps
    # (too short segments and segments not parented by global segment
    #  might have been dropped)
    t, to = [], []

    for ii in utils.sorted_keys(copa['data']):
        for i in utils.sorted_keys(copa['data'][ii]):
            
            if time_key == 'loc':
                # from loc sudict
                for j in utils.sorted_keys(copa['data'][ii][i]['loc']):
                    t.append(copa['data'][ii][i]['loc'][j]['t'][2])
                    to.append(copa['data'][ii][i]['loc'][j]['to'][0])
            else:
                # from gnl subdict
                c = copa['data'][ii][i][time_key][0]
                for j in utils.sorted_keys(c):
                    t = np.append(np.round(np.mean(c[j]['t']), 2))
                    to = np.append(np.mean(c[j]['to']))

    t = np.array(t)
    to = np.array(to)
                    
    return copa, t, to


def aug_loc_preselect(t, to, lab, at, ato, alab, opt):

    '''
    syllable pre-selection (before feature extraction)
    
    - if analysis tier empty
        -> return input
    - if analysis segment tier and criterion='left'|'right'
        -> return left-|rightmost syllable
    - if analysis point tier
        -> return analyis tier specs

    Args:
      t: (np.array) syl time stamps
      to: (np.array) t unrounded
      lab_t: (list) syl tier labels
      at: (np.array) AG tier time info ([[on off]...] or [[timeStamp]...]
      ato: (np.array) at anrounded
      lab_at: (list) analysis tier labels
      opt: (dict) copa['config']

    Returns:
      st: (np.array) selected syl time stamps
      sto: (np.array) st unrounded
      slab: (list) labels
      do_postsel: (boolean)
              True if segment analysis tier and most prominent syllable
              has to be chosen per segment after prom classification, i.e.
              if ['acc_select']=='max'
    '''

    # empty analysis tier
    if len(at) == 0:
        return t, to, lab, False
    
    # analysis point tier
    if utils.ncol(at) == 1:
        return at, ato, lab_a, False

    # max selection per segment only possible after classification
    if not re.search(r'(left|right)', opt['augment']['loc']['acc_select']):
        return t, to, lab, True

    # left/rightmost syllable
    sel = opt['augment']['loc']['acc_select']
    st, sto, slab = [], [], []
    for i in range(len(at)):
        j = utils.find_interval(t, at[i, :])
        if len(j) == 0:
            continue
        if sel == 'left':
            jj = j[0]
        else:
            jj = j[-1]
        st.append(t[jj])
        sto.append(to[jj])
        slab.append(lab[jj])

    st = np.array(st)
    sto = np.array(sto)
        
    return st, sto, slab, False


def aug_cntr_wrapper(typ, x, tc, wgt, opt, is0=set(), is1=set(), i_nan=set(), meas='abs',
                     ij=np.array([], dtype=int)):

    '''
    nearest centroid classification

    Args:
      typ: (str) 'glob'|'loc'
      x: (np.array) m x n feature matrix
      tc: (np.array) 1 x m corresponding time stamp values (bnd for glob, sylncl for loc)
      wgt: (np.array) 1 x n feat weight vector
      opt: (dict) copa['config']
      is0: (set) of clear class 0 row indices in x
      is1: (set) pre-defined class 1 row indices in x
      i_nan: (set) of row indices of featvecs containing nan
      meas: (str) measure abs|abs+delta|delta to see whether 1 or 2 rows in feattab are
            concerned by nans
      ij: (np.array) m x 2 [[fileIdx channelIdx], ...]

    Returns:
      c: (np.array) 1 x m binary vector containing classes 0 or 1
      cntr: (dict) with feature centroid vectors for classes 0 and 1
      wgt: (np.array) 1 x n weights (as user input or estimated from data, in case
          abs + delta values are taken, the weight vector gets double length)
    '''

    topt = opt['augment'][typ]

    # threshold to decide which scaling method to choose
    n_sparse = 10

    if len(x) == 0:
        return np.array([], dtype=int), wgt

    # robustness: convert nan to 0
    # ok, since features to be selected to be positive and positively
    # correlated to class 1 events (i.e. e.g. higher values for +accent
    # than for -accent)
    x = np.nan_to_num(x)

    # locally remove idx with nan-rows before centroid extraction
    # is1 needed outside of function to place events at corresp positions,
    # even if feature extraction failed
    is1 = cp.deepcopy(is1)
    is1 = rm_is1(is1, i_nan, meas)

    # centering + scaling of sparse data
    if len(x) <= n_sparse:
        cs = sp.MaxAbsScaler()
    else:
        cs = sp.RobustScaler()
    x = cs.fit_transform(x)

    # sets to index lists
    i0 = np.asarray(list(is0)).astype(int)
    i1 = np.asarray(list(is1)).astype(int)

    cntr, ww = aug_cntr_wrapper_sub(x, i0, i1, wgt, topt)

    # non-user defined weights
    if len(ww) > 0:
        wgt = ww

    # nearest centroid classification
    # vector of [0,1] for +/- bnd
    if topt['cntr_mtd'] == 'seed_wgt':
        attract = {0: topt['prct'], 1: 100 - topt['prct']}
        c = utils.cntr_classif(cntr, x, {'wgt': wgt, 'attract': attract})
    else:
        c = utils.cntr_classif(cntr, x, {'wgt': wgt})
    c = np.asarray(c)

    # set predefined boundaries
    if len(i1) > 0:
        c[i1] = 1

    # correct for min dist
    d = topt['min_l']
    for i in range(len(c)):
        if c[i] == 0 or i == 0:
            continue
        ii = (i in is1)
        for j in range(i - 1, -1, -1):
            if tc[i] - tc[j] >= d:
                break
            if (len(ij) > 0 and
                (ij[i, 0] != ij[j, 0] or ij[i, 1] != ij[j, 1])):
                break
            if c[j] == 1:
                jj = (j in is1)
                # both definitely boundaries
                if ii and jj:
                    continue
                elif ii:
                    c[j] = 0
                    break
                elif jj:
                    c[i] = 0
                    break
                xi = utils.wgt_mean(x[i, :], wgt)
                xj = utils.wgt_mean(x[j, :], wgt)
                if xj < xi:
                    c[j] = 0
                else:
                    c[i] = 0
                break

    # do not (!) set last item to globseg boundary
    # since it refers to last inter-segment boundary and not to file end
    # if (typ=='glob' and len(c)>0): c[-1]=1

    return c, cntr, wgt


def acw_rmIs1Err(is1, x):

    '''
    remove 0-only rows in x indexed by is1
    
    Args:
      is1: (set) of certain bnd/acc candidates
      x: (np.array) feature matrix, that might contain zero-only rows indicating
             extraction error
    Returns:
      is1: (set) with those indices removed that point to the errorneous rows in x
    '''

    y = set([])
    for i in is1:
        if max(x[i, :]) > 0:
            y.add(i)
            
    return y


def aug_fv_measure(x, wgt, mtd):

    '''
    transforms current to delta values or returns both
    depending on mtd. In case of both, weight vector is doubled
    (at this stage user-defined, not yet data-derived)
    
    Args:
      x: (np.array) feature matrix
      wgt: (np.array) weight vector
      mtd: (str) 'abs'|'delta'|'abs+delta'
    Returns:
      x: (np.array) updated
      wgt: (np.array) updated
    '''

    if mtd == 'delta':
        x = utils.calc_delta(x)
    elif mtd == 'abs+delta':
        x = np.concatenate((x, utils.calc_delta(x)), axis=1)
        wgt = np.concatenate((wgt, wgt))
        
    return x, wgt


def aug_cntr_wrapper_sub(x, i0, i1, wgt, topt):

    '''
    wrapper around clustering methods
    
    Args:
      x: (np.array) feature matrix
      i0: (np.array) index vector of 0 seed items in x
      i1: (np.array) index vector of 1 seed items in x
      wgt: (np.array) weight vector (n = ncol of x) -> zeros are kept
      topt: (dict) opt[config][augment][myDomain]
    Returns:
      cntr: (dict) with keys 0|1 and feature median vectors
      ww: (np.array) feature weight vector
    '''

    # cluster centroids: cntr_mtd=
    # 'seed_prct': set seed centroids, percentile-based fv-assignment
    # 'seed_kmeans': set/update seed centroids by kmeans clustering
    # 'seed_wgt': set seed centroids, weighted dist fv assignment
    if re.search(r'^seed', topt['cntr_mtd']) and len(i1) > 0:
        cntr, ww = aug_cntr_seed(x, {0: i0, 1: i1}, topt)
    else:
        # 'split': centroids from fv above, below defined percentile
        # also fallback if no seeds retrieved from data
        cntr, ww = aug_cntr_split(x, topt)

    return cntr, ww


def aug_is0(tc, is0, is1, d):

    '''
    Update class 0 set by adding indices that are too close to
    class 1 events

    Args:
      tc: (np.array) time stamps
      is0: (set) of class 0 seeds (indices in tc)
      is1: (set) of class 1 seeds (indices in tc)
      d: (float) min dist between time stamps

    Returns:
      is0: (set) of class 0 seeds updated
    '''

    for i in is1:
        for j in np.arange(i - 1, -1, -1):
            if tc[i] - tc[j] >= d:
                break
            
            if j in is1:
                continue
            
            if tc[i] - tc[j] < d:
                is0.add(j)
                
        for j in np.arange(i + 1, len(tc), 1):
            if tc[j] - tc[i] >= d:
                break
            
            if j in is1:
                continue
            
            if tc[j] < tc[i] < d:
                is0.add(j)
                
    return is0


def aug_cntr_seed(x, seed, opt):

    '''
    define 2 cluster centroids using seed feature vectors
      1. calc centroids of seed featvecs (for class 0 and 1)
      2. calc distance of each featvec to this centroid
      3. determine fv with distance above and below certain percentile
      4. calc centroid for each of the two groups

    Args:
      x: (np.array) feature matrix
      seed: (dict) of seed candidate index arrays for classes 0 and 1
      opt: (dict) config['augment']['glob|loc']

    Returns:
     c: (dict) to class 0 and 1 each assigned the median vector of the
         p closest featvecs to the respective seed
     w: (np.array) feature weights
    '''

    # initial centroids
    c = {0: np.array([]), 1: np.array([])}
    for j in [0, 1]:
        if len(seed[j]) > 0:
            c[j] = np.median(x[seed[j], :], axis=0)

    # check requirement >0
    req = min(len(c[0]), len(c[1])) > 0

    # feature weights (e.g. silhouette) based on init centroids
    w = aug_wgt(opt['wgt_mtd'], x, c)
    
    # keep initial centroids; nearest centroid classif by
    # weighted distance
    if req and opt['cntr_mtd'] == 'seed_wgt':
        return c, w

    # kmeans
    if req and opt['cntr_mtd'] == 'seed_kmeans':
        km = sc.KMeans(n_clusters=2, init=np.concatenate(([c[0]], [c[1]])),
                       n_init=1, random_state=opt["seed"])
        km.fit(x)
        c[0] = km.cluster_centers_[0, :]
        c[1] = km.cluster_centers_[1, :]
        return c, w
    
    # percentile-based centroid (for less balanced class expectations)
    # 'seed_prct'
    # n x 1 dist ratio vector, 1 element per featvec
    # d = [distTo0 / distTo1 ...], i.e. the higher the closer to 0
    d = []

    # over feature vectors
    for i in utils.idx_a(len(x)):
        if req:
            d0 = utils.dist_eucl(x[i, :], c[0], w)
            d1 = utils.dist_eucl(x[i, :], c[1], w)
            d.append(d0 / utils.rob0(d1))
        elif len(c[1]) > 0:
            d1 = utils.dist_eucl(x[i, :], c[1], w)
            d.append(d1)
        elif len(c[0]) > 0:
            d0 = utils.dist_eucl(x[i, :], c[0], w)
            d.append(1 / utils.rob0(d0))

    d = np.array(d)
    px = np.percentile(d, opt['prct'])
    i0, i1 = utils.robust_split(d, px)

    c[0] = np.median(x[i0, :], axis=0)
    c[1] = np.median(x[i1, :], axis=0)

    return c, w


def aug_wgt(mtd, x, c):

    '''
    calculate feature weights

    Args:
      mtd: (str) type of weighting 'user'|'corr'|'silhouette'
      x: (np.array) m x n feature matrix
      c: (dict) assigning feature vector centroids to classes 0 and 1

    Returns:
      w: (np.array) 1 x n weight vector (empty for mtd='user')
    '''

    if mtd == 'silhouette' and min(len(c[0]), len(c[1])) > 0 and len(x) > 2:
        ci = utils.cntr_classif(c, x, {})
        return cocl.featweights(x, ci)

    elif mtd == 'corr' and len(x) > 2:
        return utils.feature_weights_unsup(x, 'corr')
    
    return np.array([])


def aug_cntr_split(x, opt):

    '''
    define 2 cluster centroids as medians above, below percentile p
       (separately for each feature)
    
    Args:
     x: (np.array) feature matrix
     opt: (dict)
        'prct': percentile for median calc
        'wgt_mtd': 'silhouette', 'corr'

    Returns:
     c: (dict) with keys 0 and 1 that get assigned median vectors
        below/above percentile, respectively
     w: (np.array) feat weight vector <[]>
    '''

    px = np.percentile(x, opt['prct'], axis=0)
    c = {0: [], 1: []}

    # over columns
    for i in range(utils.ncol(x)):
        # values below/above px[i]
        i0, i1 = utils.robust_split(x[:, i], px[i])
        c[0].append(np.median(x[i0, i]))
        c[1].append(np.median(x[i1, i]))

    for i in [0, 1]:
        c[i] = np.array(c[i])

    # feature weights
    w = aug_wgt(opt['wgt_mtd'], x, c)

    return c, w


def aug_glob_fv_sub(typ, b, opt):

    '''
    single global feature vector
    
    Args:
      typ: (str) 'std'|'trend'|'win'
      b: (dict) boundary feature dict
             ['p']
             ['ml']|['rng']|...
                  ['rms']|['rms_pre']|['rms_post']
      opt: (dict) copa['config']

    Returns:
      v: (np.array) featvec
      w: (np.array) weights

    '''

    # feat selection dict
    sel = opt['augment']['glob']['wgt'][typ]
    
    # feature and weight vector
    v, w = [], []

    # pause duration
    if 'p' in sel:
        v.append(b['p'])
        w.append(sel['p'])

    # over register types (bl, ml, etc.)
    for rr in utils.lists('register'):
        
        # over features (r, rms, etc.)
        for x in b[rr]:
            if x == 'r':
                # feat "reset": take absolute value
                v.append(abs(b[rr][x]))
            else:
                v.append(b[rr][x])
            if ((rr in sel) and (x in sel[rr])):
                w.append(sel[rr][x])
            else:
                w.append(0.0)

    v = np.array(v)
    w = np.array(w)
                
    return v, w


def aug_msg(typ, f):

    '''
    standard message generator
    Args:
      typ: 'syl'|'glob'|'loc'
    Returns:
      msg dict
        numeric keys -> myStandardMessage
    '''

    if typ == 'chunk':
        prfx = 'chunking'
    elif typ == 'syl':
        prfx = 'syllable nucleus extraction'
    elif typ == 'glob':
        prfx = 'IP extraction'
    elif typ == 'loc':
        prfx = 'accent extraction'

    fstm = utils.stm(f)

    return {1: f"WARNING! {fstm}: {prfx} carried out without preceding chunking",
            2: f"ERROR! {fstm}: {prfx} requires chunk tier or preceding chunking!",
            3: f"ERROR! {fstm}: {prfx} requires parent tier with segments, not events!",
            4: f"WARNING! {fstm}: no analysis tier for {prfx} specified or provided. Fallback: syllable boundaries.",
            5: f"ERROR! {fstm}: neither analysis tier nor syllable nucleus/boundary fallback available for {prfx}. " \
            "(Re-)Activate syllable nucleus augmentation or define analysis tier.",
            6: f"WARNING! {fstm}: {prfx}, no parent tier specified or found. Trying glob augment tier.",
            7: f"ERROR! {fstm}: {prfx}, no parent tier found."}


def myLog(msg, e=False):

    '''
    log file output (if filehandle), else terminal output

    Args:
      msg: (str) message string
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
