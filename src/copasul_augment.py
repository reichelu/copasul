#!/usr/bin/env python3

# author: Uwe Reichel, Budapest, 2016

import scipy.io.wavfile as sio
import scipy.signal as sis
import numpy as np
import math
import matplotlib.pyplot as plt
import mylib as myl
import sys
import copy as cp
import scipy.fftpack as sf
import sklearn.cluster as sc
import sklearn.preprocessing as sp
import sigFunc as sif
import os.path as op
import shutil as sh
import copasul_init as coin
import copasul_preproc as copp
import copasul_styl as cost
import copasul_clst as cocl
import re

### common structure for filewise and batch processing for IP and accent extraction 
#aug_glob/aug_batch('glob',...):
# 1. feature matrix, weights, seed vectors, time info
#   fv, wgt, tc, is0, is1, t, to, pt, pto, fto = aug_glob_fv(ty,y,annot,opt,i,fstm,f,lng,spec)
# 2. clustering
#   c, cntr, wgt = aug_cntr_wrapper('glob',fv,tc,wgt,opt,is0,is1)
# 3. segmentation based on class vector c
#   ip = aug_glob_seg(c,tc,t,to,pt,pto,fto,opt)
#aug_loc/aug_batch('loc',...):
# 1. feature matrix, weights, seed vectors, time info
#   fv, wgt, tc, is0, is1,to, ato, do_postsel = aug_loc_fv(ty,y,bv,annot,opt,i,fstm,f,lng,add_mat,spec)
# 2. clustering
#   c, cntr, wgt = aug_cntr_wrapper('loc',fv,tc,wgt,opt,is0,is1)
# 3. accentuation based on class vector c, and further selection within accent groups/words
#   acc = aug_loc_acc(c,fv,wgt,to,ato,do_postsel,cntr)
# augments TextGrids by tiers: myChunkTierName_myChannelIdx and
#  mySylTierName_myChannelIdx
# not stored in copa dict, since relation to f0 file index which is
# the key in subdict copa['data'] is unknown
def aug_main(copa,f_log_in):
    global f_log
    f_log = f_log_in
    opt = copa['config']

    # wav files
    ff_wav = myl.file_collector(opt['fsys']['aud']['dir'],
                                opt['fsys']['aud']['ext'])
    # annot files
    ff_annot = myl.file_collector(opt['fsys']['annot']['dir'],
                                opt['fsys']['annot']['ext'])
    # f0 files
    ff_f0 = myl.file_collector(opt['fsys']['f0']['dir'],
                                opt['fsys']['f0']['ext'])

    # additional material for acc extraction
    add_mat={'ff':copp.pp_file_collector(opt)}

    # over wav files
    timeStamp = myl.isotime()
    for i in myl.idx_a(len(ff_wav)):

        # add to existing file or generation from scratch
        annot, fstm, fo = aug_annot_init(ff_annot,ff_wav,i,opt,timeStamp)
        add_mat['fi']=i

        # extending annotation by chunk, syl tiers f.a. channel
        # if unit=='file' glob and loc tiers are also added
        annot = aug_sub(ff_wav[i],ff_f0[i],annot,fstm,opt,add_mat)

        myl.output_wrapper(annot,fo,opt['fsys']['annot']['typ'])

    # boundaries batch clustering
    if opt['navigate']['do_augment_glob'] and opt['augment']['glob']['unit']=='batch':
        aug_batch('glob',ff_wav,ff_annot,ff_f0,opt,add_mat)

    # accents batch clustering
    if opt['navigate']['do_augment_loc'] and opt['augment']['loc']['unit']=='batch':
        aug_batch('loc',ff_wav,ff_annot,ff_f0,opt,add_mat)
        
    return


# batch clustering
# IN:
#   dom: domain 'glob'|'loc'
#   ff_wav: wav file list
#   ff_annot: annotation file list
#   ff_f0: f0 file list
#   opt: copa['config']

# OUT:
#   (adding tiers to annotation files)
def aug_batch(dom,ff_wav,ff_annot,ff_f0,opt,add_mat):
    print('Batch clustering {}'.format(dom))
    # feature vectors for entire corpus: ['batch'] subdict
    #    and for each file: ['file'] subdict
    ps = aug_batch_ps(dom,ff_wav,ff_annot,ff_f0,opt,add_mat)
    # clustering over all data
    d = ps['batch']
    #print(d['fv'], d['wgt'], ps['file'][0][0]['fv'])
    c, cntr, wgt = aug_cntr_wrapper(dom,d['fv'],d['tc'],d['wgt'],opt,d['is0'],d['is1'])
    #print("wgt {}: ".format(dom), wgt) #!wgt
    #myl.stopgo() #!wgt
    # distributing c to files
    ps = aug_batch_distrib(ps,c,wgt)
    for ii in myl.numkeys(ps['file']):
        annot, fstm, fo = aug_annot_init(ff_annot,ff_wav,ii,opt)
        for i in myl.numkeys(ps['file'][ii]):
            psf = ps['file'][ii][i]
            if dom=='glob':
                d = aug_glob_seg(psf['c'],psf['tc'],psf['t'],psf['to'],
                                 psf['pt'],psf['pto'],psf['fto'],opt)
            else:
                d = aug_loc_acc(psf['c'],psf['fv'],psf['wgt'],psf['to'],psf['ato'],
                                psf['do_postsel'],cntr)
            annot = aug_annot_upd(dom,annot,{'t':d},i,opt,psf['lng'])
        myl.output_wrapper(annot,fo,opt['fsys']['annot']['typ'])
    return

# distributes batch class vector over files/channels
# IN:
#   ps: prosStruct dict
#   c: class vector
#   wgt: feature weights
# OUT:
#   ps['file'] + ['c']: class-subvector for file/channel
#              + ['wgt']: feat weight vector
#              + ['fv']-replacement (since pho-features might have been added)
#                  filewise fv are needed for accent post-selection
def aug_batch_distrib(ps,c,wgt):
    ps['batch']['wgt'] = wgt
    fv = ps['batch']['fv']
    for ii in myl.numkeys(ps['file']):
        for i in myl.numkeys(ps['file'][ii]):
            ps['file'][ii][i]['c'] = c[ps['file'][ii][i]['ri']]
            ps['file'][ii][i]['fv'] = fv[ps['file'][ii][i]['ri']]
            ps['file'][ii][i]['wgt'] = wgt
    return ps

# prosodic struct dict containing batch and file-related info
# IN:
#   dom: domain 'glob'|'loc'
#   ff_wav: wav file list
#   ff_annot: annotation file list
#   ff_f0: f0 file list
#   opt: copa['config']
#   add_mat: misc additional material, customized online
# OUT:
#   ps['batch']['fv'|'tc'|'is0'|'is1'|'ij']   matrix or vectors
#                            in sync: fv, tc, ij [fileIdx, channelIdx]
#     ['file']['fi']['ci']  fileIdx x channelIdx: rowIdx in data['fv']
#                         ['is0'] set of 0-seed indices
#                         ['is1'] set of 1-seed indices
#                         ['tc'] time stamps of boundaries or accent candidates
#                         ['pt'] parent segments [[on off]...]
#                         ['pto'] pt unrounded
#                         ['t'] for acc only
#                         ['to']  t unrounded
#                         ['lng'] signal length
#                         ['ri'] row indices in featmat

def aug_batch_ps(dom,ff_wav,ff_annot,ff_f0,opt,add_mat):
    # batch prosstruct container
    ps = aug_ps_init()
    # onset idx (featmatRow-file/channel assignment)
    oi = 0

    add_pho, opt, wgt_pho = aug_do_add_pho(dom,opt)
    pho = {'dur':myl.ea(),'lab':[]}

    # over f0 files
    for ii in myl.idx_a(len(ff_f0)):
        f = ff_wav[ii]
        aug_spec, fs, s, lng, f0_dat, ncol = aug_sig(f,ff_f0[ii],opt)
        # existing file or generation from scratch
        annot, fstm, fo = aug_annot_init(ff_annot,ff_wav,ii,opt)
        add_mat['fi']=ii
        # over channels
        for i in range(ncol):
            ## Signal preprocessing ######################
            #if dom=='loc':
            #    if ncol>1:
            #        y = myl.sig_preproc(s[:,i])
            #    else:
            #        y = myl.sig_preproc(s)
            ## F0 preprocessing ##########################
            f0, f0_ut = copp.pp_read_f0(f0_dat,opt['fsys']['f0'],i)
            f0, t, z, bv = copp.pp_f0_preproc(f0,myl.trunc2(lng),opt)
            ## phrase/accent extraction ################## 
            if dom == 'glob':
                fv,wgt,tc,is0,is1,t,to,pt,pto,fto = aug_glob_fv(t,z,annot,opt,i,fstm,
                                                                f,lng,aug_spec['glob'])
                # domain dependent input
                psf = {'t':t, 'pt':pt,'pto':pto,'to':to,'fto':fto}
            else:
                fv,wgt,tc,is0,is1,to,ato,do_postsel = aug_loc_fv(t,z,bv,annot,opt,i,
                                                                 fstm,f,lng,add_mat,
                                                                 aug_spec['loc'])
                psf = {'do_postsel':do_postsel,'ato':ato,'to':to}
            if add_pho:
                pho = aug_pho(dom,pho,annot,tc,i,fstm,opt)
                #print(pho) ##!!
                #myl.stopgo() ##!!

            ps = aug_batch_ps_upd(ps,dom,ii,i,fv,wgt,tc,is0,is1,lng,psf,oi)
            
            oi = len(ps['batch']['fv'])

    ## normalize vowel durations
    if add_pho:
        ndur = aug_pho_nrm(pho)
        ## merge pho and fv, add weight
        # reinsert weight
        opt['augment'][dom]['wgt']['pho']=wgt_pho
        ps = aug_ps_pho_mrg(ps,ndur,opt['augment'][dom])

    return ps

# IN:
#   ps: prosStruct dict
#   ndur: vector of normalized durations
#   opt: copa['config']['augment'][dom]
# OUT:
#   ps with merged ndur
def aug_ps_pho_mrg(ps,ndur,opt):
    if len(ndur) != len(ps['batch']['fv']):
        #print(ps) ##!!
        #print('diff length!',len(ndur),len(ps['batch']['fv'])) ##!!
        return ps
    #print('same length!') ##!!
    ps['batch']['fv'] = np.concatenate((ps['batch']['fv'],myl.lol(ndur).T),1)
    ps['batch']['wgt'] = np.append(ps['batch']['wgt'],opt['wgt']['pho'])
    return ps

# normalize vowel durations
# IN:
#   pho: dict filled by aug_pho()
# OUT:
#   ndur: vector of normalized durations
def aug_pho_nrm(pho):

    ## collect duration values per vowel
    # myPhone -> [myDur ...]
    dur = {}
    # over vowel labels
    for i in range(len(pho['lab'])):
        v = pho['lab'][i]
        if v not in dur:
            dur[v] = myl.ea()
        dur[v] = np.append(dur[v], pho['dur'][i])

    ## center scale object per vowel
    # myPhone -> myObj
    cs = {}
    for x in dur:
        dur[x] = dur[x].reshape(-1, 1)
        cs[x] = sp.RobustScaler().fit(dur[x])

    ## vector of transformed durations
    ndur = myl.ea()
    for i in range(len(pho['lab'])):
        v = pho['lab'][i]
        nd = cs[v].transform([[pho['dur'][i]]])
        ndur = np.append(ndur,nd[0,0])

    return ndur

# extract vowel length
# IN:
#  dom: 'glob'|'loc'
#  pho: phonem dict to be updated, initialized in aug_batch()
#  annot_ annotation dict
#  tc: potential prosodic event file stamps where to look for vowel length
#  ci: channelIdx
#  stm: annotation file stem
#  opt: copa['config']
# OUT:
#  pho: updated by duration, label at tc stamps
def aug_pho(dom,pho,annot,tc,ci,stm,opt):
    # vowel pattern
    vow = opt['fsys']['pho']['vow']
    # pho-tier(s) in list
    tn = copp.pp_tiernames(opt['fsys'],'pho','tier',ci)
    # relevant options
    t, to, lab = copp.pp_read(annot,opt['fsys']['annot'],tn[0],stm)
    # segment input needed
    if np.ndim(t)<2:
        return pho
    # closest/previous vowels to tc stamps
    for i in range(len(tc)):
        # default setting if no vowel found
        vlab, vdur = 'x', 1
        for j in myl.idx_a(len(t)):
            if t[j,0] >= tc[i] or j+1>=len(t):
                pho['lab'].append(vlab)
                pho['dur'] = np.append(pho['dur'],vdur)
                break
            if re.search(vow,lab[j]):
                vlab = lab[j]
                vdur = t[j,1]-t[j,0]
    return pho

# add normalized vowel duration feature (normalization by phoneme-related mean)
# IP: last vowel in front of boundary candidate
# ACC: nearest vowel to accent candidate
# wgt['pho'] will be removed from opt since it would require special treatment
# at several feature extraction steps. PHO features will be extracted separately
# from the others.  wgt['pho'] will be re-inserted to opt at the end of processing
# IN:
#  dom: 'glob'|'loc'
#  opt: copa['config']
# OUT:
#  boolean
#  opt
#  wgt_pho
def aug_do_add_pho(dom,opt):
    if (('pho' in opt['augment'][dom]['wgt']) and
        ('pho' in opt['fsys']) and ('tier' in opt['fsys']['pho']) and
        (len(opt['fsys']['pho']['tier'])>0)):
        opt = cp.deepcopy(opt)
        wgt_pho = opt['augment'][dom]['wgt']['pho']
        del opt['augment'][dom]['wgt']['pho']
        return True, opt, wgt_pho
    # needs to be deleted in any case, otherwise more weights than features
    if 'pho' in opt['augment'][dom]['wgt']:
        del opt['augment'][dom]['wgt']['pho']
    return False, opt, 0

# update ps dict in batch augmentation
# IN:
#   ps: dict as generated so far (aug_batch_ps_init() for its structure)
#   dom: 'glob'|'loc'
#   ii: file index
#   i: channel index
#   fv: feature matrix for [ii][i]
#   wgt: feature weights
#   tc: time stamps vector
#   is0: set of 0 class seed indices
#   is1: set of 1 class seed indices
#   lng: signal length
#   psf: domain-related fields to be passed to ps
#   oi: onset index
def aug_batch_ps_upd(ps,dom,ii,i,fv,wgt,tc,is0,is1,lng,psf,oi):
    # file specs
    if ii not in ps['file']:
        ps['file'][ii]={}
    ps['file'][ii][i]={'fv':fv,'is0':is0,'is1':is1,
                       'tc':tc,'lng':lng}
    for x in psf:
        ps['file'][ii][i][x]=psf[x]

    ps['file'][ii][i]['ri'] = myl.idx_a(len(fv))+oi

    # batch
    # add onset to candidate indices
    is0 = set(np.asarray(list(is0)).astype(int) + oi)
    is1 = set(np.asarray(list(is1)).astype(int) + oi)

    ps['batch']['fv'] = myl.cmat(ps['batch']['fv'],fv)
    ps['batch']['is0'] = ps['batch']['is0'].union(is0)
    ps['batch']['is1'] = ps['batch']['is1'].union(is1)
    if len(ps['batch']['wgt'])==0:
        ps['batch']['wgt']=wgt
    # file/channel assignment
    fi = (np.ones((len(fv),1))*ii).astype(int)
    ci = (np.ones((len(fv),1))*i).astype(int)
    ps['batch']['ij'] = myl.cmat(fi,ci,1)
    # time stamps (onset adding not needed, since file/channel assignment prevents
    # from comparisons across files/channels)
    ps['batch']['tc'] = np.append(ps['batch']['tc'],tc)

    return ps



# prosodic structure dict
# OUT:
#   ps['batch']['fv'|'t'|'is0'|'is1'|'ij'|'wgt']   matrix or vectors
#                            in sync: fv, t, ij [fileIdx, channelIdx]
#                            is0, is1: index sets
#                            wgt: 1dim featweight vector 
#     ['file']['fi']['ci']  fileIdx x channelIdx: rowIdx in data['fv']
#                         ['is0'] set of 0-seed indices
#                         ['is1'] set of 1-seed indices
#                         ['tc']
#                         ['pt']
#                         ['pto']
#                         ['t']
#                         ['to']
#                         ['lng']
#                         ['ri'] row indices in featmat
# feature vectors in ['clst']['fv'] will be clustered.
# Classes will be assigned to time points in ['data']['fi']['ci']['t']
def aug_ps_init():
    ps = {'batch':{}, 'file':{}}
    for x in ['fv', 'tc', 'i1', 'i2', 'ij', 'wgt']:
        ps['batch'][x]=myl.ea()
    for x in ['is0', 'is1']:
        ps['batch'][x]=set()
    return ps

# initalizes annotation dict
# IN:
#   ff_annot: list of annotation files
#   ff_wav: list of wav files (file name taken in case ff_annot is empty since to be generated
#           from scratch)
#   i: current index in wav file list
#   opt
#   timeStamp: for secure copy <''> (if empty string, no copy made)
# OUT:
#   annot: annotation dictionary (empty if no annot file available)
#   fstm: file stem for error messages
#   fo: output annotation file (input file will be overwritten)
def aug_annot_init(ff_annot,ff_wav,i,opt,timeStamp=''):
    if len(ff_annot) > i:
        # secure copy
        if len(timeStamp)>0:
            sh.copy(ff_annot[i],"{}.copy_{}".format(ff_annot[i],timeStamp))
        # annotation file content
        annot = myl.input_wrapper(ff_annot[i],opt['fsys']['annot']['typ'])
        # file stem for error msg
        fstm = myl.stm(ff_annot[i])
        # output file
        fo = ff_annot[i]
    else:
        # empty annotation
        annot = {}
        fstm = ''
        fo = "{}.{}".format(op.join(opt['fsys']['annot']['dir'],myl.stm(ff_wav[i])),
                            opt['fsys']['annot']['ext'])
    return annot, fstm, fo

# pause/chunk/syllable/IP/AG extraction
# IN:
#   f - audioFileName
#   f_f0 - f0 fileName
#   annot - dict of annotations (empty if augmentation from scratch)
#   fstm - annotation file stem for error msg (empty if augmentation from scratch)
#   fi - file idx
#   copa - dict after copa_init()
#   opt - dict config
#   add_mat - additional material dict for accent augmentation
#       'copa':copa, 'fi':fileIdx, 'ff':outputDictOfPp_file_collector()
# OUT:
#   annot with added tiers
def aug_sub(f,f_f0,annot,fstm,opt,add_mat):

    print(fstm)
    #myLog(fstm) ##!!t

    aug_spec, fs, s, lng, f0_dat, ncol = aug_sig(f,f_f0,opt)

    ## adjust opts
    opt_chunk = cp.deepcopy(opt['augment']['chunk'])
    opt_chunk['fs']=fs
    opt_chunk['force_chunk']=True
    
    # over channels
    for i in range(ncol):
        ## Signal preprocessing ######################
        if ncol>1: y = myl.sig_preproc(s[:,i])
        else: y = myl.sig_preproc(s)
        ## F0 preprocessing ##########################
        f0, f0_ut = copp.pp_read_f0(f0_dat,opt['fsys']['f0'],i)
        f0, t, z, bv = copp.pp_f0_preproc(f0,myl.trunc2(lng),opt)        
        ## Chunking ##################################
        if opt['navigate']['do_augment_chunk']:
            print("\tchunking ...")
            chunk = sif.pau_detector(y,opt_chunk)
            annot = aug_annot_upd('chunk',annot,{'t':chunk['tc']},i,opt,lng)
        ## Syllable nucleus + boundary detection #####
        if opt['navigate']['do_augment_syl']:
            print("\tsyllable nucleus extraction ...")
            d,sb = aug_syl(y,annot,opt,fs,i,fstm,f,lng,aug_spec['syl'])
            annot = aug_annot_upd('syl',annot,{'t':d},i,opt,lng)
            annot = aug_annot_upd('syl',annot,{'t':sb},i,opt,lng,'bnd')
        ## Global segments ###########################
        if opt['navigate']['do_augment_glob'] and opt['augment']['glob']['unit']=='file':
            print("\tphrase segmentation ...")
            d = aug_glob(t,z,annot,opt,i,fstm,f,lng,aug_spec['glob'])
            annot = aug_annot_upd('glob',annot,{'t':d},i,opt,lng)

        ## Accents ####################################
        if opt['navigate']['do_augment_loc'] and opt['augment']['loc']['unit']=='file':
            print("\taccent location ...")
            d = aug_loc(t,z,bv,annot,opt,i,fstm,f,lng,add_mat,aug_spec['loc'])
            annot = aug_annot_upd('loc',annot,{'t':d},i,opt,lng)

    return annot


def aug_sig(f,f_f0,opt):
    ## aug constraints config
    aug_spec = aug_init(opt)
    ## signal input
    fs, s_in = sio.read(f)
    # int -> float
    s = myl.wav_int2float(s_in)
    # signal length (in sec)
    lng = len(s)/fs
    ## f0 input
    f0_dat = myl.input_wrapper(f_f0,opt['fsys']['f0']['typ'])
    # num of channels
    ncol = len(s.shape)
    return aug_spec, fs, s, lng, f0_dat, ncol


# aug spec init
# IN:
#   opt: copa['config']
# OUT:
#   s dict
#     ['chunk|syl|glob|loc']
#                ['tier_parent']
#                ['tier'|'tier_acc']  (_acc, _ag for 'loc' only)
#                ['tier_ag']
#                        ['candidates']
#                            - list of tier candidates
#                              (in order of been checked for existence,
#                            - '' (usually in final position) indicates
#                              that file on off is used as only segment
#                        ['constraints']
#                              ['ncol'] - 2 for 'segment', 1 for 'event'
#                              ['obligatory'] - True|False
def aug_init(opt):
    s = {}
    ua = opt['fsys']['augment']
    sd = {'candidates':[], 'constraints':{}}
    xx = ['chunk','syl','glob','loc']
    for x in xx:
        s[x] = {}
        s[x]['tier_parent'] = cp.deepcopy(sd)
        s[x]['tier_parent']['constraints']['ncol']=2
        s[x]['tier_parent']['constraints']['obligatory']=True
        if x != 'loc':
            s[x]['tier'] = cp.deepcopy(sd)
            s[x]['tier']['constraints']['obligatory']=True
        else:
            s[x]['tier_ag'] = cp.deepcopy(sd)
            s[x]['tier_acc'] = cp.deepcopy(sd)
            s[x]['tier_ag']['constraints']['ncol']=2
            s[x]['tier_ag']['constraints']['obligatory']=False
            s[x]['tier_acc']['constraints']['ncol']=1
            s[x]['tier_acc']['constraints']['obligatory']=True
    ## loosen constraints
    s['syl']['tier']['constraints']['obligatory']=False
    s['chunk']['tier']['constraints']['obligatory']=False
    ## parent tier candidates
    par={'chunk':[],'syl':['chunk'],'glob':['chunk'],'loc':['glob','chunk']}
    for x in xx:
        s[x]['tier_parent']['candidates'] = aug_init_parents(ua,x,par)
    ## analysis tiers (empty lists if not specified here)
    # glob: default - syllable boundaries
    if myl.non_empty_val(ua['glob'],'tier'):
        for y in myl.aslist(ua['glob']['tier']):
            s['glob']['tier']['candidates'].append(y)
    dflt = "{}_bnd".format(ua['syl']['tier_out_stm'])
    if dflt not in s['glob']['tier']['candidates']:
        s['glob']['tier']['candidates'].append(dflt)
    # loc
    if myl.non_empty_val(ua['loc'],'tier_acc'):
        for y in myl.aslist(ua['loc']['tier_acc']):
            s['loc']['tier_acc']['candidates'].append(y)
    if ua['syl']['tier_out_stm'] not in s['loc']['tier_acc']['candidates']:
        s['loc']['tier_acc']['candidates'].append(ua['syl']['tier_out_stm'])
    if myl.non_empty_val(ua['loc'],'tier_ag'):
        for y in myl.aslist(ua['loc']['tier_ag']):
            s['loc']['tier_ag']['candidates'].append(y)

    return s


# returns parent domains for x in ua as list
# IN:
#  ua: ['config']['augment']
#  x: 'chunk','syl'...
#  par: parent dict defined in aug_init()
# OUT:
#  z: list of parent tiers
def aug_init_parents(ua,x,par):
    # output list, predefined order, but each element just once
    z=[]
    added={}
    # user defined parent tier
    if 'tier_parent' in ua[x]:
        y = cp.deepcopy(ua[x]['tier_parent'])
        for y in myl.aslist(ua[x]['tier_parent']):
            z.append(y)
            added[z[-1]]=True
    # fallback parent tiers from 
    for y in par[x]:
        if ((y in ua) and ('tier_out_stm' in ua[y]) and
            (ua[y]['tier_out_stm'] not in added)):
            z.append(ua[y]['tier_out_stm'])
            added[z[-1]]=True
    z.append('FILE')
    return z


# updates annotation dict by augmentation dict
# IN:
#   dom: domain, chunk|syl|glob|loc
#   annot: annotation dict (derived from xml or TextGrid)
#   aug: augmentation dict
#        't':   [[on off]...], [[timeStamp]...]
#        can be extended if needed
#   i: channel idx
#   opt: copa['config']
#   lng: signal length (in sec, for TextGrid tier spec)
#   infx: infix string for syl boundary output
# OUT:
#   annot updated
def aug_annot_upd(dom,annot,aug,i,opt,lng,infx=''):

    # do not return but output empty tier
    if len(aug['t'])==0:
        myLog("{} augmentation: no event extracted. Empty tier output.")
    #    return annot
    
    # tier name
    if len(infx)==0:
        tn = "{}_{}".format(opt['fsys']['augment'][dom]['tier_out_stm'],int(i+1))
        # standard label (for chunk and syl)
        lab = "x"
    else:
        tn = "{}_{}_{}".format(opt['fsys']['augment'][dom]['tier_out_stm'],infx,int(i+1))
        lab = "x"

    # extracting time info from aug
    if opt['fsys']['annot']['typ'] == 'xml':
        return aug_annot_upd_xml(dom,annot,aug,i,opt,lng,tn,lab)
    else:
        return aug_annot_upd_tg(dom,annot,aug,i,opt,lng,tn,lab)

# called by aug_annot_upd, same I/O
#   + tn: tier name
#     lb: standard label
def aug_annot_upd_xml(dom,annot,aug,i,opt,lng,tn,lab):
    oa = opt['fsys']['augment']
    
    # tier type
    if re.search('(chunk|glob)$',dom):
        ttyp = 'segment'
    else:
        ttyp = 'event'
    
    # add new tier (incl. channelIdx)
    # or overwrite existing tier with same name
    tn = "{}_{}".format(oa[dom]['tier_out_stm'],int(i+1))
    annot[tn]={'type':ttyp,'items':{}}

    for j in myl.idx_a(len(aug['t'])):
        if ttyp == 'segment':
            annot[tn]['items'][j]={'label':lab,
                                   't_start':aug['t'][j,0],
                                   't_end':aug['t'][j,1]}
        else:
            annot[tn]['items'][j]={'label':lab,
                                   't':aug['t'][j,0]}

    return annot

# called by aug_annot_upd, same I/O
def aug_annot_upd_tg(dom,annot,aug,i,opt,lng,tn,lab):
    oa = opt['fsys']['augment']
    # from scratch
    if len(annot.keys())==0:
        annot = {'format':'long','name':'','item':{},'item_name':{},
                 'head':{'xmin':0, 'xmax':lng,'size':0,'type':'ooTextFile'}}

    specs = {'lab_pau':oa['lab_pau'],'name':tn,'xmax':lng}
    labs = []
    for j in myl.idx_a(len(aug['t'])):
        labs.append(lab)
    tt = myl.tg_tab2tier(aug['t'],labs,specs)
    annot = myl.tg_add(annot,tt,{'repl':True})

    return annot
    


# syl ncl time stamps
# IN:
#   y: signal
#   annot: annotation dict
#   opt: copa['config']
#   fs: sample rate
#   i: channel idx
#   fstm: annot file stem for error msg
#   f: signal file name for error msg
#   lng: signal file length (in sec)
#   spec: tier spec alternatives and constraints by aug_init()
# OUT:
#   syl: arrayOfTimeStamps 
def aug_syl(y,annot,opt,fs,i,fstm,f,lng,spec):
    ## adjust opt
    opt_syl = cp.deepcopy(opt['augment']['syl'])
    opt_syl['fs']=fs
    msg = aug_msg('syl',f)
    ## parent tier
    tn = aug_tn('tier_parent',spec,annot,opt,i,'syl')
    if tn=='FILE':
        myLog(msg[1])
        pt_trunc, pt, lab_pt = t_file_parent(lng)
    elif len(tn)==0:
        myLog(msg[6],True)
    else:
        pt_trunc, pt, lab_pt = copp.pp_read(annot,opt['fsys']['annot'],tn,fstm)

    # exit due to size violations
    em = {'zero':msg[2],'ncol':msg[3]}
    err = aug_viol(pt,spec,'tier_parent')
    if err in em: myLog(em[err],True)

    # time->idx
    pt = myl.sec2idx(pt,fs)
    ## time stamp array for nuclei and boundaries
    syl, bnd = myl.ea(2)
    # over parent units
    for j in myl.idx_a(len(pt)):
        # signal indices in chunk
        ii = myl.idx_seg(pt[j,0],min(pt[j,1],len(y)-1))
        # signal segment
        cs = y[ii]
        opt_syl['ons'] = ii[0]
        # add syllable time stamps
        s,b = sif.syl_ncl(cs,opt_syl)
        syl = np.append(syl,s['t'])
        bnd = np.append(bnd,b['t'])
        # add parent-tier final time stamp
        ptf = myl.idx2sec(pt[j,1],fs)
        if len(bnd)==0 or bnd[-1] < ptf:
            bnd = np.append(bnd,ptf)


    return syl, bnd

# returns first tier name in list which is available in annot file and fullfills all criteria
# criteria: 
#    (1) matches opt['fsys']['channel'] (i.e. is returned by copp.pp_tiernames())
#    (2) is in annotation file
#    (3) fullfills segment|event constraint (if FILE, then needs to be segment)
# IN:
#    tq: tier type in question, i.e. 'tier_parent', 'tier', etc
#    spec: fld-related subdict of aug_init()
#    annot: annot file content
#    opt: copa['config']
#    ci: channelIdx
#    fld: 'syl'|'glob'|'loc'
# OUT:
#    tierName string
def aug_tn(tq,spec,annot,opt,ci,fld):
    atyp = opt['fsys']['annot']['typ']
    if 'ncol' in spec[tq]['constraints']:
        ncol = spec[tq]['constraints']['ncol']
    else:
        ncol = 0
    for x in spec[tq]['candidates']:
        xc = "{}_{}".format(x,int(ci+1))
        # return x if fullname, else x+channelIdx
        xx = x
        if x == 'FILE':
            # constraint (3)
            if ncol==1:
                myLog("Warning! {} augmentation. {} - {} parent tier not usable, since parent needs to be event tier".format(fld,tq,x))
                continue
            else:
                return x
        # constraint (1)
        tn = copp.pp_tiernames(opt['fsys']['augment'],fld,tq,ci)
        if x not in tn:
            if xc not in tn:
                myLog("Info! {} augmentation. {} candidate {} not usable. Trying fallbacks".format(fld,tq,x))
                continue
            else:
                # stem+channelIdx fullfils constraint, continue with this string
                xx = xc
        # constraint (2)
        if not copp.pp_tier_in_annot_strict(xx,annot,atyp):
            if not copp.pp_tier_in_annot_strict(xc,annot,atyp):
                myLog("Info! {} augmentation. {} candidate {} not in annotation. Trying fallbacks".format(fld,tq,x))
                continue
            else:
                xx = xc
        # constraint (3)
        tc = copp.pp_tier_class(xx,annot,atyp)
        if ((tc=='segment' and ncol==1) or (tc=='event' and ncol==2)):
            myLog("Info! {} augmentation. {} candidate {} does not match required tier class. Trying fallbacks".format(fld,tq,x))
            continue
        return xx
    # no matching candidate
    return ''


# returns error type if tier content T does not match requirements in specs
# IN:
#   t: tier content (1- or 2-dim array)
#   spec: domain subdict of aug_init() dict
#   tq: tierInQuestion: e.g. 'tier_parent', 'tier'...
# OUT:
#   errorType: '' no violation
#              'zero': empty table
#              'ncol': wrong number of columns
def aug_viol(t,spec,tq):
    if spec[tq]['constraints']['obligatory'] and len(tq)==0:
        return 'zero'
    if 'ncol' in spec[tq]['constraints'] and spec[tq]['constraints']['ncol'] != myl.ncol(t):
        return 'ncol'
    return ''


# glob segment [[on off]...]
# IN:
#   ty: time
#   y: f0
#   annot: annotation dict
#   opt: copa['config']
#   i: channel idx
#   fstm: annot file stem for error msg
#   f: signal file name for error msg
#   lng: signal file length (in sec)
#   spec: tier spec alternatives and constraints by aug_init()
# OUT:
#   glob: [[on off]...]
def aug_glob(ty,y,annot,opt,i,fstm,f,lng,spec):
    fv, wgt, tc, is0, is1, t, to, pt, pto, fto = aug_glob_fv(ty,y,annot,opt,i,fstm,f,lng,spec)
    # c[j]==1: bnd following after segment j
    c = myl.eai()
    if len(tc)>0:
        #print(len(tc),len(fv),wgt,len(is0),len(is1)) #!
        c, cntr, wgt = aug_cntr_wrapper('glob',fv,tc,wgt,opt,is0,is1)
    return aug_glob_seg(c,tc,t,to,pt,pto,fto,opt)



# global segments by centroid-based classification
# IN:
#  c: class vector 0|1
#  tc: boundary candidate time stamps
#  t: [[on off]...] of inter-boundary candidate segments
#  to: t unrounded
#  pt: [[on off] ...] parent tier segments
#  pto: [[]...] unrounded
#  fto: [] file on off as segment fallback
#  opt: copa['config']
# OUT:
#  d: [[on off]...] IPs
def aug_glob_seg(c,tc,t,to,pt,pto,fto,opt):

    opt = cp.deepcopy(opt)
    opt['styl']['bnd']['cross_chunk']=1
    gopt = opt['augment']['glob']

    # no IP -> add file end
    #  (happens in 1-word utterances if word boundaries are IP candidates) 
    if len(c)==0:
        c = myl.push(c,1)
        tc = myl.push(tc,fto[0,1])

    # getting glob segments
    d = myl.ea()
    ons = to[0,0]
    for j in range(len(c)):
        # classified as boundary 
        if c[j]==1:
            d = myl.push(d,[ons, to[j,1]])
            if j+1 < len(t)-1:
                ons = to[j+1,0]
        # parent tier bnd (use nk2 for comparison)
        else:
            pti = myl.find(pt[:,1],'==',t[j,1])
            if len(pti)>0:
                d = myl.push(d,[ons, to[j,1]])
                if pti+1 < len(pt)-1:
                    ons = pto[pti[0]+1,0]
                elif j+1 < len(t)-1:
                    ons = to[j+1,0]


        #if (c[j]==1 or len(myl.find(pt[:,1],'==',t[j,1]))>0):
        #d = myl.push(d,[ons, myl.trunc2(to[j,1])])
        #    d = myl.push(d,[ons, to[j,1]])
        #    print('-> YES',d)
        #    if j+1 < len(t)-1:
        #        ons = to[j+1,0]
                

    # + adding pto boundaries
    if len(d)==0:
        d = myl.push(d,[pto[0,0],pto[-1,1]])
    elif d[-1,1] < pto[-1,1]:
        if pto[-1,1]-d[-1,1] < gopt['min_l']:
            d[-1,1] = pto[-1,1]
        else:
            d = myl.push(d,[d[-1,1], pto[-1,1]])

    return d

# featvecs and feature weights for bnd classification
# IN:
#  ty: [...] time
#  y:  [...] f0
#  annot: annotation dict
#  opt: copa['config']
#  i: channelIdx
#  fstm: fileStem
#  f: fileName
#  lng: signal length (in smpl)
#  spec: spec['glob'] alternatives and constraints by aug_init()
# OUT:
#  fv: [[...]...] feat matrix
#  wgt: [...] feature weights
#  tc: [...] time stamps for feature extraction
#  is0: set of class 0 seed indices
#  is1: set of class 1 seed indices
#  t: [[on off]...] of inter-boundary candidate segments
#  to: t unrounded
#  pt: [[on off]...] parent tier segments
#  pto: unsrounded
#  fto: file [on off] as segmentation fallback
def aug_glob_fv(ty,y,annot,opt,i,fstm,f,lng,spec):
    msg = aug_msg('glob',f)

    # need to force cross_chunk==1
    # (otherwise for trend features pauses at chunk boundaries
    #  are neglected)
    opt = cp.deepcopy(opt)
    opt['styl']['bnd']['cross_chunk']=1
    gopt = opt['augment']['glob']

    ## final fallback: file bnd
    ft, fto, lab_ft = t_file_parent(lng)

    ## parent tier
    ptn =  aug_tn('tier_parent',spec,annot,opt,i,'glob')
    if ptn=='FILE':
        myLog(msg[1])
        pt, pto, lab_pt = ft, fto, lab_ft
    else:
        pt, pto, lab_pt = copp.pp_read(annot,opt['fsys']['annot'],ptn,fstm)

    ## analysis tier
    tn = aug_tn('tier',spec,annot,opt,i,'glob')
    if len(tn)==0: myLog(msg[5],True)
    t, to, lab = copp.pp_read(annot,opt['fsys']['glob'],tn,fstm)
    if len(t)==0: myLog(msg[5],True)

    ## constrain t items by pt intervals
    # t now contains segments for any input (incl. events)
    to, to_nrm, to_trend = copp.pp_t2i(to,'bnd',opt,pto,True)
    t = myl.cellwise(myl.trunc2,to)
    t_nrm = myl.cellwise(myl.trunc2,to_nrm)
    t_trend = myl.cellwise(myl.trunc2,to_trend)

    ## r: input dict for bnd feature extraction
    r = {}
    for j in range(len(lab)):
        # fallback: file nrm
        if copp.too_short('glob',to[j,],fstm):
            to2 = myl.two_dim_array(to[j,:])
            bto, bto_nrm, bto_trend = copp.pp_t2i(to2,'bnd',opt,fto,True)
            bt = myl.cellwise(myl.trunc2,bto)
            bt_nrm = myl.cellwise(myl.trunc2,bto_nrm)
            bt_trend = myl.cellwise(myl.trunc2,bto_trend)
            r[j] = {'t':bt[0,:],'to':bto[0,:],'tn':bt_nrm[0,:],
                    'tt':bt_trend[0,:],'lab':lab[j]}
        else:
            r[j] = {'t':t[j,:],'to':to[j,:],'tn':t_nrm[j,:],
                    'tt':t_trend[j,:],'lab':lab[j]}

    ## init
    # fv: all feature vectors
    # tc: time offsets (for pruning too close boundaries)
    # wgt: feature weights for centroid classification
    #      (length is ncol(fv))
    # cluster idx vector
    fv, tc, wgt, c = myl.ea(4)
    # bnd features for each segment
    # (code taken over from cost.styl_bnd(): for k in nk...)

    # opt
    wglob = gopt['wgt']

    # over cue sets: feat and wgt for each set
    fx, wx = {}, {}
    for x in wglob:
        #!pho add pho later
        if x=='pho':
            continue
        fx[x] = np.asarray([])
        wx[x] = np.asarray([])
    
    # collect all indices for which pause length > 0
    #   -> certain boundaries
    is1 = set()
    # all indices in vicinity
    #   -> certainly no boundaries (min length assumption)
    is0 = set()

    # f0 medians [[bl ml tl]...], same length as y
    med = cost.styl_reg_med(y,opt['styl']['glob'])

    # feature matrix for seed_minmax method
    fmat = myl.ea()

    # over segments in candidate tier
    # last segment is skipped for all feature subsets
    # std: fv added for j-1
    # trend: j just increased till jj[-1]
    jj = myl.numkeys(r)
    for j in jj:
        # featvec to be added to fmat (if seed_minmax)
        fvec=myl.ea()
        if j == jj[-1]:
            break
        for x in wglob:
            if x=='std':
                tw = np.concatenate((r[j]['t'],r[j+1]['t']))
            elif x=='trend':
                tw = r[j]['tt']
            elif x=='win':
                tw = r[j]['tn']
            else:
                # no other featsets supported (pho see below)
                continue
            d = cost.styl_discont_wrapper(tw,ty,y,opt,{},med)
            if aug_certain_bnd(r,j):
                is1.add(j)
            ## use heuristics
            if 'heuristics' in opt['augment']['glob']:
                if opt['augment']['glob']['heuristics']=='ORT':
                    # no boundary following articles etc.
                    if np.diff(r[j]['t'])<0.1:
                        if j not in is1:
                            is0.add(j)
            # feature vector, user-defined weights, time stamps
            #   (time needed since not for all time stamps all
            #   features can be extracted, e.g.: syllable nucleus
            v,w = aug_glob_fv_sub(x,d,opt)
            fx[x] = myl.push(fx[x],v)
            fvec = np.append(fvec,v)
            if len(wx[x])==0:
                wx[x] = w
        fmat = myl.push(fmat,fvec)
        tc = myl.push(tc, r[j]['to'][1])

    #!pho
    add_pho, opt, wgt_pho = aug_do_add_pho('glob',opt)
    if add_pho:
        pho = {'dur':[], 'lab':[]}
        pho = aug_pho('glob',pho,annot,tc,i,fstm,opt)
        ndur = aug_pho_nrm(pho)
        zz = list(fx.keys())
        if len(zz)==0 or len(ndur)==len(fx[zz[0]]):
            fx['pho']=myl.lol(ndur).T
            wx['pho']=[wgt_pho]
    #!pho
    
    # get is1 and is0 seeds from seed_minmax
    if gopt['cntr_mtd'] == 'seed_minmax':
        is0, is1 = aug_seed_minmax(fmat,gopt)
    # get is0 seeds from min_l next to is1 boundaries
    elif len(is1)>0 and re.search('^seed',gopt['cntr_mtd']):
        is0 = aug_is0(tc,is0,is1,gopt['min_l'])



    # merge featset-related featmats and weights to single matrices
    # transform to delta values depending on opt['measure']
    # weight vector is doubled in case of abs+delta
    if len(tc)>0:
        fv, wgt = aug_fv_mrg(fx,wx)
        # abs, delta, abs+delta
        fv, wgt = aug_fv_measure(fv,wgt,gopt['measure'])

    return fv, wgt, tc, is0, is1, t, to, pt, pto, fto

# calculates for each featvec it's MAE to 0
# The items with the highest values are put to set 1 
# IN:
#   x: feature matrix, one row per item
#   o: minmax_prct <[5, 95]>
#   is0: <{}>
#   is1: <{}>
# OUT:
#   is0: set of indices of class 0 items 
#   is1: set of indices of class 1 items

def aug_seed_minmax(x,o,is0=set(),is1=set()):
    if 'minmax_prct' not in o:
        mmp=[10,90]
    else:
        mmp=o['minmax_prct']
    
    cs = sp.RobustScaler()
    x = cs.fit_transform(abs(x))
    # row-wise mean
    #y = np.mean(x,1)
    y = np.sum(x,1)
    p = np.percentile(y,mmp)
    is0 = set(np.where(y<=p[0])[0])
    is1 = set(np.where(y>=p[1])[0])
    #print(y,p,is0,is1)
    #myl.stopgo()
    return is0, is1



# returns True if feat dict pause between adjacent segments in r[j|j+1]['t']
# not possible for chunk final 
def aug_certain_bnd(r,j):
    if r[j]['t'][1] < r[j+1]['t'][0]:
        return True
    #if (('p' in d) and (d['p']>0)):
    #    return True
    return False

# accents [[timeStamp]...]
# IN:
#   ty: time for f0
#   y: f0
#   bv: f0 base value
#   annot: annotation dict
#   opt: copa['config']
#   i: channel idx
#   fstm: annot file stem for error msg
#   f: signal file name for error msg
#   lng: signal file length (in sec)
#   add_mat: dict with additional material
#       ['copa']: dict after copa_init()
#       ['fi']: fileIdx
#       ['f0']: original f0 data of file
#   spec: tier spec alternatives and constraints by aug_init()
# OUT:
#   loc: [timeStamp ...]
def aug_loc(ty,y,bv,annot,opt,i,fstm,f,lng,add_mat,spec):
    fv, wgt, tc, is0, is1, to, ato, do_postsel = aug_loc_fv(ty,y,bv,annot,opt,i,fstm,f,lng,add_mat,spec)

    # all accents to be returned
    if len(fv)==0:
        return to

    # centroid-based classification
    # c[j]==1: bnd following after segment j 
    c, cntr, wgt = aug_cntr_wrapper('loc',fv,tc,wgt,opt,is0,is1)

    return aug_loc_acc(c,fv,wgt,to,ato,do_postsel,cntr)

# returns accent time stamps
# IN:
#  c: 0|1 class vector
#  fv: feature matrix
#  wgt: feat weight vector
#  to: [...] time stamps
#  ato: [[...]...] ag time intervals
#  do_postsel: boolean, if True, multiple accents per AG are removed
#  cntr: centroid dict
#    [0|1]->centroid vector
# OUT:
#  d: [[timeStamp]] accent time stamps
def aug_loc_acc(c,fv,wgt,to,ato,do_postsel,cntr):
    # getting all accents
    d, fva = myl.ea(2)
    for j in range(len(c)):
        if c[j]==1:
            d = myl.push(d,to[j])
            fva = myl.push(fva,fv[j,:])

    # selecting max in analysis tier segments
    if do_postsel:
        d, loc_i = aug_loc_postsel(d,fva,wgt,ato,cntr)
        
    return d

# acc feature vectors
# IN:
#   ty: time
#   y: f0
#   bv: base value
#   annot: annotation dict
#   opt: copa['config']
#   i: channelIndex
#   fstm: fileStem for error messages
#   f: fileName
#   lng: signal length
#   add_mat: dict, see aug_loc
#   spec: 'loc' subdict by aug_init()
# OUT:
#  fv: feature matrix
#  wgt: feature weight vector (as specified by user,
#       no data-driven estimation yet, this is done in aug_cntr_wrapper())
#  tc: time stamp
#  is0: class 0 seed index set
#  is1: class 1 seed index set
#  to: time stamp of accent candidates
#  ato: [[on off]...] of accent group
#  do_postsel: boolean
#           True if segment analysis tier and most prominent syllable
#           has to be chosen per segment after prom classification, i.e.
#           if ['acc_select']=='max'
def aug_loc_fv(ty,y,bv,annot,opt,i,fstm,f,lng,add_mat,spec):

    global f_log
    msg = aug_msg('loc',f)

    # opt
    lopt = opt['augment']['loc']
    
    # file idx
    ii = add_mat['fi']

    ## acc tier
    tn = aug_tn('tier_acc',spec,annot,opt,i,'loc')
    
    if len(tn)==0: myLog(msg[5],True)

    # time stamps and labels of accent candidates
    t, to, lab = copp.pp_read(annot,opt['fsys']['loc'],tn,fstm)

    if len(t)==0: myLog(msg[5],True)
    
    ## AG tier (not oblig)
    # used to reduce syllable candidates
    atn = aug_tn('tier_ag',spec,annot,opt,i,'loc')

    # time intervals and labels of AG
    if atn=='FILE':
        myLog(msg[1])
        at, ato, lab_pt = t_file_parent(lng)
    elif len(atn)>0:
        at, ato, alab = copp.pp_read(annot,opt['fsys']['loc'],atn,fstm)
    else:
        at, ato, alab = copp.pp_read_empty()

    ## syllable pre-selection or copy from analysis point tier
    # - if analysis segment tier and criterion='left'|'right'
    # - if analysis point tier
    t,to,lab_t,do_postsel = aug_loc_preselect(t,to,lab,at,ato,alab,opt)

    # keep all accents if:
    # - analysis tier is available and consists of segments
    # - all AGs are to be selected
    # - do_postsel is False, i.e. accents are assigned to 
    #   left/rightmost syl in analysis tier segments
    if (lopt['ag_select']=='all' and
        do_postsel==False and len(at)>0 and myl.ncol(at)==2):
        return [], [], [], [], [], to, [], do_postsel

    ## fallback: file segments
    ft, fto, flab = t_file_parent(lng)
    ## parent tier
    ptn =  aug_tn('tier_parent',spec,annot,opt,i,'loc')
    if ptn=='FILE':
        myLog(msg[1])
        pt, pto, plab = ft, fto, flab
    else:
        pt, pto, plab = copp.pp_read(annot,opt['fsys']['annot'],ptn,fstm)

    # reduce accent candidates to the ones within parent segments

    ## no needed, will be done in pp_channel()
    #t, to, lab_t = aug_loc_parented(t,to,lab_t,pt)

    add_mat['lng']=lng
    add_mat['y']=y
    add_mat['t']=ty
    add_mat['bv']=bv
    # prepare copa for stylizations
    # update, t, to since some too short segments might be dropped
    # during copa generation.
    # All time stamps not within a parent tier are dropped, too
    # (otherwise gestalt and gnl feature matrices would not have same number
    # of rows; the former is restricted to parent tier segments, the latter
    # is not)
    copa, t_upd, to_upd = aug_prep_copy(t,to,pto,annot,i,opt,add_mat)
    #sys.exit()
    wloc = lopt['wgt']

    ## parent tier declination
    copa = cost.styl_glob(copa)
    ## fallback: file-level
    if 0 not in copa['data'][ii][i]['glob']:
        copa, t_upd, to_upd = aug_prep_copy(t,to,fto,annot,i,opt,add_mat)
        copa = cost.styl_glob(copa)
        if 0 not in copa['data'][ii][i]['glob']:
            return myl.ea()

    t = t_upd
    to = to_upd

    # all feature vectors and weights
    # keys: featsets (wloc.keys())
    # e.g. fx['gnl_en'] = [[featsOfSeg1], [featsOfSeg2], ...]
    #      wx['gnl_en'] = [myFeatWeights], same len as ncol([featsOfSegX])
    fx, wx = {}, {}
    for x in wloc:

        #!pho add pho later
        if x=='pho':
            continue
        if x == 'acc':
            copa = cost.styl_loc(copa)
        elif re.search('gst|decl',x):
            copa = cost.styl_loc_ext(copa,f_log)
        elif x == 'gnl_f0':
            copa = cost.styl_gnl(copa,'f0')
        elif x == 'gnl_en':
            copa = cost.styl_gnl(copa,'en')
        v,w,tc,tco = aug_loc_feat(x,copa,ii,i)
        fx[x] = v
        wx[x] = w

    #!pho
    add_pho, opt, wgt_pho = aug_do_add_pho('loc',opt)
    if add_pho:
        pho = {'dur':[], 'lab':[]}
        pho = aug_pho('loc',pho,annot,tc,i,fstm,opt)
        ndur = aug_pho_nrm(pho)
        zz = list(fx.keys())
        if len(zz)==0 or len(ndur)==len(fx[zz[0]]):
            fx['pho']=myl.lol(ndur).T
            wx['pho']=[wgt_pho]
    #!pho

    # merge featset-related featmats and weights to single matrices
    fv, wgt = aug_fv_mrg(fx,wx)
    if len(fv)==0: return myl.ea()

    # get seed candidates for stressed and unstressed words
    # if entroid method is seed_{split|kmeans} and there is an AG
    # tier below the file level which is assumed to contain word segments
    # (heuristics ORT)
    if lopt['cntr_mtd'] == 'seed_minmax':
        is0, is1 = aug_seed_minmax(fv,lopt)
    elif (re.search('seed',lopt['cntr_mtd']) and
        len(atn)>0 and atn != 'FILE' and ('heuristics' in lopt) and
        lopt['heuristics']=='ORT'):
        is0, is1 = aug_loc_seeds(tco,ato,fv,wgt,lopt['min_l_a'],lopt['max_l_na'])
    else:
        is0, is1 = set(), set()

    # get is0 seeds from min_l next to is1 candidates
    if len(is1)>0 and re.search('^seed',lopt['cntr_mtd']):
        is0 = aug_is0(tc,is0,is1,lopt['min_l'])

    # abs, delta, abs+delta
    # if specified: delta values, weight vector doubling in case of abs+delta
    fv, wgt = aug_fv_measure(fv,wgt,lopt['measure'])
    
    return fv, wgt, tc, is0, is1, to, ato, do_postsel


# removes those time stamps that are not within a global segment
# IN:
#   t  nx1 timeStamps
#   to nx1 timeStamps unrounded
#   lab corresponding label list
#   pt [[on off]...] of parent segments
# OUT:
#   t  mx1 timeStamps
#   to mx1 timeStamps unrounded
#   lab corresponding label list
#      within pt segments
def aug_loc_parented(t,to,lab,pt):
    tr,tor=myl.ea(2)
    labr=[]
    for i in range(len(t)):
        j = myl.first_interval(t,pt)
        if j>-1:
            tr = myl.push(tr,t[i])
            tor = myl.push(tor,to[i])
            labr.append(lab[i])
    
    return tr, tor, labr

# get locseg indices of certain accent/no accent candidates based on length of AG
# (underlying this call: ORT heuristics, i.e. AG tier is assumed to contain word segments)
# IN:
#   to:  [[t_sylncl] ...]
#   ato: [[t_on, t_off] ...] of word segments
#   fv:  feature matrix
#   wgt: feature weight vector
#   min_l_a: min word length for accented seeds
#   max_l_na: max word length for not-accented seeds
# OUT:
#   is0: index set of non-accented seeds
#   is1: index set of accented seeds
def aug_loc_seeds(to,ato,fv,wgt,min_l_a,max_l_na):
    is0, is1 = set(), set()
    
    # is0, is1
    # is1 needs to be corrected to carry only 1 element per word
    for i in range(len(ato)):
        #print(ato[i,])
        d = ato[i,1]-ato[i,0]
        if d <= max_l_na:
            v = 0
        elif d >= min_l_a:
            v = 1
        else:
            continue
        j = myl.find_interval(to,[ato[i,0],ato[i,1]])
        
        if len(j)==0: continue
        #print('j:',j)
        #print('v:',v)
        if v==0:
            for x in j:
                is0.add(x)
                #print('push0',x)
        elif len(j)==1:
            is1.add(j[0])
            #print('push1',j[0])
        else:
            # only most prominent syllable in word as 1-candidate
            to1, fv1 = myl.ea(2)
            for x in j:
                to1 = myl.push(to1,to[x])
                fv1 = myl.push(fv1,fv[x,:])
            to1, to_i = aug_loc_postsel(to1,fv1,wgt,ato)
            is1.add(j[to_i[0]])
            #print('to_i',to_i)
            #print('pushcorr1', j[to_i])

    #print('0:',is0)
    #print('1:',is1)
    #myl.stopgo()

    return is0, is1


# merge feat dicts to feature and weight tables
# IN:
#   fx: dict myFeatSet -> myFeatTab
#   wx: dict myFeatSet -> myWeightVector
# OUT:
#   fv: feat matrix, col-concat over featSets
#   wgt: weight vector, col-concat
def aug_fv_mrg(fx,wx):
    fv, wgt = myl.ea(2)
    # featsets
    xx = sorted(fx.keys())
    if len(xx)==0:
        return fv, wgt

    # over segments
    for i in range(len(fx[xx[0]])):
        err=False
        v = myl.ea()
        # over featsets
        for x in xx:
            # col concat
            if len(fx[x])<i+1:
                err=True
                if i==0: wgt = myl.ea()
                break
            ##!!np: later numpy versions do not support
            ## column concat with empty arrays
            ##!!np v = np.append(v,fx[x][i,:], 1)
            v = np.append(v,fx[x][i,:])
            if i>0 and len(wgt)>0: continue
            ##!!np wgt = np.append(wgt, wx[x],1)
            wgt = np.append(wgt, wx[x])
        if err: continue
        fv = myl.push(fv,v)
        if err: break

    fv = np.nan_to_num(fv)
    wgt = np.nan_to_num(wgt)

    return fv, wgt

# returns t, to, lab for file-wide default segment
def t_file_parent(lng):
    return np.asarray([[myl.trunc2(0),myl.trunc2(lng)]]), np.asarray([[0,lng]]), ['file']

# selecting max prominent element in analysis tier segments
# currently: highest weighted mean of feature values
# comment 1==1 if 'most prominent' should be defined as
#   'closest to the centroid'
# IN:
#   loc: timeStamps of accents
#   fv: corresponding featvecs
#   ato: analysis tier [on off]
#   cntr: <{}> centroid dict
#     [0|1]: centroid vector
# OUT:
#   ret: timeStamps
#   ret_i: their indices in loc
def aug_loc_postsel(loc,fv,wgt,ato,cntr={}):
    ret = myl.ea()
    ret_i = myl.eai()
    for x in ato:
        i = myl.find_interval(loc,x)
        if len(i)==0: continue
        if len(i)==1:
            ret = myl.push(ret,loc[i[0]])
            ret_i = np.append(ret_i,i[0])
        else:
            # if no centroids provided (call from initial acc selection)
            # -> find max weighted mean
            #if len(cntr.keys())==0:
            if 1==1:
                s_max, j_max = -1,-1
                for j in i:
                    s = myl.wgt_mean(fv[j,:],wgt)
                    if s>s_max:
                        s_max=s
                        j_max=j
                ret = myl.push(ret,loc[j_max])
                ret_i = np.append(ret_i, j_max)
            # -> find featvec closest to class 1 centroid
            else:
                s_min, j_min = -1,-1
                for j in i:
                    s = myl.dist_eucl(fv[j,:],cntr[1],wgt)
                    if s_min < 0 or s < s_min:
                        s_min=s
                        j_min=j
                ret = myl.push(ret,loc[j_min])
                ret_i = np.append(ret_i, j_min)

    return ret, ret_i
            
# returns nxm feature matrix and 1xm weight vector
# for featset of type TYP (e.g. 'acc', 'gst', 'gnl_en',...)
# IN:
#   typ: featSet string
#   copa
#   ii: fileIdx
#   i: channelIdx
# OUT:
#   fv: n x m feature matrix
#   w:  1 x m weight vector
#   tc:  time stamp vector
#   tco: unrounded
def aug_loc_feat(typ,copa,ii,i):
    fv, wgt, tc, tco = myl.ea(4)
    # feat -> userWeight
    sel = copa['config']['augment']['loc']['wgt'][typ]
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
        return fv, wgt, tc, tco

    do_wgt = True
    #print('\n\nNEW: typ and fld:', typ, fld)
    # over segments
    for j in myl.numkeys(c):
        #print('segment', j)
        if fld not in c[j]:
            continue
        # segment-related row in feat matrix
        v,w = aug_loc_feat_sub(myl.ea(),myl.ea(),c[j][fld],sel)
        if len(v)==0: continue
        fv = myl.push(fv,v)
        if do_wgt:
            wgt = w
            do_wgt = False
        # add syl ncl time stamp
        # within loc (len>2) or gnl|... subdict
        if len(c[j]['t'])>2:
            tc = myl.push(tc,c[j]['t'][2])
            tco = myl.push(tco,c[j]['to'][0])
        else:
            tc = myl.push(tc, myl.trunc2(np.mean(c[j]['t'])))
            tco = myl.push(tco, np.mean(c[j]['to']))
    #print(fv)
    #print(wgt)
    #sys.exit()

    return fv,wgt,tc,tco

# adds features and weights for single value in copa dict
# if value is list/array: all elements are added with the same weight
# dicts are recursively processed
# IN:
#   fv: featmat of current segment (init: [])
#   wgt: weight vector (init: [])
#   c: current (sub-)dict in copa['data']...[mySegmentIdx]...
#        e.g. 'gst', 'ml', 'rms' (recursively stepped through until
#                                 non-dict value is reached)
#   sel: corresponding dict in copa['augmentation']...['wgt']...
# OUT:
#   fv: updated featvec
#   wgt: updated weigths
# PROCESS:
#   iteration over keys (same in c and sel):
#     if value is dict -> recursive processing
#     if value is list -> collect values, uniform or element-wise weighting
#     if value is scalar-> collect value, weighting
def aug_loc_feat_sub(fv,wgt,c,sel):
    #print('\n\nentering fv_sub ...')
    #print('select:', sel)
    #print('in:', c)
    # robust termination
    if type(c) is not dict:
        #print('robust term!!')
        return fv, wgt
    # over feature(subsets)
    for x in sel:
        #print('... check x:', x)
        if x not in c: continue
        #print('... available')
        if type(sel[x]) is dict:
            #print('--> recursive...')
            # recursive processing
            fv,wgt = aug_loc_feat_sub(fv,wgt,c[x],sel[x])
        elif myl.of_list_type(c[x]):
            #print('--> list term.')
            # list of variables (e.g. coef vector)
            for i in range(len(c[x])):
                # for polycoefs use abs values
                if x=='c':
                    fv = np.append(fv,abs(c[x][i]))
                else:
                    fv = np.append(fv,c[x][i])
                # weight each list item differently or uniformly
                if type(sel[x]) is list:
                    wgt = np.append(wgt,sel[x][i])
                else:
                    wgt = np.append(wgt,sel[x])
            #print('fv upd:', fv)
            #print('wgt upd:', wgt)
        else:
            #print('--> scalar term.')
            # scalar (most features)
            fv = np.append(fv,c[x])
            wgt = np.append(wgt,sel[x])
            #print('fv upd:', fv)
            #print('wgt upd:', wgt)
    return fv, wgt


# copy + adjust annot,opt,copa for acc feature extraction
# (for uniform call of functions in copasul_preproc and _styl)
# IN:
#   t: truncated syllable time stamps
#   to: syllable time stamps
#   pto: parent tier on offsets
#   annot: tg or xml at current state of augmentation
#   i: channelIdx
#   opt: copa['config']
#   lng: file length (for tg)
#   add_mat dict
#       'fi'|'lng'|'ff'|'y'|'t'|'bv'
# OUT:
#   copa (data subdict contains only one file and channel key, resp)
#   t
#   to
def aug_prep_copy(t,to,pto,annot,i,opt,add_mat):
    global f_log

    ## adjust config
    opt = cp.deepcopy(opt)
    # time-sync gnl_* and loc(_ext) time stamps
    # both to be within global segment
    # (otherwise this is only checked for loc(_ext) time stamps
    #  in pp_channel())
    opt['preproc']['loc_sync']=False
    time_key='loc'
    for x in ['acc','decl','gst','class']:
        if x in opt['augment']['loc']['wgt']:
            opt['preproc']['loc_sync']=True
            break
    # get subdict with time info if not provided by loc
    if not opt['preproc']['loc_sync']:
        for x in ['gnl_f0','gnl_en','rhy_f0','rhy_en']:
            if x in opt['augment']['loc']['wgt']:
                time_key=x
                break

    tnc = copp.pp_tiernames(opt['fsys'],'chunk','tier',i)
    tna = copp.pp_tiernames(opt['fsys']['augment'],'chunk','tier',i)
    if (len(tnc)==0 and len(tna)>0):
        opt['fsys']['chunk']=tna

    # temporary tiers for superpos analyses
    tn_loc = 'augloc_loc'
    tn_glob = 'augloc_glob'
    opt['fsys']['augment']['glob']['tier_out_stm']=tn_glob
    opt['fsys']['augment']['loc']['tier_out_stm']=tn_loc
    opt['fsys']['glob']['tier'] = tn_glob
    opt['fsys']['loc']['tier_ag'] = []
    opt['fsys']['loc']['tier_acc'] = tn_loc
    opt['fsys']['gnl_f0']['tier']=[tn_loc]
    opt['fsys']['gnl_en']['tier']=[tn_loc]
    opt['fsys']['bnd']['tier']=[]
    opt['fsys']['rhy_f0']['tier']=[]
    opt['fsys']['rhy_en']['tier']=[]
    for x in tna:
        opt['fsys']['channel'][x]=i

    opt['fsys']['channel']["{}_{}".format(tn_loc,int(i+1))]=i
    opt['fsys']['channel']["{}_{}".format(tn_glob,int(i+1))]=i

    ## f0 is not to be preprocessed again in pp_channel()
    opt['skip_f0'] = True
    ## just inform about missing tiers not needed for augmentation
    # (might be added by augmentation and will be re-read before analyses)
    opt['sloppy'] = True
    ## adjust annotation
    annot = cp.deepcopy(annot)
    annot = aug_annot_upd('glob',annot,{'t':pto},i,opt,add_mat['lng'])
    annot = aug_annot_upd('loc',annot,{'t':to},i,opt,add_mat['lng'])

    ## re-init copa (wih selected f0 and time stamps)
    copa = coin.copa_init(opt)

    ## add preprocessed f0 to copa
    ii = add_mat['fi']
    copa['data']={ii:{i:{'f0':{}}}}
    for x in ['t','y','bv']:
        copa['data'][ii][i]['f0'][x] = add_mat[x]

    copa = copp.pp_channel(copa,opt,add_mat['fi'],i,myl.ea(),
                           annot,add_mat['ff'],f_log)

    # update time stamps
    # (too short segments and segments not parented by global segment
    #  might have been dropped)
    t, to = myl.ea(2)
    for ii in myl.numkeys(copa['data']):
        for i in myl.numkeys(copa['data'][ii]):
            # from loc or gnl_* subdict
            if time_key == 'loc':
                for j in myl.numkeys(copa['data'][ii][i]['loc']):
                    t = np.append(t,copa['data'][ii][i]['loc'][j]['t'][2])
                    to = np.append(t,copa['data'][ii][i]['loc'][j]['to'][0])
            else:
                c = copa['data'][ii][i][time_key][0]
                for j in myl.numkeys(c):
                    t = np.append(t,myl.trunc2(np.mean(c[j]['t'])))
                    to = np.append(t,np.mean(c[j]['to']))

    return copa, t, to

# syllable pre-selection (before feature extraction)
# - if analysis tier empty
#     -> return input
# - if analysis segment tier and criterion='left'|'right'
#     -> return left-|rightmost syllable
# - if analysis point tier
#     -> return analyis tier specs
# IN:
#   t: syl time stamps
#   to: orig time
#   lab_t: syl tier labels
#   at: AG tier time info ([[on off]...] or [[timeStamp]...]
#   ato: orig time
#   lab_at: analysis tier labels
#   opt: copa['config']
# OUT:
#   st: selected syl time stamps
#   sto: orig time
#   slab: labels
#   do_postsel: True|False
#           True if segment analysis tier and most prominent syllable
#           has to be chosen per segment after prom classification, i.e.
#           if ['acc_select']=='max'
def aug_loc_preselect(t,to,lab,at,ato,alab,opt):
    # empty analysis tier
    if len(at)==0:
        return t,to,lab,False
    # analysis point tier
    if myl.ncol(at)==1:
        return at,ato,lab_a,False
    # max selection per segment only possible after classification
    if not re.search('(left|right)',opt['augment']['loc']['acc_select']):
        return t,to,lab,True
    # left/rightmost syllable
    sel = opt['augment']['loc']['acc_select']
    st, sto, slab = copp.pp_read_empty()
    for i in range(len(at)):
        j = myl.find_interval(t,at[i,:])
        if len(j)==0: continue
        if sel=='left':
            jj=j[0]
        else:
            jj=j[-1]
        st = np.append(st,t[jj])
        sto = np.append(sto,to[jj])
        slab.append(lab[jj])
    return st, sto, slab, False

# nearest centroid classification
# IN:
#   typ: 'glob'|'loc'
#   x: feature matrix
#   tc: corresponding time stamp values (bnd for glob, sylncl for loc)
#   wgt: feat weight vector
#   opt: copa['config']
#   is0: set of clear non-bound indices
#   is1: set pre-defined indices of 1 class
#   ij: matrix of [fileIdx channelIdx] (one per matrix row)
# OUT:
#   c: vector, c[j] - +boundary/accent
#   cntr: centroid dict
#      [0|1]: centroidVector
#   wgt: weights (as user input or estimated from data, in case
#       abs + delta values are taken, the weight vector gets double length)
def aug_cntr_wrapper(typ,x,tc,wgt,opt,is0=set(),is1=set(),ij=myl.eai()):
    topt = opt['augment'][typ]

    # threshold to decide which scaling method to choose
    n_sparse = 10

    if len(x)==0:
        return myl.eai(), wgt

    # robustness: convert nan to 0
    x = np.nan_to_num(x)

    # centering + scaling of sparse data
    if len(x) <= n_sparse:
        cs = sp.MaxAbsScaler()
    else:
        cs = sp.RobustScaler()
    x = cs.fit_transform(x)

    # sets to index lists
    i0 = np.asarray(list(is0)).astype(int)
    i1 = np.asarray(list(is1)).astype(int)

    cntr, ww = aug_cntr_wrapper_sub(x,i0,i1,topt)

    # non-user defined weights
    if len(ww)>0: wgt=ww

    ## nearest centroid classification
    # vector of [0,1] for +/- bnd
    if topt['cntr_mtd']=='seed_wgt':
        attract = {0:topt['prct'], 1:100-topt['prct']}
        c = myl.cntr_classif(cntr,x,{'wgt':wgt,'attract':attract})
    else:
        c = myl.cntr_classif(cntr,x,{'wgt':wgt})
    c = np.asarray(c)

    ## set predefined boundaries
    if len(i1)>0: c[i1]=1

    ## correct for min dist
    d = topt['min_l']
    for i in range(len(c)):
        if c[i]==0 or i==0: continue
        ii = (i in is1)
        for j in range(i-1,-1,-1):
            if tc[i]-tc[j] >= d:
                break
            if (len(ij)>0 and
                (ij[i,0] != ij[j,0] or ij[i,1] != ij[j,1])):
                break
            if c[j]==1:
                jj = (j in is1)
                # both definitely boundaries
                if ii and jj:
                    continue
                elif ii:
                    c[j]=0
                    break
                elif jj:
                    c[i]=0
                    break
                xi = myl.wgt_mean(x[i,:],wgt)
                xj = myl.wgt_mean(x[j,:],wgt)
                if xj < xi:
                    c[j]=0
                else:
                    c[i]=0
                #print(i,j,tc[i],'-',tc[j],':',xi,xj,'c',c)
                #myl.stopgo()
                break

    #print('x',x)
    #print('wgt',wgt)
    #print('cntr',cntr)
    #print('c',c)
    #myl.stopgo()

    ## do not ! set last item to globseg boundary
    # since it refers to last inter-segment boundary and not to file end
    #if (typ=='glob' and len(c)>0): c[-1]=1
    return c, cntr, wgt

# transforms current into delta values or returns both
# depending on mtd. On case of both, weight vector is doubled
# (at this stage user-defined, not yet data-derived)
# IN:
#   x: feature matrix
#   wgt: weight vector
#   mtd: 'abs'|'delta'|'abs+delta'
# OUT:
#   x
#   wgt
def aug_fv_measure(x,wgt,mtd):
    if mtd=='delta':
        x = myl.calc_delta(x)
    elif mtd=='abs+delta':
        x = np.concatenate((x,myl.calc_delta(x)),axis=1)
        wgt = np.concatenate((wgt,wgt))
    return x, wgt

# wrapper around clustering methods
# IN:
#   x: feature matrix
#   i0: index vector of 0 seed items in x
#   i1: index vector of 1 seed items in x
# OUT:
#   cntr: dict [0|1] median vectors
#   ww: feature weight vector
def aug_cntr_wrapper_sub(x,i0,i1,topt):

    # data-driven feature weights
    ww = myl.ea()

    ## cluster centroids: cntr_mtd=
    # 'seed_prct': set seed centroids, percentile-based fv-assignment
    # 'seed_kmeans': set/update seed centroids by kmeans clustering
    # 'seed_wgt': set seed centroids, weighted dist fv assignment
    if re.search('^seed',topt['cntr_mtd']) and len(i1)>0:
        cntr, ww = aug_cntr_seed(x,{0:i0,1:i1},topt)
    else:
        # 'split': centroids from fv above, below defined percentile
        cntr, ww = aug_cntr_split(x,topt)

    return cntr, ww


# IN:
#   tc: time stamp array
#   is0: int set of class 0 seeds (indices in tc)
#   is1: int set of class 1 seeds (indices in tc)
#   d: min dist between time stamps
# OUT:
#   is0:  int array of class 0 seeds (indices in tc too close
#         to class 1 items)
def aug_is0(tc,is0,is1,d):
    for i in is1:
        for j in np.arange(i-1,-1,-1):
            if tc[i]-tc[j] >= d: break
            if j in is1: continue
            if tc[i]-tc[j] < d:
                is0.add(j)
        for j in np.arange(i+1,len(tc),1):
            if tc[j]-tc[i] >= d: break
            if j in is1: continue
            if tc[j]<tc[i] < d:
                is0.add(j)
    return is0

# define 2 cluster centroids using seed featvecs
#   1. calc centroids of seed featvecs (for class 0 and 1)
#   2. calc distance of each featvec to this centroid
#   3. determine fv with distance above and below certain percentile
#   4. calc centroid for each of the two groups
# IN:
#   x: feature matrix
#   seed[0|1]: dict of seed candidate index arrays (for classes 0 and 1)
#   opt: config['augment']['glob|loc']
#             needed fields: 'wgt_mtd', 'cntr_mtd'
# OUT:
#  c: dict [1] median of fv among the p closest fv to s[1] seeds
#          [0]
#  w: <[]> feature weights.
def aug_cntr_seed(x,seed,opt):

    ## initial centroids
    c = {0:myl.ea(),1:myl.ea()}
    for j in [0,1]:
        if len(seed[j])>0:
            c[j] = np.median(x[seed[j],:],axis=0)

    ## check requirement >0
    req = min(len(c[0]),len(c[1]))>0

    ## feature weights:
    # e.g. 'silhouette' = feature value separability, based on init centroids
    w = aug_wgt(opt['wgt_mtd'],x,c)

    ## keep initial centroids; nearest centroid classif by 
    ## weighted distance
    if req and opt['cntr_mtd'] == 'seed_wgt':
        return c,w

    ## kmeans
    if req and opt['cntr_mtd'] == 'seed_kmeans':
        km = sc.KMeans(n_clusters=2, init=np.concatenate(([c[0]],[c[1]])),
                       n_init=1)
        km.fit(x)
        c[0] = km.cluster_centers_[0,:]
        c[1] = km.cluster_centers_[1,:]
        return c, w

    ## percentile-based centroid (for less balanced class expectations)
    # nx1 delta-dist vector, 1 element per featvec
    # d = [distTo0/distTo1 ...], i.e. the higher the closer to 0
    d = myl.ea()
    for i in myl.idx_a(len(x)):
        if req:
            d0, d1 = myl.dist_eucl(x[i,:],c[0],w), myl.dist_eucl(x[i,:],c[1],w)
            d = myl.push(d,d0/myl.rob0(d1))
        elif len(c[1])>0:
            d1 = myl.dist_eucl(x[i,:],c[1],w)
            d = myl.push(d,d1)
        elif len(c[0])>0:
            d0 = myl.dist_eucl(x[i,:],c[0],w)
            d = myl.push(d,1/myl.rob0(d0))
            
    px = np.percentile(x,opt['prct'])
    i0, i1 = myl.robust_split(d,px)

    c[0] = np.median(x[i0,:],axis=0)
    c[1] = np.median(x[i1,:],axis=0)
    return c, w
    
# get feature weights
# embedded in aug_cntr_* since 'silhouette' method is to be carried
# out on seed centroid-based classification (see aug_cntr_seed())
# IN:
#   mtd: 'user'|'corr'|'silhouette'
#   x: n x m feature matrix
#   c: dict [0] - class 0 centroid
#           [1] - class 1 centroid
# OUT:
#   w: 1 x m weight vector (empty for mtd='user')
def aug_wgt(mtd,x,c):
    if mtd == 'silhouette' and min(len(c[0]),len(c[1]))>0 and len(x)>2:
        ci = myl.cntr_classif(c,x,{})
        return cocl.featweights(x,ci)
    elif mtd == 'corr' and len(x)>2:
        return myl.feature_weights_unsup(x,'corr')
    return myl.ea()

# define 2 cluster centroids as medians above, below percentile p
# IN:
#  x: feature matrix
#  opt: 
#     'prct': percentile for median calc
#     'wgt_mtd': 'silhouette',...
# OUT:
#  c: dict [0] median below prct
#          [1] median above prct
#  w: feat weight vector <[]> 
def aug_cntr_split(x,opt):
    px = np.percentile(x,opt['prct'],axis=0)
    c={0:myl.ea(),1:myl.ea()}
    # over columns
    for i in range(myl.ncol(x)):
        # values below/above px[i]
        # both incl = so that in any case 2 centroids emerge
        i0, i1 = myl.robust_split(x[:,i],px[i])
        c[0] = np.append(c[0],np.median(x[i0,i]))
        c[1] = np.append(c[1],np.median(x[i1,i]))

    # feature weights
    w = aug_wgt(opt['wgt_mtd'],x,c)

    return c, w

# single glob feature vector
# IN:
#   typ: 'std'|'trend'|'win'
#   b: feature dict
#          ['p']
#          ['ml']|['rng']|...
#               ['rms']|['rms_pre']|['rms_post'] 
#   opt: copa['config']
# OUT:
#   v: featvec
#   wgt: init or left as is if given
# REMARKS:
#   for reset 'r' absolute value is taken 
def aug_glob_fv_sub(typ,b,opt):
    # feat selection dict
    sel = opt['augment']['glob']['wgt'][typ]
    # feature and weight vector
    v, w = myl.ea(2)
    # pause
    if 'p' in sel:
        v = np.append(v,b['p'])
        w = np.append(w,sel['p'])
    # over register discont rep
    for rr in myl.lists('register'):
        for x in b[rr]:
            if x=='r':
                v = np.append(v,abs(b[rr][x]))
            else:
                v = np.append(v,b[rr][x])
            if ((rr in sel) and (x in sel[rr])):
                w = np.append(w,sel[rr][x])
            else:
                w = np.append(w,0)
    return v, w

# standard message generator
# IN:
#   typ: 'syl'|'glob'|'loc'
# OUT:
#   msg dict
#     numeric keys -> myStandardMessage
def aug_msg(typ,f):
    if typ=='chunk':
        prfx = 'chunking'
    elif typ=='syl':
        prfx = 'syllable nucleus extraction'
    elif typ=='glob':
        prfx = 'IP extraction'
    elif typ=='loc':
        prfx = 'accent extraction'
    fstm = myl.stm(f)
    return {1:"Warning! {}: {} carried out without preceding chunking".format(fstm,prfx),
            2:"Fatal! {}: {} requires chunk tier or preceding chunking!".format(fstm,prfx),
            3:"Fatal! {}: {} requires parent tier with segments, not events!".format(fstm,prfx),
            4:"Warning! {}: no analysis tier for {} specified or provided. Fallback: syllable boundaries.".format(fstm,prfx),
            5:"Fatal! {}: neither analysis tier nor syllable nucleus/boundary fallback available for {}. (Re-)Activate syllable nucleus augmentation or define analysis tier.".format(fstm,prfx),
            6:"Warning! {}: {}, no parent tier specified or found. Trying glob augment tier.".format(fstm,prfx),
            7:"Fatal! {}: {}, no parent tier found.".format(fstm,prfx)}


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
    
