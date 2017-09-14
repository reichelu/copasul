#!/usr/bin/env python3

# author: Uwe Reichel, Budapest, 2016

import mylib as myl
import numpy as np
import pandas as pd
import os
import copy as cp
import sys
import re
import copasul_resyn as core
from pandas import DataFrame


### export to f0 files ######
# preproc and residual subdir
### {f0_preproc|f0_residual|f0_resyn}/myStem.f0
### export to csv files #####
# myStem.myFeatset.csv
# myFeatset:
#    gnl_f0_file
#      rows: files
#    gnl_en_file
#      rows: files
#    gnl_f0
#      rows: segments
#    gnl_en
#      rows: segments
#    rhy_f0_file
#      rows: files
#    rhy_en_file
#      rows: files
#    rhy_f0
#      rows: segments
#    rhy_en
#      rows: segments
#    glob
#      rows: global segments
#    loc
#      rows: local segments
#    bnd
#      rows: boundaries

# face "arrays must all be same length" error
# in respective export_loc|glob|...()
# by pasting:
# for x in list(d.keys()):
#    print("{}: {}".format(x,len(d[x])))
# (resp df.keys) in front of exp_to_file()

def export_main(copa):
    if copa['config']['fsys']['export']['csv']:
        export_csv(copa)
    if copa['config']['fsys']['export']['f0_preproc']:
        export_f0(copa,'y')
    if copa['config']['fsys']['export']['f0_residual']:
        export_f0(copa,'r')
    if copa['config']['fsys']['export']['f0_resyn']:
        # resynthesize superpos F0: add ...['f0']['resyn']
        if 'resyn' not in copa['data'][0][0]['f0'].keys():
            copa = core.resyn(copa)
        export_f0(copa,'resyn')


### f0 ##############################################

def export_f0(copa,fld):
    opt = copa['config']
    
    c = copa['data']
    # subdir
    if fld=='y':
        sd = 'f0_preproc'
    elif fld=='r':
        sd = 'f0_residual'
    elif fld=='resyn':
        sd = 'f0_resyn'
    if not os.path.isdir(opt['fsys']['export']['dir']):
        os.mkdir(opt['fsys']['export']['dir'])
    if not os.path.isdir(os.path.join(opt['fsys']['export']['dir'],sd)):
         os.mkdir(os.path.join(opt['fsys']['export']['dir'],sd))
        
    pth = os.path.join(opt['fsys']['export']['dir'],sd)
    # over files
    for ii in myl.numkeys(c):
        fo = "{}.f0".format(os.path.join(pth,c[ii][0]['fsys']['f0']['stm']))
        x = np.asarray([c[ii][0]['f0']['t']]).T
        # over channels
        for i in myl.numkeys(c[ii]):
            if fld not in c[ii][i]['f0']:
                c[ii][i]['f0'][fld]=c[ii][i]['f0']['y']
            x = np.concatenate((x,np.asarray([c[ii][i]['f0'][fld]]).T),axis=1)
            np.savetxt(fo,x,fmt='%.2f')


### csv bracket ######################################

def export_csv(copa):
    c = copa['data']
    opt = copa['config']

    # backward compatibility
    if 'fullpath' not in opt['fsys']['export']:
        opt['fsys']['export']['fullpath'] = False

    if not os.path.isdir(opt['fsys']['export']['dir']):
        os.mkdir(opt['fsys']['export']['dir'])

    # to check whether dict contains resp feature set
    # and not only time information
    fld={'glob':'decl', 'loc':'acc', 'bnd':'std', 'gnl_f0':'std',
         'gnl_en':'std', 'rhy_f0':'rhy', 'rhy_en':'rhy'}
    
    # output file stem
    fo = os.path.join(opt['fsys']['export']['dir'],
                      opt['fsys']['export']['stm'])
    
    # dataFrames to be merged
    #   f0|en -> 'file'|'seg' -> dataFrame
    df_gnl={}
    df_rhy={}

    #print(c[0][0])
    #sys.exit()


    if copa_contains(c,'glob',fld):
        export_glob(c,fo,opt)
    if copa_contains(c,'loc',fld):
        export_loc(c,fo,opt)
    if copa_contains(c,'bnd',fld):
        export_bnd(c,fo,opt)
    if copa_contains(c,'gnl_f0',fld):
        df_gnl['f0']=export_gnl(c,fo,opt,'gnl_f0')
    if copa_contains(c,'gnl_en',fld):
        df_gnl['en']=export_gnl(c,fo,opt,'gnl_en')
    if copa_contains(c,'rhy_f0',fld):
        df_rhy['f0']=export_rhy(c,fo,opt,'rhy_f0')
    if copa_contains(c,'rhy_en',fld):
        df_rhy['en']=export_rhy(c,fo,opt,'rhy_en')

    export_merge(fo,'gnl',df_gnl,opt)
    export_merge(fo,'rhy',df_rhy,opt)


    
#### glob ###################################################

def export_glob(c,fo,opt):
    cn = ['fi','ci','si','stm','t_on','t_off','bv',
          'bl_c1','bl_c0','bl_r','bl_rate',
          'ml_c1','ml_c0','ml_r','ml_rate',
          'tl_c1','tl_c0','tl_r','tl_rate',
          'rng_c1','rng_c0','rng_r','rng_rate',
          'lab','m','sd','med','iqr','max','min','dur']
    if 'class' in c[0][0]['glob'][0]:
        cn.append('class')

    d = init_exp_dict(cn,opt)

    for ii in myl.numkeys(c):
        for i in myl.numkeys(c[ii]):
            for j in myl.numkeys(c[ii][i]['glob']):
                d['fi'].append(ii)
                d['ci'].append(i)
                d['si'].append(j)
                d['stm'].append(c[ii][i]['fsys']['f0']['stm'])
                d['t_on'].append(c[ii][i]['glob'][j]['to'][0])
                if len(c[ii][i]['glob'][j]['to'])>1:
                    d['t_off'].append(c[ii][i]['glob'][j]['to'][1])
                else:
                    d['t_off'].append(c[ii][i]['glob'][j]['to'][0])
                d['bv'].append(c[ii][i]['f0']['bv'])
                d['lab'].append(c[ii][i]['glob'][j]['lab'])
                for x in c[ii][i]['glob'][j]['gnl']:
                    d[x].append(c[ii][i]['glob'][j]['gnl'][x])
                dd = c[ii][i]['glob'][j]['decl']
                for x in myl.lists():
                    d["{}_c1".format(x)].append(dd[x]['c'][0])
                    d["{}_c0".format(x)].append(dd[x]['c'][1])
                    d["{}_r".format(x)].append(dd[x]['r'])
                    d["{}_rate".format(x)].append(dd[x]['rate'])
                if 'class' in cn:
                    d['class'].append(c[ii][i]['glob'][j]['class'])
                # grouping
                d = export_grp_upd(d,c[ii][i]['grp'])

    #for x in list(d.keys()): print("{}: {}".format(x,len(d[x])))

    exp_to_file(d,fo,'glob',fullPath=opt['fsys']['export']['fullpath'])

    return



### loc #############################################

def export_loc(c,fo,opt):
    # base set
    cn = ['fi','ci','si','gi','stm','t_on','t_off','lab_ag',
          'lab_acc','m','sd','med','iqr','max','min','dur']
    # normalized values
    for x in cn:
        y = "{}_nrm".format(x)
        if y in c[0][0]['loc'][0]['gnl']:
            cn.append(y)
    # contour class
    if 'class' in c[0][0]['loc'][0]:
        cn.append('class')
    # gestalt feature set
    cne = ['bl_rms','bl_sd','bl_d_init','bl_d_fin',
           'ml_rms','ml_sd','ml_d_init','ml_d_fin',
           'tl_rms','tl_sd','tl_d_init','tl_d_fin',
           'rng_rms','rng_sd','rng_d_init','rng_d_fin']
    # declination feature set
    cnd = ['bl_c0','bl_c1','bl_rate',
           'ml_c0','ml_c1','ml_rate',
           'tl_c0','tl_c1','tl_rate',
           'rng_c0','rng_c1','rng_rate']
    # poly coef columns for cno and cne dep on polyord
    po = opt['styl']['loc']['ord']
    for i in range(po+1):
        cn.append("c{}".format(i))
        # over register
        for x in myl.lists():
            cne.append("res_{}_c{}".format(x,i))

    # + extended feature set (gst + decl)
    if 'gst' in c[0][0]['loc'][0].keys():
        for x in cne: cn.append(x)
        for x in cnd: cn.append(x)

    d = init_exp_dict(cn,opt)

    for ii in myl.numkeys(c):
        for i in myl.numkeys(c[ii]):
            for j in myl.numkeys(c[ii][i]['loc']):
                #!print("### j is: ", j)
                #!print(c[ii][i]['loc'][j])
                #!print(c[ii][i]['loc'][j]['lab_ag'])
                d['fi'].append(ii)
                d['ci'].append(i)
                d['si'].append(j)
                d['gi'].append(c[ii][i]['loc'][j]['ri'])
                d['stm'].append(c[ii][i]['fsys']['f0']['stm'])
                d['t_on'].append(c[ii][i]['loc'][j]['to'][0])
                if len(c[ii][i]['loc'][j]['to'])>1:
                    d['t_off'].append(c[ii][i]['loc'][j]['to'][1])
                else:
                    d['t_off'].append(c[ii][i]['loc'][j]['to'][0])
                d['lab_ag'].append(c[ii][i]['loc'][j]['lab_ag'])
                d['lab_acc'].append(c[ii][i]['loc'][j]['lab_acc'])
                if 'class' in cn:
                    d['class'].append(c[ii][i]['loc'][j]['class'])
                for x in c[ii][i]['loc'][j]['gnl']:
                    d[x].append(c[ii][i]['loc'][j]['gnl'][x])
                for o in range(po+1):
                    d["c{}".format(o)].append(c[ii][i]['loc'][j]['acc']['c'][po-o])
                # gestalt featset
                if 'bl_rms' in cn:
                    if 'gst' in c[ii][i]['loc'][j]:
                        gg = c[ii][i]['loc'][j]['gst']
                        for x in myl.lists():
                            for y in ['rms','sd','d_init','d_fin']:
                                d["{}_{}".format(x,y)].append(gg[x][y])
                            for o in range(po+1):
                                d["res_{}_c{}".format(x,o)].append(gg['residual'][x]['c'][po-o])
                    else: # e.g. empty segment. TODO, repair already in preproc!!
                        for x in myl.lists():
                            for y in ['rms','sd','d_init','d_fin']:
                                d["{}_{}".format(x,y)].append('NA')
                            for o in range(po+1):
                                d["res_{}_c{}".format(x,o)].append('NA')

                # declination featset
                if 'bl_c1' in cn:
                    if 'decl' in c[ii][i]['loc'][j]:
                        gg = c[ii][i]['loc'][j]['decl']
                        for x in myl.lists():
                            d["{}_c1".format(x)].append(gg[x]['c'][0])
                            d["{}_c0".format(x)].append(gg[x]['c'][1])
                            d["{}_rate".format(x)].append(gg[x]['rate'])
                    else: # e.g. empty segment
                        for x in myl.lists():
                            d["{}_c1".format(x)].append('NA')
                            d["{}_c0".format(x)].append('NA')
                            d["{}_rate".format(x)].append('NA')
                    

                # grouping
                d = export_grp_upd(d,c[ii][i]['grp'])


    # for x in list(d.keys()): print("{}: {}".format(x,len(d[x])))

    exp_to_file(d,fo,'loc',fullPath=opt['fsys']['export']['fullpath'])

    return


### bnd ###########################################

def export_bnd(c,fo,opt):
    cn = ['ci','fi','si','tier','p','lab','lab_next','t_on','t_off']
    # [std|win|trend]_[bl|ml|tl|rng]_[r|rms|rms_pre|rms_post]
    for x in myl.lists('bndtyp'):
        if (x in c[0][0]['bnd'][0][0]):
            for y in myl.lists('register'):
                for z in myl.lists('bndfeat'):
                    cn.append("{}_{}_{}".format(x,y,z))

    # labels of current and next segment
    # (either both interval or both point, thus col is called 'lab')
    d = init_exp_dict(cn,opt)

    # files
    for ii in myl.numkeys(c):
        # channels
        for i in myl.numkeys(c[ii]):
            # tiers
            for j in myl.numkeys(c[ii][i]['bnd']):
                # boundaries
                for k in myl.numkeys(c[ii][i]['bnd'][j]):
                    if not ('std' in c[ii][i]['bnd'][j][k]): continue
                    d['fi'].append(ii)
                    d['ci'].append(i)
                    d['si'].append(k)
                    d['tier'].append(c[ii][i]['bnd'][j][k]['tier'])
                    d['p'].append(c[ii][i]['bnd'][j][k]['std']['p'])
                    d['t_on'].append(c[ii][i]['bnd'][j][k]['std']['t_on'])
                    d['t_off'].append(c[ii][i]['bnd'][j][k]['std']['t_off'])
                    d['lab'].append(c[ii][i]['bnd'][j][k]['lab'])
                    if k+1 in c[ii][i]['bnd'][j]:
                        d['lab_next'].append(c[ii][i]['bnd'][j][k+1]['lab'])
                    else:
                        d['lab_next'].append('')
                    # std|win|trend
                    for w in myl.lists('bndtyp'):
                        if w in c[ii][i]['bnd'][j][k]:
                            # bl|ml|tl|rng
                            for y in myl.lists('register'):
                                # r|rms|...
                                for z in myl.lists ('bndfeat'):
                                    d["{}_{}_{}".format(w,y,z)].append(c[ii][i]['bnd'][j][k][w][y][z])
                    # grouping
                    d = export_grp_upd(d,c[ii][i]['grp'])
  
    # for x in list(d.keys()): print("{}: {}".format(x,len(d[x])))

    exp_to_file(d,fo,'bnd',fullPath=opt['fsys']['export']['fullpath'])

    return


### rhy ##############################################

# typ: 'rhy_f0'|'rhy_en'
# segment- and file-level output
def export_rhy(c,fo,opt,typ):
    # segment-level
    cn = ['fi','ci','si','stm','t_on','t_off','tier','lab','dur']
    # file-level
    typf = "{}_file".format(typ)
    cnf = ['fi','ci','stm','dur']
    # spectral moments
    smo = opt['styl'][typ]['rhy']['nsm']
    for i in myl.idx_seg(1,smo,1):
        cn.append("sm{}".format(i))
        cnf.append("sm{}".format(i))
    # influence of tier_rate (syl, AG etc) in tier (chunk etc) 
    # 'analysisSegment_influenceSegment_mae|prop|rate'
    # tier.tierRate -> TRUE, vs. double-inits on segment level
    tierComb={}
    # tierRate -> True, vs. double-inits on file level
    seen_r={}
    # over interval tiers
    for j in myl.idx_a(len(opt['fsys'][typ]['tier'])):
        tj = opt['fsys'][typ]['tier'][j]
        if tj not in tierComb:
            tierComb[tj]={}
        # over rate tiers (for tj)
        for tk in rhy_rate_tiers(c,tj,opt):
            # segment level tier_tierRate_param
            if tk not in seen_r.keys():
                cn.append("{}_prop".format(tk))
                cn.append("{}_mae".format(tk))
                cn.append("{}_rate".format(tk))
                cnf.append("{}_prop".format(tk))
                cnf.append("{}_mae".format(tk))
                cnf.append("{}_rate".format(tk))
                seen_r[tk]=True
            tierComb[tj][tk]=True
    # segment-level
    d = init_exp_dict(cn,opt)
    # file-level
    df = init_exp_dict(cnf,opt)

    # over files
    for ii in myl.numkeys(c):
        # over channels
        for i in myl.numkeys(c[ii]):
            ## file-level
            if typf not in c[ii][i]: continue
            df['fi'].append(ii)
            df['ci'].append(i)
            df['stm'].append(c[ii][i]['fsys']['f0']['stm'])
            df['dur'].append(c[ii][i][typf]['dur'])
            df = export_grp_upd(df,c[ii][i]['grp'])
            # spec moms
            for q in myl.idx_seg(1,smo,1):
                df["sm{}".format(q)].append(c[ii][i][typf]['sm'][q-1])

            ## influence of tier_rates (syl, AG etc) in file and in other tier (chunk etc)
            ## file-level
            for rt in seen_r:
                for kk in ['rate','mae','prop']:
                    zz = "{}_{}".format(rt,kk)
                    if rt not in c[ii][i][typf]['wgt']:
                        df[zz].append(np.nan)
                    else:
                        df[zz].append(c[ii][i][typf]['wgt'][rt][kk])

            ## segment-level
            if 'rhy' not in c[ii][i][typ][0][0]: continue
            # over tiers
            for j in myl.numkeys(c[ii][i][typ]):
                # over segments
                for k in myl.numkeys(c[ii][i][typ][j]):
                    tt = c[ii][i][typ][j][k]['tier']
                    d['fi'].append(ii)
                    d['ci'].append(i)
                    d['si'].append(k)
                    d['tier'].append(tt)
                    d['stm'].append(c[ii][i]['fsys']['f0']['stm'])
                    d['t_on'].append(c[ii][i][typ][j][k]['to'][0])
                    if len(c[ii][i][typ][j][k]['to'])>1:
                        d['t_off'].append(c[ii][i][typ][j][k]['to'][1])
                    else:
                        d['t_off'].append(c[ii][i][typ][j][k]['to'][0])
                    d['lab'].append(c[ii][i][typ][j][k]['lab'])
                    d['dur'].append(c[ii][i][typ][j][k]['rhy']['dur'])
                    # spec moms
                    for q in myl.idx_seg(1,smo,1):
                        d["sm{}".format(q)].append(c[ii][i][typ][j][k]['rhy']['sm'][q-1])
                    # influence of tier_rates (syl, AG etc) in tier (chunk etc) 
                    for rt in seen_r:
                        #print(tt,rt) ##!!
                        for kk in ['rate','mae','prop']:
                            zz = "{}_{}".format(rt,kk)
                            if rt in c[ii][i][typ][j][k]['rhy']['wgt']:
                                d[zz].append(c[ii][i][typ][j][k]['rhy']['wgt'][rt][kk])
                            # no tier x rateTier combi -> set to nan to get uniform length
                            # for all tierRate_param columns
                            else:
                                d[zz].append(np.nan)
                    # grouping
                    d = export_grp_upd(d,c[ii][i]['grp'])

    #for x in list(d.keys()): print("{}: {}".format(x,len(d[x])))

    exp_to_file(d,fo,typ,fullPath=opt['fsys']['export']['fullpath'])
    exp_to_file(df,fo,typf,fullPath=opt['fsys']['export']['fullpath'])
    return {'seg':d, 'file':df}

# return list of rate tier names for tier t
# IN:
#   c: copa['data']
#   t: tierName
#   opt: copa['config']
# OUT:
#   r: list of rate tier names
def rhy_rate_tiers(c,t,opt):
    # channel idx of t
    if t not in opt['fsys']['channel']:
        return []
    ci = opt['fsys']['channel'][t]
    # find entry in c[0][ci][rhy_f0]
    for i in myl.numkeys(c[0][ci]['rhy_f0']):
        if c[0][ci]['rhy_f0'][i][0]['tier']==t:
            return c[0][ci]['rhy_f0'][i][0]['rhy']['wgt'].keys()
    return []

### gnl ####################################

# typ: 'gnl_f0'|'gnl_en'
# segment- and file-level output
def export_gnl(c,fo,opt,typ):
    typf = "{}_file".format(typ)
    # segment-/file-level
    cn = ['m','sd','med','iqr','max','min','dur']
    cnf = ['m','sd','med','iqr','max','min','dur']
    # en: rms, *_nrm, and sb (*_nrm and sb for segment level only) 
    # f0: *_nrm, bv for file level only
    for x in cp.deepcopy(cn):
        cn.append("{}_nrm".format(x))
    if typ == 'gnl_en':
        cn.append('rms')
        cn.append('rms_nrm')
        #cnf.append('rms')
        cn.append('sb')
    else:
        cnf.append('bv')
        
    for x in ['fi','ci','si','stm','t_on','t_off','tier','lab']:
        cn.append(x)
    for x in ['fi','ci','stm']:
        cnf.append(x)

    # segment-level
    d = init_exp_dict(cn,opt)
    # file-level
    df = init_exp_dict(cnf,opt)

    # files
    for ii in myl.numkeys(c):
        # channels
        for i in myl.numkeys(c[ii]):
            ## file-level
            if typf not in c[ii][i].keys(): continue
            for x in c[ii][i][typf].keys():
                if x in df.keys():
                    df[x].append(c[ii][i][typf][x])
            df['fi'].append(ii)
            df['ci'].append(i)
            df['stm'].append(c[ii][i]['fsys']['f0']['stm'])
            df = export_grp_upd(df,c[ii][i]['grp'])
            if typ=='gnl_f0':
                df['bv'].append(c[ii][i]['f0']['bv'])
            ## segment level
            if typ not in c[ii][i]: continue
            # over segments
            for j in myl.numkeys(c[ii][i][typ]):
                # over tiers    
                for k in myl.numkeys(c[ii][i][typ][j]):
                    d['fi'].append(ii)
                    d['ci'].append(i)
                    d['si'].append(j)
                    d['stm'].append(c[ii][i]['fsys']['f0']['stm'])
                    d['t_on'].append(c[ii][i][typ][j][k]['to'][0])
                    if len(c[ii][i][typ][j][k]['to'])>1:
                        d['t_off'].append(c[ii][i][typ][j][k]['to'][1])
                    else:
                        d['t_off'].append(c[ii][i][typ][j][k]['to'][0])
                    d['tier'].append(c[ii][i][typ][j][k]['tier'])
                    d['lab'].append(c[ii][i][typ][j][k]['lab'])
                    for x in c[ii][i][typ][j][k]['std'].keys():
                        if x in d.keys():
                            d[x].append(c[ii][i][typ][j][k]['std'][x])
                    # grouping
                    d = export_grp_upd(d,c[ii][i]['grp'])

    # for x in list(d.keys()): print("{}: {}".format(x,len(d[x])))

    exp_to_file(d,fo,typ,fullPath=opt['fsys']['export']['fullpath'])
    exp_to_file(df,fo,typf,fullPath=opt['fsys']['export']['fullpath'])
    return {'seg':d, 'file':df}

#### merge export tables ####################################

# merges gnl_f0|en and rhy_f0|en
# IN:
#   fo - outputFileStem
#   infx - infix string 'gnl'|'rhy'
#   dd - dict ['en'|'f0']['seg|file'] -> output of export_gnl or _rhy
#             featSubset  segment-/file-level
# OUT:
#   fileOutput of merged dataFrames 
def export_merge(fo,infx,dd,opt):
    # unique field names
    ff = {'fi','ci','si','stm','t_on','t_off','tier','lab','dur','dur_nrm'}
    pat = '^grp'
    dk = dd.keys()
    if len(dk)<=1: return
    d={}
    df={}
    #'en','f0'
    for x in dk:
        # features, file-level
        for y in dd[x]['file']:
            if ((y not in ff) and (not re.search(pat,y))):
                n = "{}_{}".format(x,y)
            else :
                n = y
            df[n] = dd[x]['file'][y]
        # features, segment-level
        for y in dd[x]['seg']:
            if ((y not in ff) and (not re.search(pat,y))):
                n = "{}_{}".format(x,y)
            else :
                n = y
            d[n] = dd[x]['seg'][y]
    
    exp_to_file(df,fo,"{}_file".format(infx),fullPath=opt['fsys']['export']['fullpath'])
    exp_to_file(d,fo,infx,fullPath=opt['fsys']['export']['fullpath'])
    return


### helpers #################################

# initializes grouping (based on opt['fsys']['grp'])
# updates column name list
# IN:
#   opt   copa['config']
#   cn    columnNameList
# OUT:
#   cn    with added grouping columns with own namespace 'grp_*'
def export_grp_init(opt,cn):
    # unique grouping namespace
    ns = 'grp'
    grp = opt['fsys']['grp']
    if len(grp['lab'])>0:
        for x in grp['lab']:
            if len(x)==0: continue
            cn.append("{}_{}".format(ns,x))
    return cn

# updates filename-based grouping columns in output dict
# IN:
#   d   output dict
#   grp ['data'][ii][i]['grp']
# OUT:
#   d   with updated grouping columns
def export_grp_upd(d,grp):
    ns = 'grp'
    for x in grp.keys():
        d["{}_{}".format(ns,x)].append(grp[x])
    return d


# csv and .R template file output
# IN:
#   d: dict
#   fo: outputFileStem
#   infx: infix
#   checkFld: <'fi'> key to check whether d is empty
#   facpat: <''> pattern additionally to treat as factor
#   fullPath: <False> whether or not to write full path of csv file in
#             R template
def exp_to_file(d,fo,infx,checkFld='fi',facpat='',fullPath=False):
    if ((checkFld in d) and (len(d[checkFld])>0)):
        pd.DataFrame(d).to_csv("{}.{}.csv".format(fo,infx),na_rep='NA',index_label=False, index=False)
        exp_R(d,"{}.{}".format(fo,infx),facpat,fullPath)


# R table read export
# IN: output table, file stem, pattern additionally to treat as factor
def exp_R(d,fo,facpat='',fullPath=False):
    # fo +/- full path
    if fullPath:
        foo = fo
    else:
        foo = os.path.basename(fo)
    # factors (next to grp_*, lab_*)
    fac = {'class','ci','fi','si','gi','stm','tier','spk'}
    o = ["d<-read.table(\"{}.csv\",header=T,fileEncoding=\"UTF-8\",sep =\",\",".format(foo),"\tcolClasses=c("]
    for x in sorted(d.keys()):
        #if ((x in fac) or re.search('^(grp|lab|class)',x)):typ = 'factor'
        if ((x in fac) or re.search('^(grp|lab|class|spk|tier)',x) or
            re.search('_(grp|lab|class|tier)$',x) or
            re.search('_(grp|lab|class|tier)_',x)):
            typ = 'factor'
        elif (len(facpat)>0 and re.search(facpat,x)):
            typ = 'factor'
        else:
            typ = 'numeric'
        z = "{}=\'{}\',".format(x,typ)
        if (not re.search(',$',o[-1])):
            o[-1] += z
        else:
            o.append("\t\t"+z)
    o[-1] = o[-1].replace(',','))')
    myl.output_wrapper(o,"{}.R".format(fo),'list')

# export dict init from colnames in list C (-> keys)
# add grouping columns
def init_exp_dict(c,opt):

    # grouping columns from filename
    c = export_grp_init(opt,c)

    d={}
    for x in c:
        d[x] = []
    return d

# checks whether copa dict contains domain-related feature sets
# IN:
#  c - copa['data']
#  dom - domain 'glob'|'loc'|'bnd' etc
#  ft - dict domain -> characteristic feature set ('std' etc)
# OUT:
#  boolean
def copa_contains(c,dom,ft):
    if ((0 in c.keys()) and (0 in c[0].keys()) and
        (dom in c[0][0].keys()) and (0 in c[0][0][dom]) and
        ((ft[dom] in c[0][0][dom][0]) or
         ((0 in c[0][0][dom][0]) and (ft[dom] in c[0][0][dom][0][0])))):
        return True
    return False

    
 
 
