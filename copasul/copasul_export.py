import copy as cp
import csv
import numpy as np
import os
import pandas as pd
import re
import sys

import copasul.copasul_resyn as core
import copasul.copasul_utils as utils


### export to f0 files ######
# preproc and residual subdir
# {f0_preproc|f0_residual|f0_resyn}/myStem.f0
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
# adds subdict "export" to copa

def export_main(copa):

    '''
    face "arrays must all be same length" error
    in respective export_loc|glob|...()
    by pasting:
    for x in list(d.keys()):
       print(f"{x}: {len(d[x])}")
    (resp df.keys) in front of exp_to_file()
    '''

    if copa['config']['fsys']['export']['csv']:
        copa = export_csv(copa)
    if copa['config']['fsys']['export']['summary']:
        export_summary(copa)
    if copa['config']['fsys']['export']['f0_preproc']:
        export_f0(copa, 'y')
    if copa['config']['fsys']['export']['f0_residual']:
        export_f0(copa, 'r')
    if copa['config']['fsys']['export']['f0_resyn']:
        # resynthesize superpos F0: add ...['f0']['resyn']
        if 'resyn' not in copa['data'][0][0]['f0'].keys():
            copa = core.resyn(copa)
        export_f0(copa, 'resyn')

    return copa


def export_f0(copa, fld):

    '''
    f0 export
    '''

    opt = copa['config']

    c = copa['data']

    # subdir
    if fld == 'y':
        sd = 'f0_preproc'
    elif fld == 'r':
        sd = 'f0_residual'
    elif fld == 'resyn':
        sd = 'f0_resyn'

    if not os.path.isdir(opt['fsys']['export']['dir']):
        os.mkdir(opt['fsys']['export']['dir'])
    if not os.path.isdir(os.path.join(opt['fsys']['export']['dir'], sd)):
        os.mkdir(os.path.join(opt['fsys']['export']['dir'], sd))

    pth = os.path.join(opt['fsys']['export']['dir'], sd)

    # over files
    for ii in utils.sorted_keys(c):

        u = os.path.join(pth, c[ii][0]['fsys']['f0']['stm'])
        fo = f"{u}.f0"
        x = np.asarray([c[ii][0]['f0']['t']]).T

        # over channels
        for i in utils.sorted_keys(c[ii]):
            if fld not in c[ii][i]['f0']:
                c[ii][i]['f0'][fld] = c[ii][i]['f0']['y']
            x = np.concatenate(
                (x, np.asarray([c[ii][i]['f0'][fld]]).T), axis=1)
            np.savetxt(fo, x, fmt='%.2f')
            

def export_summary(copa):

    '''
    summary csv
    reads all feature csv tables outputted by export:csv
    and computes mean and variance scores for all variables
    
    Args:
      copa
    
    Returns:
      files: <output_stem>.summary.csv, 1 row per channel
                                  .R input template
    VAR:
      suma[fileIdx][channelIndex][fset][featKey][v|m|sd|med|iqr|h|g|u]
      v: array of values on which m|sd|med|iqr|h is calculated
      m, med: mean values
      s, iqr: variance
      h: entropy (for contour classes only)
      g: grouping value
      u: 'segment'|'file', features from segment or file level
    featKey := feat_analysisTier in case fset contains the column 'tier'
    '''

    opt = copa['config']

    # output files stem
    fo = utils.fsys_stm(opt, 'export')

    # list of feature sets
    fset = utils.lists('featsets', 'list')

    # general factor variables
    fac = utils.lists('factors', 'set')
    suma = {}

    found_csv = False

    # over feature sets
    for fs in sorted(fset):
        for u in ['segment', 'file']:
            if u == 'segment':
                f = f"{fo}.{fs}.csv"
            else:
                f = f"{fo}.{fs}_file.csv"
            if not os.path.isfile(f):
                continue

            found_csv = True

            d = utils.input_wrapper(
                f, 'csv', {'sep': opt['fsys']['export']['sep']})
            fileIdx = utils.uniq(d['fi'])
            channelIdx = utils.uniq(d['ci'])
            d = exp_dataframe(d)
            
            # over file indices
            for fi in fileIdx:
                for ci in channelIdx:
                    suma = upd_suma(suma, fs, d, fi, ci, fac, u)

    if not found_csv:
        sys.exit("Nothing to summarize. Output csv tables first.")

    # from suma to export dict
    fp = opt['fsys']['export']['fullpath']
    sep = opt['fsys']['export']['sep']
    suma2exp(suma, fo, fp, sep)


def upd_suma(suma, fs_in, d, fi, ci, fac, u):

    '''
    update suma dict (see export_summary())
    
    Args:
      suma dict
      fs_in: featSet name
      d: input dict from csv file
      fi: file index
      ci: channel index
      fac: list of factor variable names
      u: unit 'segment'|'file' (to indicate whether gnl_en or gnl_en_file
                               has been read)
    
    Returns:
      suma updated
    '''

    if u == 'file':
        fs = f"{fs_in}_{u}"
    else:
        fs = fs_in
    if fi not in suma:
        suma[fi] = {ci: {fs: {}}}
    elif ci not in suma[fi]:
        suma[fi][ci] = {fs: {}}
    elif fs not in suma[fi][ci]:
        suma[fi][ci][fs] = {}
        
    # subset for resp. file idx
    dd = d[d.fi == fi]
    dd = dd[dd.ci == ci]

    # analysis tiers if any (featsets gnl, rhy, etc)
    if 'tier' in dd:
        tiers = utils.uniq(dd['tier'])
    else:
        tiers = ['*']

    # over tiers
    for t in tiers:
        if t == '*':
            ds = dd
        else:
            ds = dd[dd.tier == t]

        # over features
        for x in ds.keys():
            suma = upd_suma_feat(suma, fs, ds, fi, ci, fac, u, x, t)

    return suma



def upd_suma_feat(suma, fs, ds, fi, ci, fac, u, x, t):

    '''
    called by upd_suma for single feat
    
    Args:
      see upd_suma()
      x: featName
      t: tierName
    
    Returns:
      suma upd
    '''


    if t == '*':
        # key = x
        key = f"CHANNEL{int(ci) + 1}_{x}"
    else:
        key = f"{t}_{x}"

    # file-level grouping
    if (x == 'stm' or re.search(r'^grp_', x)):
        v = list(ds[x])
        suma[fi][ci][fs][key] = {'g': v[0], 'u': u}
        return suma
    
    # skip factor variables (except of class)
    if ((x in fac) or re.search(r'^(lab|spk|tier)', x) or
        re.search(r'_(lab|tier)$', x) or
            re.search(r'_(lab|tier)_', x)):
        return suma

    v = np.asarray(ds[x])
    # feat values: NA -> np.nan -> remove
    v = utils.nan_repl(ds[x])
    v = v.astype('float')
    v = utils.rm_nan(v)

    # contour class feat
    if x == 'class':
        suma[fi][ci][fs][key] = {'v': v, 'h': np.nan, 'u': u}
        if len(v) > 0:
            suma[fi][ci][fs][key]['h'] = utils.list2entrop(v)
        return suma

    # all other features
    suma[fi][ci][fs][key] = {'v': v, 'm': np.nan, 'sd': np.nan,
                             'med': np.nan, 'iqr': np.nan, 'u': u}

    # segment- or file-level
    if u == 'segment':
        if len(v) > 0:
            suma[fi][ci][fs][key]['m'] = np.mean(v)
            suma[fi][ci][fs][key]['med'] = np.median(v)
            suma[fi][ci][fs][key]['sd'] = np.std(v)
            suma[fi][ci][fs][key]['iqr'] = np.percentile(v, 75) - np.percentile(v, 25)
    else:
        if len(v) == 0:
            suma[fi][ci][fs][key]['v'] = [np.nan]

    return suma


def suma2exp(suma, fo, fp, sep):

    '''
    summary -> export dict -> .csv|R
    
    Args:
      suma: summary dict (see export_summary())
      fo: output stem
      fp: fullPath for csv reference in R output True|False from opt
      sep: column separator
    
    Returns:
      csv and R file; 1 row per file
    
    VAR:
    exp export dict
      'fi' -> [fileIndices]
      'ci' -> [channelIndices]
      myGrouping -> [fileLevel grouping vars]
      myFeatSet_myFeat_myMeas -> [featureValMeansAndVars]
    '''

    exp = suma2exp_init(suma)

    # incremental d update
    for fi in utils.sorted_keys(suma):
        for ci in utils.sorted_keys(suma[fi]):
            exp = suma2exp_upd(exp, suma, fi, ci)

    # remove redundant grouping columns
    exp = exp_rm_redun(exp)
    exp_to_file(exp, fo, 'summary', checkFld='fi',
                facpat='', fullPath=fp, sep=sep)


def exp_rm_redun(exp):

    '''
    remove redundant grouping and stem columns from export dict
    
    Args:
      exp: export dict
    
    Returns:
      exp: without redundant columns
    '''


    # over column names
    cnn = utils.sorted_keys(exp)
    for cn in cnn:
        if not re.search(r'_(grp_|stm)', cn):
            continue
        nc = cn
        nc = re.sub(r'^(.+?)_grp', 'grp', nc)
        nc = re.sub(r'^(.+?)_stm', 'stm', nc)
        if nc not in exp:
            exp[nc] = cp.deepcopy(exp[cn])
        del exp[cn]

    return exp


def suma2exp_init(suma):

    '''
    inits export dict exp from summary dict suma
    
    Args:
      suma: (dict)
    
    Returns:
      exp: (dict)
    '''

    i = utils.sorted_keys(suma)
    fi = i[0]
    i = utils.sorted_keys(suma[fi])
    exp = {'fi': []}
    # over channel idx (since tier names as IP_1 and IP_2 are part of keys)
    for ci in utils.sorted_keys(suma[fi]):
        # over featsets
        for fs in suma[fi][ci]:
            # over features/groupings
            for x in suma[fi][ci][fs]:
                # file or segment level features
                if suma[fi][ci][fs][x]['u'] == 'file':
                    key = exp_suma_key(fs, x, 'g')
                    exp[key] = []
                else:
                    # over subfields m,sd,med...
                    for s in suma[fi][ci][fs][x]:
                        if re.search(r'[vu]', s):
                            continue
                        key = exp_suma_key(fs, x, s)
                        exp[key] = []

    return exp


def suma2exp_upd(exp, suma, fi, ci):

    '''
    update export dict
    
    Args:
      exp: export dict (becoming csv and R file later on)
      suma: summary dict
      fi: file index
      ci: channel index
    
    Returns:
      d: updated
    '''

    if ci == 0 or ci == '0':
        exp['fi'].append(fi)
        
    # over featsets
    for fs in suma[fi][ci]:
        # over features/groupings
        for x in suma[fi][ci][fs]:
            if suma[fi][ci][fs][x]['u'] == 'file':
                key = exp_suma_key(fs, x, 'g')

                # featval or grouping val?
                if 'v' in suma[fi][ci][fs][x]:
                    s = 'v'
                else:
                    s = 'g'
                exp[key].append(suma[fi][ci][fs][x][s][0])
                continue

            # over subfields m,sd,med...
            for s in suma[fi][ci][fs][x]:
                if re.search(r'[vu]', s):
                    continue
                key = exp_suma_key(fs, x, s)
                exp[key].append(suma[fi][ci][fs][x][s])
    return exp


def exp_suma_key(fs, x, s):

    '''
    generates key for output dict
    grouping variable: <featSet>_<varName>
    other variables: <featSet>_<varName>_{m|med|...}
    
    Args:
      fs: featSetName
      x:  varName
      s:  statMeasName
    
    Returns:
      key
    '''

    if s == 'g':
        key = f"{fs}_{x}"
    else:
        key = f"{fs}_{x}_{s}"
        
    return key


def export_csv(copa):

    '''
    csv bracket
    '''

    c = copa['data']
    opt = copa['config']

    # backward compatibility
    if 'fullpath' not in opt['fsys']['export']:
        opt['fsys']['export']['fullpath'] = False
    if 'sep' not in opt['fsys']['export']:
        opt['fsys']['export']['sep'] = ','

    # export directory
    if not os.path.isdir(opt['fsys']['export']['dir']):
        os.mkdir(opt['fsys']['export']['dir'])

    # obligatory subfields to check whether dict contains
    # resp feature set and not only time information
    fld = {'glob': 'decl', 'loc': 'acc', 'bnd': 'std', 'gnl_f0': 'std',
           'gnl_en': 'std', 'rhy_f0': 'rhy', 'rhy_en': 'rhy',
           'voice': 'jit'}

    # output file stem
    fo = utils.fsys_stm(opt, 'export')

    # dataFrames to be merged
    #   f0|en -> 'file'|'seg' -> dataFrame
    df_gnl = {}
    df_rhy = {}


    copa["export"] = {"glob": None,
                      "loc": None,
                      "bnd": None,
                      "voice": None,
                      "gnl_f0": None,
                      "gnl_en": None,
                      "gnl_f0_file": None,
                      "gnl_en_file": None,
                      "rhy_f0": None,
                      "rhy_en": None,
                      "rhy_f0_file": None,
                      "rhy_en_file": None}

    # global f0 features
    if copa_contains(c, 'glob', fld):
        copa['export']['glob'] = export_glob(c, fo, opt)

    # local f0 features
    if copa_contains(c, 'loc', fld):
        copa['export']['loc'] = export_loc(c, fo, opt)

    # boundary features
    if copa_contains(c, 'bnd', fld):
        copa['export']['bnd'] = export_bnd(c, fo, opt)

    # voice quality
    if copa_contains(c, 'voice', fld):
        copa['export']['voice'] = export_voice(c, fo, opt)

    # standard f0 features
    if copa_contains(c, 'gnl_f0', fld):
        df_gnl['f0'] = export_gnl(c, fo, opt, 'gnl_f0')
        try:
            copa['export']['gnl_f0'] = exp_dataframe(df_gnl['f0']['seg'])
        except:
            copa['export']['gnl_f0'] = None
        try:
            copa['export']['gnl_f0_file'] = exp_dataframe(df_gnl['f0']['file'])
        except:
            copa['export']['gnl_f0_file'] = None

    # standard energy features
    if copa_contains(c, 'gnl_en', fld):
        df_gnl['en'] = export_gnl(c, fo, opt, 'gnl_en')
        try:
            copa['export']['gnl_en'] = exp_dataframe(df_gnl['en']['seg'])
        except:
            copa['export']['gnl_en'] = None
        try:
            copa['export']['gnl_en_file'] = exp_dataframe(df_gnl['en']['file'])
        except:
            copa['export']['gnl_en_file'] = None

    # f0 rhythm features
    if copa_contains(c, 'rhy_f0', fld):
        df_rhy['f0'] = export_rhy(c, fo, opt, 'rhy_f0')
        try:
            copa['export']['rhy_f0'] = exp_dataframe(df_rhy['f0']['seg'])
        except:
            copa['export']['rhy_f0'] = None
        try:
            copa['export']['rhy_f0_file'] = exp_dataframe(df_rhy['f0']['file'])
        except:
            copa['export']['rhy_f0_file'] = None

    # energy rhythm features
    if copa_contains(c, 'rhy_en', fld):
        df_rhy['en'] = export_rhy(c, fo, opt, 'rhy_en')
        try:
            copa['export']['rhy_en'] = exp_dataframe(df_rhy['en']['seg'])
        except:
            copa['export']['rhy_en'] = None
        try:
            copa['export']['rhy_en_file'] = exp_dataframe(df_rhy['en']['file'])
        except:
            copa['export']['rhy_en_file'] = None

    export_merge(fo, 'gnl', df_gnl, opt)
    export_merge(fo, 'rhy', df_rhy, opt)

    return copa


def export_glob(c, fo, opt):

    '''
    fset glob
    '''

    cn = ['fi', 'ci', 'si', 'stm', 't_on', 't_off', 'bv',
          'bl_c1', 'bl_c0', 'bl_r', 'bl_rate', 'bl_m',
          'ml_c1', 'ml_c0', 'ml_r', 'ml_rate', 'ml_m',
          'tl_c1', 'tl_c0', 'tl_r', 'tl_rate', 'tl_m',
          'rng_c1', 'rng_c0', 'rng_r', 'rng_rate', 'rng_m',
          'lab', 'm', 'sd', 'med', 'iqr', 'max', 'maxpos', 'min', 'dur',
          'is_init_chunk', 'is_fin_chunk', 'tier',
          'tl_ml_cross_f0', 'tl_ml_cross_t',
          'tl_bl_cross_f0', 'tl_bl_cross_t',
          'ml_bl_cross_f0', 'ml_bl_cross_t',
          'tl_drop', 'ml_drop', 'bl_drop', 'rng_drop']

    if 'class' in c[0][0]['glob'][0]:
        cn.append('class')

    d = init_exp_dict(cn, opt)

    for ii in utils.sorted_keys(c):
        for i in utils.sorted_keys(c[ii]):
            for j in utils.sorted_keys(c[ii][i]['glob']):
                d['fi'].append(ii)
                d['ci'].append(i)
                d['si'].append(j)
                d['stm'].append(c[ii][i]['fsys']['f0']['stm'])
                d['t_on'].append(c[ii][i]['glob'][j]['to'][0])
                if len(c[ii][i]['glob'][j]['to']) > 1:
                    d['t_off'].append(c[ii][i]['glob'][j]['to'][1])
                else:
                    d['t_off'].append(c[ii][i]['glob'][j]['to'][0])
                d['bv'].append(c[ii][i]['f0']['bv'])
                d['lab'].append(c[ii][i]['glob'][j]['lab'])
                d['is_init_chunk'].append(c[ii][i]['glob'][j]['is_init_chunk'])
                d['is_fin_chunk'].append(c[ii][i]['glob'][j]['is_fin_chunk'])
                d['tier'].append(c[ii][i]['glob'][j]['tier'])
                for x in c[ii][i]['glob'][j]['gnl']:
                    d[x].append(c[ii][i]['glob'][j]['gnl'][x])
                dd = c[ii][i]['glob'][j]['decl']
                eou = c[ii][i]['glob'][j]['eou']
                for x in utils.lists():
                    d[f"{x}_c1"].append(dd[x]['c'][0])
                    d[f"{x}_c0"].append(dd[x]['c'][1])
                    d[f"{x}_r"].append(dd[x]['r'])
                    d[f"{x}_rate"].append(dd[x]['rate'])
                    d[f"{x}_m"].append(dd[x]['m'])
                    q = f"{x}_drop"
                    d[q].append(eou[q])
                if 'class' in cn:
                    d['class'].append(c[ii][i]['glob'][j]['class'])
                # line crossings
                for ra in utils.lists():
                    for rb in utils.lists():
                        qx = f"{ra}_{rb}_cross_t"
                        qy = f"{ra}_{rb}_cross_f0"
                        if qx not in eou:
                            continue
                        d[qx].append(eou[qx])
                        d[qy].append(eou[qy])
                # grouping
                d = export_grp_upd(d, c[ii][i]['grp'])

    exp_to_file(d, fo, 'glob', fullPath=opt['fsys']['export']['fullpath'],
                sep=opt['fsys']['export']['sep'])

    try:
        return exp_dataframe(d)
    except:
        return None



def export_loc(c, fo, opt):

    '''
    fset loc
    '''

    # base set
    cn = ['fi', 'ci', 'si', 'gi', 'stm', 't_on', 't_off', 'lab_ag',
          'lab_acc', 'm', 'sd', 'med', 'iqr', 'max', 'maxpos', 'min', 'dur',
          'is_init', 'is_fin', 'is_init_chunk', 'is_fin_chunk',
          'tier_ag', 'tier_acc']

    # normalized values
    for x in cn:
        y = f"{x}_nrm"
        if y in c[0][0]['loc'][0]['gnl']:
            cn.append(y)

    # contour class
    if 'class' in c[0][0]['loc'][0]:
        cn.append('class')

    # gestalt feature set
    cne = ['bl_rms', 'bl_sd', 'bl_d_init', 'bl_d_fin', 'bl_d',
           'ml_rms', 'ml_sd', 'ml_d_init', 'ml_d_fin', 'ml_d',
           'tl_rms', 'tl_sd', 'tl_d_init', 'tl_d_fin', 'tl_d',
           'rng_rms', 'rng_sd', 'rng_d_init', 'rng_d_fin', 'rng_d']

    # declination feature set
    cnd = ['bl_c0', 'bl_c1', 'bl_rate', 'bl_m',
           'ml_c0', 'ml_c1', 'ml_rate', 'ml_m',
           'tl_c0', 'tl_c1', 'tl_rate', 'tl_m',
           'rng_c0', 'rng_c1', 'rng_rate', 'rng_m']

    # area under polyval
    cn.append("rmsd")

    # poly coef columns for cno and cne dep on polyord
    po = opt['styl']['loc']['ord']
    for i in range(po+1):
        cn.append(f"c{i}")
        cn.append(f"rms_c{i}")
        # over register types
        for x in utils.lists():
            cne.append(f"res_{x}_c{i}")

    # + extended feature set (gst + decl)
    if 'gst' in c[0][0]['loc'][0].keys():
        for x in cne:
            cn.append(x)
        for x in cnd:
            cn.append(x)

    d = init_exp_dict(cn, opt)

    for ii in utils.sorted_keys(c):
        for i in utils.sorted_keys(c[ii]):
            for j in utils.sorted_keys(c[ii][i]['loc']):
                d['fi'].append(ii)
                d['ci'].append(i)
                d['si'].append(j)
                d['gi'].append(c[ii][i]['loc'][j]['ri'])
                d['is_init'].append(c[ii][i]['loc'][j]['is_init'])
                d['is_fin'].append(c[ii][i]['loc'][j]['is_fin'])
                d['is_init_chunk'].append(c[ii][i]['loc'][j]['is_init_chunk'])
                d['is_fin_chunk'].append(c[ii][i]['loc'][j]['is_fin_chunk'])
                d['stm'].append(c[ii][i]['fsys']['f0']['stm'])
                d['t_on'].append(c[ii][i]['loc'][j]['to'][0])
                if len(c[ii][i]['loc'][j]['to']) > 1:
                    d['t_off'].append(c[ii][i]['loc'][j]['to'][1])
                else:
                    d['t_off'].append(c[ii][i]['loc'][j]['to'][0])
                d['lab_ag'].append(c[ii][i]['loc'][j]['lab_ag'])
                d['lab_acc'].append(c[ii][i]['loc'][j]['lab_acc'])
                d['tier_ag'].append(c[ii][i]['loc'][j]['tier_ag'])
                d['tier_acc'].append(c[ii][i]['loc'][j]['tier_acc'])
                if 'class' in cn:
                    d['class'].append(c[ii][i]['loc'][j]['class'])
                for x in c[ii][i]['loc'][j]['gnl']:
                    d[x].append(c[ii][i]['loc'][j]['gnl'][x])
                for o in range(po+1):
                    d[f"c{o}"].append(
                        c[ii][i]['loc'][j]['acc']['c'][po-o])
                    d[f"rms_c{o}"].append(
                        c[ii][i]['loc'][j]['acc']['rms'][po-o])
                d["rmsd"].append(c[ii][i]['loc'][j]['acc']['rmsd'])

                # gestalt featset
                if 'bl_rms' in cn:
                    if 'gst' in c[ii][i]['loc'][j]:
                        gg = c[ii][i]['loc'][j]['gst']
                        for x in utils.lists():
                            for y in ['rms', 'sd', 'd_init', 'd_fin', 'd']:
                                d[f"{x}_{y}"].append(gg[x][y])
                            for o in range(po + 1):
                                d[f"res_{x}_c{o}"].append(
                                    gg['residual'][x]['c'][po-o])
                    else:  # e.g. empty segment. TODO, repair already in preproc!!
                        for x in utils.lists():
                            for y in ['rms', 'sd', 'd_init', 'd_fin', 'd']:
                                d[f"{x}_{y}"].append('NA')
                            for o in range(po + 1):
                                d[f"res_{x}_c{o}"].append('NA')

                # declination featset
                if 'bl_c1' in cn:
                    if 'decl' in c[ii][i]['loc'][j]:
                        gg = c[ii][i]['loc'][j]['decl']
                        for x in utils.lists():
                            d[f"{x}_c1"].append(gg[x]['c'][0])
                            d[f"{x}_c0"].append(gg[x]['c'][1])
                            d[f"{x}_rate"].append(gg[x]['rate'])
                            d[f"{x}_m"].append(gg[x]['m'])
                    else:  # e.g. empty segment
                        for x in utils.lists():
                            d[f"{x}_c1"].append('NA')
                            d[f"{x}_c0"].append('NA')
                            d[f"{x}_rate"].append('NA')
                            d[f"{x}_m"].append('NA')

                # grouping
                d = export_grp_upd(d, c[ii][i]['grp'])

    exp_to_file(d, fo, 'loc', fullPath=opt['fsys']['export']['fullpath'],
                sep=opt['fsys']['export']['sep'])

    try:
        return exp_dataframe(d)
    except:
        return None



def export_bnd(c, fo, opt):

    '''
    fset bnd
    '''

    cn = ['ci', 'fi', 'si', 'tier', 'p', 'lab', 'lab_next', 't_on', 't_off',
          'is_init', 'is_fin', 'is_init_chunk', 'is_fin_chunk']
    # [std|win|trend]_[bl|ml|tl|rng]_[r|rms|rms_pre|rms_post]

    for x in utils.lists('bndtyp'):
        if (x in c[0][0]['bnd'][0][0]):
            for y in utils.lists('register'):
                for z in utils.lists('bndfeat'):
                    cn.append(f"{x}_{y}_{z}")

    # labels of current and next segment
    # (either both interval or both point, thus col is called 'lab')
    d = init_exp_dict(cn, opt)

    # files
    for ii in utils.sorted_keys(c):
        # channels
        for i in utils.sorted_keys(c[ii]):
            # tiers
            for j in utils.sorted_keys(c[ii][i]['bnd']):
                # boundaries
                for k in utils.sorted_keys(c[ii][i]['bnd'][j]):
                    if not ('std' in c[ii][i]['bnd'][j][k]):
                        continue
                    d['fi'].append(ii)
                    d['ci'].append(i)
                    d['si'].append(k)
                    d['tier'].append(c[ii][i]['bnd'][j][k]['tier'])
                    d['p'].append(c[ii][i]['bnd'][j][k]['std']['p'])
                    d['t_on'].append(c[ii][i]['bnd'][j][k]['std']['t_on'])
                    d['t_off'].append(c[ii][i]['bnd'][j][k]['std']['t_off'])
                    d['lab'].append(c[ii][i]['bnd'][j][k]['lab'])
                    d['is_init'].append(c[ii][i]['bnd'][j][k]['is_init'])
                    d['is_fin'].append(c[ii][i]['bnd'][j][k]['is_fin'])
                    d['is_init_chunk'].append(
                        c[ii][i]['bnd'][j][k]['is_init_chunk'])
                    d['is_fin_chunk'].append(
                        c[ii][i]['bnd'][j][k]['is_fin_chunk'])
                    if k + 1 in c[ii][i]['bnd'][j]:
                        d['lab_next'].append(c[ii][i]['bnd'][j][k+1]['lab'])
                    else:
                        d['lab_next'].append('')
                    # std|win|trend
                    for w in utils.lists('bndtyp'):
                        if w in c[ii][i]['bnd'][j][k]:
                            # bl|ml|tl|rng
                            for y in utils.lists('register'):
                                # r|rms|...
                                for z in utils.lists('bndfeat'):
                                    d[f"{w}_{y}_{z}"].append(
                                        c[ii][i]['bnd'][j][k][w][y][z])
                    # grouping
                    d = export_grp_upd(d, c[ii][i]['grp'])

    exp_to_file(d, fo, 'bnd', fullPath=opt['fsys']['export']['fullpath'],
                sep=opt['fsys']['export']['sep'])

    try:
        return exp_dataframe(d)
    except:
        return None


def export_rhy(c, fo, opt, typ):

    '''
    fset rhy
    typ: 'rhy_f0'|'rhy_en'
    segment- and file-level output
    '''

    # segment-level
    cn = ['fi', 'ci', 'si', 'stm', 't_on', 't_off', 'tier', 'lab', 'dur',
          'f_max', 'n_peak',
          'is_init', 'is_fin', 'is_init_chunk', 'is_fin_chunk']

    # file-level
    typf = f"{typ}_file"
    cnf = ['fi', 'ci', 'stm', 'dur', 'f_max', 'n_peak']

    # spectral moments
    smo = opt['styl'][typ]['rhy']['nsm']
    for i in utils.idx_seg(1, smo, 1):
        cn.append(f"sm{i}")
        cnf.append(f"sm{i}")

    # influence of tier_rate (syl, AG etc) in tier (chunk etc)
    # 'analysisSegment_influenceSegment_mae|prop|rate'
    # tier.tierRate -> TRUE, vs. double-inits on segment level
    tierComb = {}
    # tierRate -> True, vs. double-inits on file level
    seen_r = {}
    # over interval tiers
    for j in utils.idx_a(len(opt['fsys'][typ]['tier'])):
        tj = opt['fsys'][typ]['tier'][j]
        if tj not in tierComb:
            tierComb[tj] = {}
        # over rate tiers (for tj)
        for tk in rhy_rate_tiers(c, tj, opt):
            # segment level tier_tierRate_param
            if tk not in seen_r.keys():
                cn.append(f"{tk}_prop")
                cn.append(f"{tk}_mae")
                cn.append(f"{tk}_rate")
                cn.append(f"{tk}_dlm")
                cn.append(f"{tk}_dgm")
                cnf.append(f"{tk}_prop")
                cnf.append(f"{tk}_mae")
                cnf.append(f"{tk}_rate")
                cnf.append(f"{tk}_dlm")
                cnf.append(f"{tk}_dgm")
                seen_r[tk] = True
            tierComb[tj][tk] = True
            
    # segment-level
    d = init_exp_dict(cn, opt)

    # file-level
    df = init_exp_dict(cnf, opt)

    # over files
    for ii in utils.sorted_keys(c):
        # over channels
        for i in utils.sorted_keys(c[ii]):
            # file-level
            if typf not in c[ii][i]:
                continue
            df['fi'].append(ii)
            df['ci'].append(i)
            df['stm'].append(c[ii][i]['fsys']['f0']['stm'])
            df['dur'].append(c[ii][i][typf]['dur'])
            df = export_grp_upd(df, c[ii][i]['grp'])
            # spec moms
            for q in utils.idx_seg(1, smo, 1):
                df[f"sm{q}"].append(c[ii][i][typf]['sm'][q-1])

            # freq of glob amplitude maximum and num of local peaks
            for q in ['f_max', 'n_peak']:
                df[q].append(c[ii][i][typf][q])

            # influence of tier_rates (syl, AG etc) in file and in other tier (chunk etc)
            # file-level
            for rt in seen_r:
                for kk in ['rate', 'mae', 'prop', 'dlm', 'dgm']:
                    zz = f"{rt}_{kk}"
                    if rt not in c[ii][i][typf]['wgt']:
                        df[zz].append(np.nan)
                    else:
                        df[zz].append(c[ii][i][typf]['wgt'][rt][kk])

            # segment-level
            if 'rhy' not in c[ii][i][typ][0][0]:
                continue
            
            # over tiers
            for j in utils.sorted_keys(c[ii][i][typ]):
                # over segments
                for k in utils.sorted_keys(c[ii][i][typ][j]):
                    tt = c[ii][i][typ][j][k]['tier']
                    d['fi'].append(ii)
                    d['ci'].append(i)
                    d['si'].append(k)
                    d['tier'].append(tt)
                    d['stm'].append(c[ii][i]['fsys']['f0']['stm'])
                    d['t_on'].append(c[ii][i][typ][j][k]['to'][0])
                    if len(c[ii][i][typ][j][k]['to']) > 1:
                        d['t_off'].append(c[ii][i][typ][j][k]['to'][1])
                    else:
                        d['t_off'].append(c[ii][i][typ][j][k]['to'][0])
                    d['lab'].append(c[ii][i][typ][j][k]['lab'])
                    d['is_init'].append(c[ii][i][typ][j][k]['is_init'])
                    d['is_fin'].append(c[ii][i][typ][j][k]['is_fin'])
                    d['is_init_chunk'].append(
                        c[ii][i][typ][j][k]['is_init_chunk'])
                    d['is_fin_chunk'].append(
                        c[ii][i][typ][j][k]['is_fin_chunk'])
                    d['dur'].append(c[ii][i][typ][j][k]['rhy']['dur'])
                    # spec moms
                    for q in utils.idx_seg(1, smo, 1):
                        d[f"sm{q}"].append(
                            c[ii][i][typ][j][k]['rhy']['sm'][q-1])
                    for q in ['f_max', 'n_peak']:
                        d[q].append(c[ii][i][typ][j][k]['rhy'][q])
                    # influence of tier_rates (syl, AG etc) in tier (chunk etc)
                    for rt in seen_r:
                        for kk in ['rate', 'mae', 'prop', 'dlm', 'dgm']:
                            zz = f"{rt}_{kk}"
                            if rt in c[ii][i][typ][j][k]['rhy']['wgt']:
                                d[zz].append(c[ii][i][typ][j][k]
                                             ['rhy']['wgt'][rt][kk])
                            # no tier x rateTier combi -> set to nan to get uniform length
                            # for all tierRate_param columns
                            else:
                                d[zz].append(np.nan)
                    # grouping
                    d = export_grp_upd(d, c[ii][i]['grp'])

    exp_to_file(d, fo, typ, fullPath=opt['fsys']['export']
                ['fullpath'], sep=opt['fsys']['export']['sep'])
    exp_to_file(df, fo, typf, fullPath=opt['fsys']['export']
                ['fullpath'], sep=opt['fsys']['export']['sep'])
    return {'seg': d, 'file': df}


def rhy_rate_tiers(c, t, opt):

    '''
    return list of rate tier names for tier t
    
    Args:
      c: copa['data']
      t: tierName
      opt: copa['config']
    
    Returns:
      r: list of rate tier names
    '''

    # channel idx of t
    if t not in opt['fsys']['channel']:
        return []

    ci = opt['fsys']['channel'][t]

    # find entry in c[0][ci][rhy_f0]
    for i in utils.sorted_keys(c[0][ci]['rhy_f0']):
        if c[0][ci]['rhy_f0'][i][0]['tier'] == t:
            return c[0][ci]['rhy_f0'][i][0]['rhy']['wgt'].keys()
    return []



def export_voice(c, fo, opt):

    '''
    fset voice
    '''

    typ = 'voice'
    typf = 'voice_file'

    # segment-level
    cn = ['fi', 'ci', 'si', 'stm', 't_on', 't_off', 'tier', 'lab',
          'is_init', 'is_fin', 'is_init_chunk', 'is_fin_chunk']

    # file-level
    typf = f"{typ}_file"
    cnf = ['fi', 'ci', 'stm']

    for fld in ['jit', 'shim']:
        cn.append(fld)
        cnf.append(fld)
        cn.append(f"{fld}_nrm")
        for subfld in ['m', 'sd']:
            cn.append(f"{fld}_{subfld}")
            cn.append(f"{fld}_{subfld}_nrm")
            cnf.append(f"{fld}_{subfld}")
        for i in [0, 1, 2, 3]:
            cn.append(f"{fld}_c{i}")
            cnf.append(f"{fld}_c{i}")

    # segment-level
    d = init_exp_dict(cn, opt)

    # file-level
    df = init_exp_dict(cnf, opt)

    # files
    for ii in utils.sorted_keys(c):
        # channels
        for i in utils.sorted_keys(c[ii]):
            # file-level
            df['fi'].append(ii)
            df['ci'].append(i)
            df['stm'].append(c[ii][i]['fsys']['f0']['stm'])
            df = export_grp_upd(df, c[ii][i]['grp'])
            # over jit, shim
            for x in c[ii][i][typf].keys():
                df[x].append(c[ii][i][typf][x]['v'])
                df[f"{x}_m"].append(c[ii][i][typf][x]['m'])
                df[f"{x}_sd"].append(c[ii][i][typf][x]['sd'])
                coef = c[ii][i][typf][x]['c']
                coef = coef[::-1]
                # over coefs
                for ci in utils.idx(coef):
                    df[f"{x}_c{ci}"].append(coef[ci])

            # segment level
            # over segments
            for j in utils.sorted_keys(c[ii][i][typ]):
                # over tiers
                for k in utils.sorted_keys(c[ii][i][typ][j]):
                    d['fi'].append(ii)
                    d['ci'].append(i)
                    d['si'].append(j)
                    d['stm'].append(c[ii][i]['fsys']['f0']['stm'])
                    d['t_on'].append(c[ii][i][typ][j][k]['to'][0])
                    if len(c[ii][i][typ][j][k]['to']) > 1:
                        d['t_off'].append(c[ii][i][typ][j][k]['to'][1])
                    else:
                        d['t_off'].append(c[ii][i][typ][j][k]['to'][0])
                    d['tier'].append(c[ii][i][typ][j][k]['tier'])

                    # grouping
                    d = export_grp_upd(d, c[ii][i]['grp'])
                    d['lab'].append(c[ii][i][typ][j][k]['lab'])
                    d['is_init'].append(c[ii][i][typ][j][k]['is_init'])
                    d['is_fin'].append(c[ii][i][typ][j][k]['is_fin'])
                    d['is_init_chunk'].append(
                        c[ii][i][typ][j][k]['is_init_chunk'])
                    d['is_fin_chunk'].append(
                        c[ii][i][typ][j][k]['is_fin_chunk'])

                    # over jit, shim
                    for x in ['jit', 'shim']:
                        d[x].append(c[ii][i][typ][j][k][x]['v'])
                        d[f"{x}_nrm"].append(
                            c[ii][i][typ][j][k][x]['v_nrm'])
                        for zz in ["m", "sd"]:
                            d[f"{x}_{zz}"].append(
                                c[ii][i][typ][j][k][x][zz])
                            d[f"{x}_{zz}_nrm"].append(
                                c[ii][i][typ][j][k][x][f"{zz}_nrm"])
                        coef = c[ii][i][typ][j][k][x]['c']
                        coef = coef[::-1]
                        # over coefs
                        for ci in utils.idx(coef):
                            d[f"{x}_c{ci}"].append(coef[ci])

    exp_to_file(d, fo, typ, fullPath=opt['fsys']['export']
                ['fullpath'], sep=opt['fsys']['export']['sep'])
    exp_to_file(df, fo, typf, fullPath=opt['fsys']['export']
                ['fullpath'], sep=opt['fsys']['export']['sep'])


def export_gnl(c, fo, opt, typ):

    '''
    fset gnl
    typ: 'gnl_f0'|'gnl_en'
    segment- and file-level output
    '''

    typf = f"{typ}_file"

    # segment-/file-level
    cn = ['m', 'sd', 'med', 'iqr', 'max', 'maxpos', 'min', 'dur']
    cnf = cp.deepcopy(cn)

    # en: rms, *_nrm, r_en_f0, and sb (*_nrm and sb for segment level only)
    # f0: *_nrm, bv for file level only
    for x in cp.deepcopy(cn):
        cn.append(f"{x}_nrm")
    if typ == 'gnl_en':
        cn.append('rms')
        cn.append('rms_nrm')
        cn.append('sb')
        cn.append('r_en_f0')
    else:
        cnf.append('bv')

    # quotient variables (not nrm)
    for x in ['qi', 'qf', 'qb', 'qm', 'c0', 'c1', 'c2']:
        cn.append(x)
        cnf.append(x)

    # linking variables
    for x in ['fi', 'ci', 'si', 'stm', 't_on', 't_off', 'tier', 'lab',
              'is_init', 'is_fin', 'is_init_chunk', 'is_fin_chunk']:
        cn.append(x)
    for x in ['fi', 'ci', 'stm']:
        cnf.append(x)

    # segment-level
    d = init_exp_dict(cn, opt)
    # file-level
    df = init_exp_dict(cnf, opt)

    # files
    for ii in utils.sorted_keys(c):
        # channels
        for i in utils.sorted_keys(c[ii]):
            # file-level
            if typf not in c[ii][i].keys():
                continue
            for x in c[ii][i][typf].keys():
                if x in df.keys():
                    df[x].append(c[ii][i][typf][x])
            df['fi'].append(ii)
            df['ci'].append(i)
            df['stm'].append(c[ii][i]['fsys']['f0']['stm'])
            df = export_grp_upd(df, c[ii][i]['grp'])
            if typ == 'gnl_f0':
                df['bv'].append(c[ii][i]['f0']['bv'])
            # segment level
            if typ not in c[ii][i]:
                continue
            # over segments
            for j in utils.sorted_keys(c[ii][i][typ]):
                # over tiers
                for k in utils.sorted_keys(c[ii][i][typ][j]):
                    d['fi'].append(ii)
                    d['ci'].append(i)
                    d['si'].append(j)
                    d['stm'].append(c[ii][i]['fsys']['f0']['stm'])
                    d['t_on'].append(c[ii][i][typ][j][k]['to'][0])
                    if len(c[ii][i][typ][j][k]['to']) > 1:
                        d['t_off'].append(c[ii][i][typ][j][k]['to'][1])
                    else:
                        d['t_off'].append(c[ii][i][typ][j][k]['to'][0])
                    d['tier'].append(c[ii][i][typ][j][k]['tier'])
                    d['lab'].append(c[ii][i][typ][j][k]['lab'])
                    d['is_init'].append(c[ii][i][typ][j][k]['is_init'])
                    d['is_fin'].append(c[ii][i][typ][j][k]['is_fin'])
                    d['is_init_chunk'].append(
                        c[ii][i][typ][j][k]['is_init_chunk'])
                    d['is_fin_chunk'].append(
                        c[ii][i][typ][j][k]['is_fin_chunk'])
                    for x in c[ii][i][typ][j][k]['std'].keys():
                        if x in d.keys():
                            d[x].append(c[ii][i][typ][j][k]['std'][x])
                    # grouping
                    d = export_grp_upd(d, c[ii][i]['grp'])

    exp_to_file(d, fo, typ, fullPath=opt['fsys']['export']
                ['fullpath'], sep=opt['fsys']['export']['sep'])
    exp_to_file(df, fo, typf, fullPath=opt['fsys']['export']
                ['fullpath'], sep=opt['fsys']['export']['sep'])

    return {'seg': d, 'file': df}


def export_merge(fo, infx, dd, opt):

    '''
    merges gnl_f0|en and rhy_f0|en export tables
    
    Args:
      fo: outputFileStem
      infx: infix string 'gnl'|'rhy'
      dd: dict ['en'|'f0']['seg|file'] -> output of export_gnl or _rhy
                featSubset  segment-/file-level
    
    Returns:
      fileOutput of merged dataFrames
    '''

    # unique field names
    ff = {'fi', 'ci', 'si', 'stm', 't_on',
          't_off', 'tier', 'lab', 'dur', 'dur_nrm'}
    pat = r'^grp'
    dk = dd.keys()
    if len(dk) <= 1:
        return
    d = {}
    df = {}
    # 'en','f0'
    for x in dk:
        # features, file-level
        for y in dd[x]['file']:
            if ((y not in ff) and (not re.search(pat, y))):
                n = f"{x}_{y}"
            else:
                n = y
            df[n] = dd[x]['file'][y]
            
        # features, segment-level
        for y in dd[x]['seg']:
            if ((y not in ff) and (not re.search(pat, y))):
                n = f"{x}_{y}"
            else:
                n = y
            d[n] = dd[x]['seg'][y]

    # check for same length (not provided, if not same number of tiers used for gnl_f0|en)
    nl = -1
    for x in list(d.keys()):
        if nl < 0:
            nl = len(d[x])
        if len(d[x]) != nl:
            return

    exp_to_file(df, fo, f"{infx}_file", fullPath=opt['fsys']['export']['fullpath'],
                sep=opt['fsys']['export']['sep'])
    exp_to_file(d, fo, infx, fullPath=opt['fsys']['export']['fullpath'],
                sep=opt['fsys']['export']['sep'])
    return


def export_grp_init(opt, cn):

    '''
    initializes grouping (based on opt['fsys']['grp'])
    updates column name list
    
    Args:
      opt:   copa['config']
      cn:    columnNameList
    
    Returns:
      cn:    with added grouping columns with own namespace 'grp_*'
    '''

    # unique grouping namespace
    ns = 'grp'
    grp = opt['fsys']['grp']
    if len(grp['lab']) > 0:
        for x in grp['lab']:
            if len(x) == 0:
                continue
            cn.append(f"{ns}_{x}")
    return cn


def export_grp_upd(d, grp):

    '''
    updates filename-based grouping columns in output dict
    
    Args:
      d:   output dict
      grp: ['data'][ii][i]['grp']
    
    Returns:
      d:   with updated grouping columns
    '''

    ns = 'grp'
    for x in grp.keys():
        d[f"{ns}_{x}"].append(grp[x])
    return d


def exp_to_file(d, fo, infx, checkFld='fi', facpat='', fullPath=False, sep=','):

    '''
    csv and .R template file output
    
    Args:
      d: dict
      fo: outputFileStem
      infx: infix
      checkFld: <'fi'> key to check whether d is empty
      facpat: <''> pattern additionally to treat as factor
      fullPath: <False> whether or not to write full path of csv file in
                R template
      sep: <','> column separator
    
    Returns:
      df: dataframe (same content as csv output)
    '''

    if ((checkFld in d) and (len(d[checkFld]) > 0)):
        df = exp_dataframe(d)
        df.to_csv(f"{fo}.{infx}.csv", na_rep='NA',
                  index_label=False, index=False, sep=sep)
        exp_R(d, f"{fo}.{infx}", facpat, fullPath, sep)


def exp_dataframe(d):
    ''' converts dict to dataframe with alphanumerically sorted columns '''

    df = pd.DataFrame(d)
    return df.reindex(columns=sorted(df.columns))


def exp_to_file_quoteNonnum(d, fo, infx, checkFld='fi',
                            facpat='', fullPath=False, sep=','):

    if ((checkFld in d) and (len(d[checkFld]) > 0)):
        df = exp_dataframe(d)
        df.to_csv(f"{fo}.{infx}.csv", na_rep='NA', index_label=False,
                  index=False, quoting=csv.QUOTE_NONNUMERIC, sep=sep)
        exp_R(d, f"{fo}.{infx}", facpat, fullPath, sep)


def exp_R(d, fo, facpat='', fullPath=False, sep=','):

    '''
    R table read export
    
    Args:
     d output table
     fo file stem
     facpat pattern additionally to treat as factor
     fullPath <False> should csv file be addressed by full path or file name only
     sep: <','> column separator
    '''

    # fo +/- full path
    if fullPath:
        foo = fo
    else:
        foo = os.path.basename(fo)
        
    # factors (next to grp_*, lab_*)
    fac = utils.lists('factors', 'set')
    o = [f"d<-read.table('{foo}.csv', header=T, fileEncoding='UTF-8', sep='{sep}',",
         "\tcolClasses=c("]

    for x in sorted(d.keys()):
        if ((x in fac) or re.search(r'^(grp|lab|class|spk|tier)', x) or
            re.search(r'_(grp|lab|class|tier)$', x) or
            re.search(r'_(grp|lab|class|tier)_', x) or
            re.search(r'(is_fin|is_init)', x)):
            typ = 'factor'
        elif (len(facpat) > 0 and re.search(r"{}".format(facpat), x)):
            typ = 'factor'
        else:
            typ = 'numeric'
        z = f"{x}='{typ}',"
        if (not re.search(r',$', o[-1])):
            o[-1] += z
        else:
            o.append("\t\t"+z)
    o[-1] = o[-1].replace(',', '))')
    utils.output_wrapper(o, f"{fo}.R", 'list')


def init_exp_dict(c, opt):

    '''
    export dict init from colnames in list C (-> keys)
    add grouping columns
    '''

    # grouping columns from filename
    c = export_grp_init(opt, c)

    d = {}
    for x in c:
        d[x] = []
    return d


def copa_contains(c, dom, ft):

    '''
    checks whether copa dict contains domain-related feature sets
    
    Args:
     c: copa['data']
     dom: domain 'glob'|'loc'|'bnd' etc
     ft: dict domain -> characteristic feature set ('std' etc)
    
    Returns:
      bool
    '''

    if ((0 in c.keys()) and (0 in c[0].keys()) and
        (dom in c[0][0].keys()) and (0 in c[0][0][dom]) and
        ((ft[dom] in c[0][0][dom][0]) or
         ((0 in c[0][0][dom][0]) and (ft[dom] in c[0][0][dom][0][0])))):
        return True
    return False


def selAbs(v, n, fset, sel=('glob', 'loc', 'bnd')):

    '''
    returns abs value if feature describes assymmetric aspects
    
    Args:
     v: featval
     n: featname
     fset: feature set
     sel_fset: which fsets to consider
    
    Returns:
     v or abs(v)

    assym features:
      loc: c1, c3, bl|ml|tl|rng_c1|r|rate|d_init|d_fin|sd|d
      glob: bl|ml|tl|rng_c1|rate
    '''

    if fset not in sel:
        return v
    
    # loc
    if re.search(r'^c[13579]$', n):
        return abs(v)

    # gnl
    if re.search(r'(f0|en)_c[13]', n):
        return abs(v)

    # glob/loc
    if re.search(r'^([bmt]l|rng)_(r|c1|rate|d_(init|fin)|s?d)$', n):
        return abs(v)

    # bnd
    if re.search(r'_([bmt]l|rng)_r$', n):
        return abs(v)
    return v

