#!/usr/bin/env python3

# author: Uwe Reichel, Budapest, 2016

import mylib as myl
import numpy as np
import scipy as si
import sklearn.cluster as sc
import sklearn.preprocessing as sp
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import math

# cluster constraints
# ns: max n_samples for MeanShift
max_ns = 1000


# IN:
#  dict copa
#  dom := 'glob'|'loc'
# OUT:
#  copa
#    .data
#       .<dom>
#           .ci  # clusterIdx
#    .clst
#       .<dom>
#           .cntr # matrix of centroids (one per row)
def clst_main(copa,dom,f_log_in):
    global f_log
    f_log = f_log_in

    myl.check_var({'var':copa,'env':'copasul',
                   'spec':{'step':'clst','dom':dom}})

    opt = copa['config']['clst'][dom]
    x = copa['clst'][dom]['c']

    ## preprocessing ################################################
    # robust centering+scaling (not as vulnerable to outliers)
    # cs
    #   -- attributes
    #   .center_  [colMedians]
    #   .scale_   [colIQR]
    #   -- methods
    #   .transform(x) -> operation
    #   .inverse_transform(xn)
    cs = sp.RobustScaler().fit(x)
    xn = cs.transform(x)

    mtd = opt['mtd']
    copt = opt[mtd]
    bopt = opt['estimate_bandwidth']

    ## clustering ####################################################
    if ((mtd=='meanShift') or (copt['init']=='meanShift')):
        if opt['meanShift']['bandwidth']==0:
            try:
                bw = sc.estimate_bandwidth(xn, quantile = bopt['quantile'],
                                           n_samples = min([bopt['n_samples'],len(xn)]))
            except:
                bw = 0.5
        else:
            bw = opt['meanShift']['bandwidth']
        if bw==0:
            bw=0.5
        ms = sc.MeanShift(bandwidth=bw,
                          bin_seeding=opt['meanShift']['bin_seeding'],
                          min_bin_freq=opt['meanShift']['min_bin_freq'])
        ms.fit(xn)
        obj = ms
        
    if mtd=='kMeans':
        if copt['init'] == 'meanShift':
            kmi = ms.cluster_centers_
            k = len(kmi)
            km = sc.KMeans(init=kmi,max_iter=copt['max_iter'],n_clusters=k,n_init=1)
        else:
            kmi = copt['init']
            k = copt['n_cluster']
            km = sc.KMeans(n_clusters=k,max_iter=copt['max_iter'],n_init=copt['n_init'])
 
        km.fit(xn)
        obj = km

    cc = obj.cluster_centers_
    ci = obj.labels_
    
    ## validation ############################################
    try:
        copa['clst'][dom]['val'] = sm.silhouette_score(xn,ci,metric='euclidean')
    except:
        copa['clst'][dom]['val'] = np.nan
    copa['val']['clst'][dom]['sil_mean'] = np.mean(copa['clst'][dom]['val'])
    
    ## updating copa ##########################################
    # cluster object
    copa['clst'][dom]['obj'] = obj
    # denormalized centroids
    copa['clst'][dom]['cntr'] = cs.inverse_transform(cc)

    for n in myl.idx_a(len(ci)):
        ii, i, j = copa['clst'][dom]['ij'][n,:]
        copa['data'][ii][i][dom][j]['class'] = ci[n]
    
    return copa

# derived feature weights normalized to 1
# based on related mean silhouette value
# IN:
#   x: n x m feature matrix
#   c: n x 1 cluster index vector (must be numeric!)
# OUT:
#   w: 1 x m weight vector
def featweights(x,c):
    w = np.ones(myl.ncol(x))
    c = np.asarray(c).astype(int)
    # only one class -> equal wgt
    if len(np.unique(c))==1:
        return w
    # over columns
    for i in myl.idx_a(len(w)):
        s = sm.silhouette_score(x[:,i].reshape(-1,1),c,metric='euclidean')
        w[i] = np.mean(s)+1
    # normalize
    w = w/np.sum(w)
    return w

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
