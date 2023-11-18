import numpy as np
import scipy as si
import sklearn.cluster as sc
import sklearn.preprocessing as sp
import sklearn.metrics as sm

import copasul.copasul_utils as utils


def clst_main(copa, dom, f_log_in=None):

    '''
    local and global contour clustering

    Args:
     copa: (dict)
     dom: (str) 'glob'|'loc'
     f_log_in: (filehandle) of logfile    

    Returns:
     copa (dict)
       .data
          .<dom>
              .ci      clusterIdx
       .clst
          .<dom>
              .cntr     matrix of centroids (one per row)
    '''

    global f_log
    f_log = f_log_in

    utils.check_var({'var': copa, 'env': 'copasul',
                     'spec': {'step': 'clst', 'dom': dom}})

    opt = copa['config']['clst'][dom]
    x = copa['clst'][dom]['c']

    # preprocessing ################################################
    # robust centering+scaling (not as vulnerable to outliers)
    cs = sp.RobustScaler().fit(x)
    xn = cs.transform(x)

    mtd = opt['mtd']
    copt = opt[mtd]
    bopt = opt['estimate_bandwidth']

    # clustering ####################################################
    if ((mtd == 'meanShift') or (copt['init'] == 'meanShift')):
        if opt['meanShift']['bandwidth'] is None or opt['meanShift']['bandwidth'] == 0:
            try:
                bw = sc.estimate_bandwidth(
                    xn, quantile=bopt['quantile'],
                    n_samples=min([bopt['n_samples'], len(xn)]),
                    random_state=opt["seed"])
            except:
                bw = 0.5
        else:
            bw = opt['meanShift']['bandwidth']
        if bw == 0:
            bw = 0.5
        ms = sc.MeanShift(bandwidth=bw,
                          bin_seeding=opt['meanShift']['bin_seeding'],
                          min_bin_freq=opt['meanShift']['min_bin_freq'])
        ms.fit(xn)
        obj = ms

    if mtd == 'kMeans':
        if copt['init'] == 'meanShift':
            kmi = ms.cluster_centers_
            k = len(kmi)
            km = sc.KMeans(
                init=kmi, max_iter=copt['max_iter'], n_clusters=k,
                n_init=1, random_state=opt["seed"])
        else:
            kmi = copt['init']
            k = copt['n_cluster']
            km = sc.KMeans(
                n_clusters=k, max_iter=copt['max_iter'], n_init=copt['n_init'],
                random_state=opt["seed"])

        km.fit(xn)
        obj = km
        
    cc = obj.cluster_centers_
    ci = obj.labels_

    # validation ############################################
    try:
        copa['clst'][dom]['val'] = sm.silhouette_score(
            xn, ci, metric='euclidean')
    except:
        copa['clst'][dom]['val'] = np.nan
    copa['val']['clst'][dom]['sil_mean'] = np.mean(copa['clst'][dom]['val'])

    # updating copa ##########################################

    # cluster object
    copa['clst'][dom]['obj'] = obj

    # denormalized centroids
    copa['clst'][dom]['cntr'] = cs.inverse_transform(cc)

    for n in utils.idx_a(len(ci)):
        ii, i, j = copa['clst'][dom]['ij'][n, :]
        copa['data'][ii][i][dom][j]['class'] = ci[n]

    return copa


def featweights(x, c, do_nrm=True):

    '''
    derived feature weights normalized to 1
    based on related mean silhouette value
    
    Args:
      x: (np.array) m x n feature matrix
      c: (np.array) 1 x m cluster index vector (numeric!)
      do_nrm: (boolean) it True, do normalize features
    
    Returns:
      w: (np.array) 1 x n weight vector
    '''

    w = np.ones(x.shape[1])
    c = np.asarray(c).astype(int)
    
    # only one class -> equal wgt
    if len(np.unique(c)) == 1:
        return w

    # over columns
    for i in utils.idx_a(len(w)):
        s = sm.silhouette_score(x[:, i].reshape(-1, 1), c, metric='euclidean')
        w[i] = np.mean(s) + 1

    # normalize
    if do_nrm:
        w = w / np.sum(w)
    else:
        w -= 1
        
    return w


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
    

