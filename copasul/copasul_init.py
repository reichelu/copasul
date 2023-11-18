import argparse
import numpy as np
import os
import re
import sys

import copasul.copasul_utils as utils


def copa_init(opt):

    '''
    initialize copa dict
    '''

    copa = {'config': opt, 'data': {}, 'clst': {}, 'val': {}}
    copa['clst'] = {'loc': {}, 'glob': {}}
    copa['clst']['loc'] = {'c': [], 'ij': []}
    copa['clst']['glob'] = {'c': [], 'ij': []}
    copa['val']['styl'] = {'glob': {}, 'loc': {}}
    copa['val']['clst'] = {'glob': {}, 'loc': {}}

    return copa


def copa_opt_init(conf):

    '''
    initialize config dict
    
    Args:
      conf: config dict or string myConfigFile.json
    
    Returns:
      opt: initialized config (defaults added etc.)
    '''

    # input
    if type(conf) is dict:
        # dictionary input
        opt = conf
        ref_path = None
    else:
        # json file input
        opt = utils.input_wrapper(conf, 'json')
        ref_path = os.path.dirname(conf)
        
    # merged with defaults
    opt = utils.opt_default(opt, copasul_default())

    # adjust variable types
    # filter frequencies as list
    for x in ['chunk', 'syl']:
        opt['augment'][x]['flt']['f'] = np.array(
            opt['augment'][x]['flt']['f'])
        
    # channels -> array indices, i.e. -1
    # store highest idx+1 (for range()) in  opt['fsys']['nc']
    opt['fsys']['nc'] = 1
    for x in opt['fsys']['channel']:
        if opt['fsys']['channel'][x] > opt['fsys']['nc']:
            opt['fsys']['nc'] = opt['fsys']['channel'][x]
        opt['fsys']['channel'][x] -= 1

    # add defaults and channel idx for AUGMENT tier_out_stm . '_myChannelIdx'
    if 'augment' in opt['fsys']:
        # over units (order important)
        for x in ['chunk', 'syl', 'glob', 'loc']:
            if x not in opt['fsys']['augment']:
                opt['fsys']['augment'][x] = {}
            if (('tier_out_stm' not in opt['fsys']['augment'][x]) or
                    len(opt['fsys']['augment'][x]['tier_out_stm']) == 0):
                opt['fsys']['augment'][x]['tier_out_stm'] = f"copa_{x}"
            # over channelIdx
            for i in range(opt['fsys']['nc']):
                z1 =  opt['fsys']['augment'][x]['tier_out_stm']
                z2 = int(i+1)
                opt['fsys']['channel'][f"{z1}_{z2}"] = i
            # special add of _bnd for syllable boundaries
            if x == 'syl':
                for i in range(opt['fsys']['nc']):
                    z1 = opt['fsys']['augment'][x]['tier_out_stm']
                    z2 = int(i+1)
                    opt['fsys']['channel'][f"{z1}_bnd_{z2}"] = i
            # add further defaults
            for y in ['glob', 'loc', 'syl']:
                if x != y:
                    continue
                if re.search('(glob|syl)', y):
                    parent = 'chunk'
                else:
                    parent = 'glob'
                if 'tier_parent' not in opt['fsys']['augment'][x]:
                    opt['fsys']['augment'][x]['tier_parent'] = opt['fsys']['augment'][parent]['tier_out_stm']
                if y == 'syl':
                    continue
                if 'tier' not in opt['fsys']['augment'][x]:
                    opt['fsys']['augment'][x]['tier'] = opt['fsys']['augment']['syl']['tier_out_stm']

    # normalize path names across platforms
    for x in opt["fsys"].keys():
        if type(opt["fsys"][x]) is dict and "dir" in opt["fsys"][x]:
            if ref_path is None or os.path.isabs(opt["fsys"][x]["dir"]):
                opt["fsys"][x]["dir"] = os.path.abspath(opt["fsys"][x]["dir"])
            else:
                # expand paths relative to config file location
                #print(ref_path)
                #print(opt["fsys"][x]["dir"])
                jn = os.path.join(ref_path, opt["fsys"][x]["dir"])
                opt["fsys"][x]["dir"] = os.path.abspath(jn)

    # add empty interp subdict to opt["preproc"]
    if 'interp' not in opt['preproc']:
        opt['preproc']['interp'] = {}

    # unify bnd, gnl_* and rhy_* tiers as lists
    for x in ['bnd', 'gnl_en', 'gnl_f0', 'rhy_en', 'rhy_f0', 'voice']:
        for y in ['tier', 'tier_rate']:
            if x in opt['fsys'] and y in opt['fsys'][x]:
                if type(opt['fsys'][x][y]) is not list:
                    opt['fsys'][x][y] = [opt['fsys'][x][y]]

    # dependencies
    if re.search('^seed', opt['augment']['glob']['cntr_mtd']):
        if not opt['augment']['glob']['use_gaps']:
            print(f"Init: IP augmentation by means of {opt['augment']['glob']['cntr_mtd']} centroid " \
                  "method requires value true for use_gaps. Changed to true.")
            opt['augment']['glob']['use_gaps'] = True

    # force at least 1 ncl and accent to be in file. Otherwise creation summary statistics table fails
    # due to unequal number of rows
    if utils.ext_true(opt['fsys']['export'], 'summary'):
        opt['augment']['loc']['force'] = True
    if utils.ext_true(opt['augment']['loc'], 'force'):
        opt['augment']['syl']['force'] = True

    # distribute label and channel specs to subdicts
    # take over dir|typ|ext from 'annot'
    # needed e.g. to extract tiers by pp_tiernames()
    fs = utils.lists('featsets')
    fs.append('augment')
    fs.append('chunk')
    for x in fs:
        if x not in opt['fsys']:
            continue
        opt['fsys'][x]['lab_pau'] = opt['fsys']['label']['pau']
        opt['fsys'][x]['channel'] = opt['fsys']['channel']
        opt['fsys'][x]['nc'] = opt['fsys']['nc']
        for y in ['dir', 'typ', 'ext']:
            opt['fsys'][x][y] = opt['fsys']['annot'][y]

    # add labels to 'annot' and 'augment' subdicts
    for x in ['syl', 'chunk', 'pau']:
        opt['fsys']['augment'][f'lab_{x}'] = opt['fsys']['label'][x]
        opt['fsys']['annot'][f'lab_{x}'] = opt['fsys']['label'][x]

    # distribute sample rate, and set to 100
    opt['fs'] = 100
    for x in utils.lists('featsets'):
        if (x not in opt['styl']):
            opt['styl'][x] = {}
        opt['styl'][x]['fs'] = opt['fs']
        if ((x in opt['preproc']) and ('point_win' in opt['preproc'][x])):
            opt['styl'][x]['point_win'] = opt['preproc'][x]['point_win']
        else:
            opt['styl'][x]['point_win'] = opt['preproc']['point_win']

    # add missing navigations
    for x in ["from_scratch", "overwrite_config", "do_diagnosis",
              "do_augment_chunk", "do_augment_syl", "do_augment_glob",
              "do_augment_loc", "do_preproc", "do_styl_glob",
              "do_styl_loc", "do_styl_loc_ext",
              "do_styl_gnl_en", "do_styl_bnd", "do_styl_bnd_win",
              "do_styl_bnd_trend", "do_styl_rhy_f0", "do_styl_rhy_en",
              "do_clst_loc", "do_clst_glob", "do_styl_voice", "do_export",
              "do_plot"]:
        if x not in opt['navigate']:
            opt['navigate'][x] = False

    # set navigate macros
    opt['navigate']['do_augment'] = False
    for x in ['chunk', 'syl', 'glob', 'loc']:
        if opt['navigate'][f"do_augment_{x}"]:
            opt['navigate']['do_augment'] = True
            break

    # full path export (default False; transform [0,1] to boolean)
    if 'fullpath' not in opt['fsys']['export']:
        opt['fsys']['export']['fullpath'] = False
    else:
        opt['fsys']['export']['fullpath'] = bool(
            opt['fsys']['export']['fullpath'])

    # set fromScratch to True if no copa.pickle file available
    z = os.path.join(opt['fsys']['export']['dir'],
                     opt['fsys']['export']['stm'])
    f = f"{z}.pickle"
    if not os.path.isfile(f):
        opt['navigate']['from_scratch'] = True

    # default empty grouping
    if ('grp' not in opt['fsys']):
        opt['fsys']['grp'] = {'lab': [], 'src': '', 'sep': ''}

    # plot options
    if 'color' not in opt['plot']:
        opt['plot']['color'] = True
    for x in ['browse', 'grp']:
        if 'color' not in opt['plot'][x]:
            opt['plot'][x]['color'] = opt['plot']['color']

    # check [preproc][base_prct_grp] <-> [fsys][grp][lab] compliance
    if 'base_prct_grp' in opt['preproc']:
        # empty or not needed subdict
        if ((len(list(opt['preproc']['base_prct_grp'].keys())) == 0) or
                opt['preproc']['base_prct'] == 0):
            del opt['preproc']['base_prct_grp']
        else:
            # over channel idx
            for ci in opt['preproc']['base_prct_grp']:
                g = opt['preproc']['base_prct_grp'][ci]
                # label cannot be inferred from file name
                if g not in opt['fsys']['grp']['lab']:
                    print('WARNING: labels in opt.preproc.base_prct_grp need to be ' \
                          'extractable from filename by opt.fsys.grp.lab. Removed. ' \
                          'Base percentile will be calculated separately for each file.')
                    del opt['preproc']['base_prct_grp']
                    break

    # transform string key to channel idx in copa['data']['fileIdx']
    if 'base_prct_grp' in opt['preproc']:
        bpg = {}
        lcis = utils.sorted_keys(opt['preproc']['base_prct_grp'])
        for cis in lcis:
            ci = int(cis)-1
            opt['preproc']['base_prct_grp'][ci] = opt['preproc']['base_prct_grp'][cis]
            del opt['preproc']['base_prct_grp'][cis]

    return opt


def copasul_default():
    return {
        "fs": 100,
        "navigate": {
            "do_preproc": True,
            "do_styl_glob": True,
            "from_scratch": True,
            "overwrite_config": False,
            "do_diagnosis": False,
            "do_augment_chunk": False,
            "do_augment_syl": False,
            "do_augment_glob": False,
            "do_augment_loc": False,
            "do_styl_loc": True,
            "do_styl_gnl_f0": False,
            "do_styl_gnl_en": False,
            "do_styl_loc_ext": False,
            "do_styl_bnd": False,
            "do_styl_bnd_win": False,
            "do_styl_bnd_trend": False,
            "do_styl_voice": False,
            "do_clst_loc": True,
            "do_clst_glob": True,
            "do_styl_rhy_f0": True,
            "do_styl_rhy_en": True,
            "do_export": True,
            "do_plot": False,
            "do_precheck": False,
            "do_styl_glob_ext": False,
            "do_rhy_f0": False,
            "do_rhy_en": 0
        },
        "preproc": {
            "st": True,
            "base_prct": 5,
            "smooth": {
                "mtd": "sgolay",
                "win": 7,
                "ord": 3
            },
            "out": {
                "f": 3,
                "m": "mean"
            },
            "nrm_win": 0.6,
            "point_win": 0.3,
            "loc_sync": False,
            "loc_align": "skip"
        },
        "styl": {
            "glob": {
                "nrm": {
                    "mtd": "minmax",
                    "rng": [0, 1]
                },
                "decl_win": 0.1,
                "align": "center",
                "prct": {
                    "bl": 10,
                    "tl": 90
                }
            },
            "loc": {
                "nrm": {
                    "mtd": "minmax",
                    "rng": [-1, 1]
                },
                "ord": 3
            },
            "register": "ml",
            "bnd": {
                "nrm": {
                    "mtd": "minmax",
                    "rng": [0, 1]
                },
                "decl_win": 0.1,
                "prct": {
                    "bl": 10,
                    "tl": 90
                },
                "win": 1,
                "cross_chunk": True,
                "residual": 0
            },
            "gnl": {
                "win": 0.3
            },
            "gnl_en": {
                "sb": {
                    "domain": "time",
                    "win": -1,
                    "f": -1,
                    "bytpe": "none",
                    "alpha": 0.95
                },
                "sts": 0.01,
                "win": 0.05,
                "wintyp": "hamming",
                "winparam": None,
                "centering": 1
            },
            "rhy_f0": {
                "typ": "f0",
                "sig": {
                },
                "rhy": {
                    "wintyp": "kaiser",
                    "winparam": 1,
                    "lb": 0,
                    "ub": 10,
                    "nsm": 3,
                    "rmo": False,
                    "peak_prct": 80,
                    "wgt": {
                        "rb": 1
                    }
                }
            },
            "rhy_en": {
                "typ": "en",
                "sig": {
                    "sts": 0.01,
                    "win": 0.05,
                    "wintyp": "hamming",
                    "winparam": None,
                    "scale": True
                },
                "rhy": {
                    "wintyp": "kaiser",
                    "winparam": 1,
                    "lb": 0,
                    "ub": 10,
                    "nsm": 3,
                    "rmo": False,
                    "peak_prct": 80,
                    "wgt": {
                        "rb": 1
                    }
                }
            },
            "voice": {
                "jit": {
                    "t_max": 0.02,
                    "t_min": 0.0001,
                    "fac_max": 1.3
                },
                "shim": {}
            }
        },
        "clst": {
            "glob": {
                "mtd": "meanShift",
                "kMeans": {
                    "init": "meanShift",
                    "n_cluster": 3,
                    "max_iter": 300,
                    "n_init": 10
                },
                "meanShift": {
                    "bin_seeding": False,
                    "min_bin_freq": 1,
                    "bandwidth": None
                },
                "estimate_bandwidth": {
                    "quantile": 0.3,
                    "n_samples": 1000
                },
                "seed": 42
            },
            "loc": {
                "mtd": "meanShift",
                "kMeans": {
                    "init": "meanShift",
                    "n_cluster": 5,
                    "max_iter": 300,
                    "n_init": 10
                },
                "meanShift": {
                    "bin_seeding": False,
                    "min_bin_freq": 1,
                    "bandwidth": None
                },
                "estimate_bandwidth": {
                    "quantile": 0.3,
                    "n_samples": 1000
                },
                "seed": 42
            }
        },
        "augment": {
            "chunk": {
                "e_rel": 0.1,
                "l": 0.1524,
                "l_ref": 5,
                "n": -1,
                "fbnd": False,
                "min_pau_l": 0.5,
                "min_chunk_l": 0.2,
                "margin": 0.0,
                "flt": {
                    "btype": "low",
                    "f": 8000,
                    "ord": 5
                }
            },
            "syl": {
                "e_rel": 1.07,
                "e_val": 0.91,
                "l": 0.08,
                "l_ref": 0.15,
                "d_min": 0.05,
                "e_min": 0.16,
                "center": 0,
                "flt": {
                    "f": [200, 4000],
                    "btype": "band",
                    "ord": 5
                },
                "force": 0
            },
            "glob": {
                "measure": "abs",
                "prct": 95,
                "wgt": {
                },
                "wgt_mtd": "silhouette",
                "cntr_mtd": "seed_prct",
                "min_l": 0.5,
                "use_gaps": True,
                "unit": "file",
                "seed": 42
            },
            "loc": {
                "measure": "abs",
                "prct": 80,
                "wgt": {
                },
                "cntr_mtd": "split",
                "wgt_mtd": "corr",
                "min_l": 0.3,
                "min_l_a": 0.6,
                "max_l_na": 0.1,
                "acc_select": "max",
                "ag_select": "max",
                "unit": "file",
                "c_no_abs": False,
                "force": 0,
                "seed": 42
            }
        },
        "plot": {
            "browse": {
                "time": "final",
                "type": {
                    "loc": {
                        "decl": False,
                        "acc": False
                    },
                    "glob": {
                        "decl": False
                    },
                    "rhy_f0": {
                        "rhy": False
                    },
                    "rhy_en": {
                        "rhy": False
                    },
                    "complex": {
                        "superpos": False,
                        "gestalt": False,
                        "bnd": False,
                        "bnd_win": False,
                        "bnd_trend": False
                    },
                    "clst": {
                        "contours": False
                    }
                },
                "single_plot": {
                    "active": False,
                    "file_i": 0,
                    "channel_i": 0,
                    "segment_i": 0
                },
                "verbose": False,
                "save": False
            },
            "grp": {
                "type": {
                    "loc": {
                        "decl": False,
                        "acc": False
                    },
                    "glob": {
                        "decl": False
                    }
                },
                "grouping": [
                    "lab"
                ],
                "save": False
            }
        },
        "fsys": {
            "grp": {
                "src": "",
                "sep": "",
                "lab": []
            },
            "f0": {
                "dir": "",
                "ext": "",
                "typ": "tab"
            },
            "aud": {
                "dir": "",
                "typ": "wav",
                "ext": ""
            },
            "annot": {
                "dir": "",
                "typ": "",
                "ext": ""
            },
            "pulse": {
                "dir": "",
                "typ": "",
                "ext": ""
            },
            "label": {
                "pau": "<P>",
                "chunk": "c",
                "syl": "s"
            },
            "augment": {
                "chunk": {
                    "tier_out_stm": "copa_chunk"
                },
                "syl": {
                    "tier_out_stm": "copa_syl",
                    "tier_parent": "copa_chunk"
                },
                "glob": {
                    "tier_parent": "copa_chunk",
                    "tier": "copa_syl_bnd",
                    "tier_out_stm": "copa_glob"
                },
                "loc": {
                    "tier_out_stm": "copa_acc",
                    "tier_ag": "",
                    "tier_acc": "copa_syl",
                    "tier_parent": "copa_glob"
                }
            },
            "chunk": {
                "tier": [
                    "copa_chunk"
                ]
            },
            "pho": {
                "vow": "[AEIOUYaeiouy29}]"
            },
            "glob": {
                "tier": ""
            },
            "loc": {
                "tier_ag": "",
                "tier_acc": ""
            },
            "bnd": {
                "tier": []
            },
            "gnl_f0": {
                "tier": []
            },
            "gnl_en": {
                "tier": []
            },
            "rhy_f0": {
                "tier": [],
                "tier_rate": []
            },
            "rhy_en": {
                "tier": [],
                "tier_rate": []
            },
            "voice": {
                "tier": []
            },
            "channel": {
            },
            "export": {
                "dir": "",
                "stm": "copasul",
                "csv": True,
                "sep": ",",
                "summary": False,
                "f0_preproc": False,
                "f0_residual": False,
                "f0_resyn": False
            },
            "pic": {
                "dir": "",
                "stm": "copasul"
            }
        }
    }

