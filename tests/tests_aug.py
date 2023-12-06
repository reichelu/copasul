import argparse
import audeer
import copy
import glob
import json
import numpy as np
import os
import pandas as pd
import re
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
pth_copa = os.path.normpath(f"{cwd}/..")
sys.path.append(pth_copa)
from copasul import copasul

def tests_aug(task):

    ''' copasul augmenting test collection

    Args:
    task: (str) "ref" - add reference tiers CHUNK_REF_1, SYL_REF_1, LOC_REF_1,
                        GLOB_REF_1 (user will be asked before overwriting
                        existing tiers)
                "apply" - add application tiers GLOB_1, SYL_1, LOC_1, GLOB_1,
                        and compare them with reference

    Returns:
    diagnosis:
       task "ref": None
       task "apply": (dict)
         None if all fine, else:
         keys:
             {CHUNK, SYL, LOC, GLOB}.{ref, apply} = np.arrays of time stamps

    '''
    
    assert task in ["ref", "apply", "stereo"], \
        "task must be 'ref', 'apply'"

    # task "stereo" - run augmentation on more and stereo data
    # works only locally on developer's machine
    if (task == "stereo" and \
        (not re.search("reichel", os.path.expanduser("~")))):
        sys.exit(f"chosen {task} task is not public. Exit.")
    
    # config, featext
    if task != "stereo":
        f_config = os.path.normpath(f"{cwd}/minex/config_aug/minex.json")
    else:
        f_config = os.path.normpath(f"{cwd}/minex/config_aug/hgc_augment.json")

    with open(f_config, 'r') as h:
        opt = json.load(h)

    # expand relative paths to absolute paths relative to config location
    ref_path = os.path.dirname(f_config)
    for x in opt["fsys"].keys():
        if type(opt["fsys"][x]) is dict and "dir" in opt["fsys"][x]:
            jn = os.path.join(ref_path, opt["fsys"][x]["dir"])
            opt["fsys"][x]["dir"] = os.path.abspath(jn)

    fex = copasul.Copasul()

    # simply run and watch wether it fails on large stereo file
    if task == "stereo":
        _ = fex.process(config=opt)
        sys.exit("all ok")

    pat = os.path.normpath(f"{opt['fsys']['annot']['dir']}/*TextGrid")
    ff_tg = sorted(glob.glob(pat))
    tg = i_tg(ff_tg[0])
    tiernames = tg_tn(tg)
    
    if task == "ref":

        if "SYL_REF_1" in tiernames:
            ans = input("Reference tiers already generated! Overwrite? (yes, no): ")
            if ans != "yes":
                sys.exit("Aborted. If wanted, manually remove from " \
                         f"{ff_tg[0]}" \
                         "the 4 tiers *_REF_1 and reduze the size header info " \
                         "accordingly.")

        repl = {
            "chunk": "CHUNK_REF",
            "syl": "SYL_REF",
            "loc": "LOC_REF",
            "glob": "GLOB_REF"
        }
        for dom in repl:
            opt["fsys"]["augment"][dom]["tier_out_stm"] = repl[dom]
        opt["fsys"]["augment"]["syl"]["tier_parent"] = repl["chunk"]
        opt["fsys"]["augment"]["glob"]["tier_parent"] = repl["chunk"]
        opt["fsys"]["augment"]["loc"]["tier_acc"] = repl["syl"]
        opt["fsys"]["augment"]["loc"]["tier_ag"] = repl["glob"]
        opt["fsys"]["augment"]["loc"]["tier_parent"] = repl["glob"]

        for tier in ["SYL", "LOC", "GLOB"]:
            if tier in opt["fsys"]["channel"]:
                del opt["fsys"]["channel"][tier]
            opt["fsys"]["channel"][f"{tier}_REF"] = 1

        copa = fex.process(config=opt)
        print(f"output written to {ff_tg[0]}")
        diagnosis = None

    elif task == "apply":

        # remove CHUNK, GLOB, LOC, SYL _1 before processing        
        tgo = {}
        for tn in ["WRD", "CHUNK_REF_1", "SYL_REF_1", "LOC_REF_1", "GLOB_REF_1"]:
            if tn not in tiernames:
                sys.exit("reference tiers missing. Run tests_aug.py -t ref first")
            tgo = tg_add(tgo, tg_tier(tg, tn))

        # add CHUNK, GLOB, LOC, SYL _1
        o_tg(tgo, ff_tg[0])
        copa = fex.process(config=opt)
        # print(f"output written to {ff_tg[0]}")

        tg = i_tg(ff_tg[0])

        diagnosis = {}
        for tn in  ["CHUNK", "SYL", "LOC", "GLOB"]:
            tier_ref = tg_tier(tg, f"{tn}_REF_1")
            tier_apl = tg_tier(tg, f"{tn}_1")
            t_ref, _ = tg_tier2tab(tier_ref)
            t_apl, _ = tg_tier2tab(tier_apl)
            if not np.allclose(t_ref, t_apl):
                print(f"{tn} mismatch!")
                diagnosis[tn] = {"ref": t_ref,
                                 "apply": t_apl}
                input(diagnosis[tn])
            else:
                print(f"{tn} ok!")
                
        if len(diagnosis) == 0:
            diagnosis = None
            
    return diagnosis


# TextGrid Processing ##############################

def i_tg(ff):

    '''
    TextGrid from file
    
    Args:
      s fileName
    
    Returns:
      dict tg
         type: TextGrid
         format: short|long
         name: s name of file
         head: hoh
             xmin|xmax|size|type
         item_name ->
                 myTiername -> myItemIdx
         item
            myItemIdx ->        (same as for item_name->myTiername)
                 class
                 name
                 size
                 xmin
                 xmax
                 intervals   if class=IntervalTier
                       myIdx -> (xmin|xmax|text) -> s
                 points
                       myIdx -> (time|mark) -> s
    '''
    
    tg = {'name': ff, 'format': 'long', 'head': {},
          'item_name': {}, 'item': {}}
    (key, skip) = ('head', True)
    idx = {'item': 0, 'points': 0, 'intervals': 0}
    with open(ff, encoding='utf-8') as f:
        for z in f:
            z = re.sub(r'\s*\n$', '', z)
            if re.search(r'object\s*class', z, re.I):
                skip = False
                continue
            else:
                if ((skip == True) or re.search(r'^\s*$', z) or
                        re.search('<exists>', z)):
                    continue
            if re.search(r'item\s*\[\s*\]:?', z, re.I):
                key = 'item'
            elif re.search(r'(item|points|intervals)\s*\[(\d+)\]\s*:?', z, re.I):
                m = re.search(
                    r'(?P<typ>(item|points|intervals))\s*\[(?P<idx>\d+)\]\s*:?', z)
                i_type = m.group('typ').lower()
                idx[i_type] = int(m.group('idx'))
                if i_type == 'item':
                    idx['points'] = 0
                    idx['intervals'] = 0
            elif re.search(r'([^\s+]+)\s*=\s*\"?(.*)', z):
                m = re.search(r'(?P<fld>[^\s+]+)\s*=\s*\"?(?P<val>.*)', z)
                (fld, val) = (m.group('fld').lower(), m.group('val'))
                fld = re.sub('number', 'time', fld)
                val = re.sub(r'[\"\s]+$', '', val)
                # type cast
                if fld == 'size':
                    val = int(val)
                elif fld in ['xmin', 'xmax', 'time']:
                    val = float(val)
                # head specs
                if key == 'head':
                    tg[key][fld] = val
                else:
                    # link itemName to itemIdx
                    if fld == 'name':
                        tg['item_name'][val] = idx['item']
                    # item specs
                    if ((idx['intervals'] == 0) and (idx['points'] == 0)):
                        tg[key] = add_subdict(tg[key], idx['item'])
                        tg[key][idx['item']][fld] = val
                    # points/intervals specs
                    else:
                        tg[key] = add_subdict(tg[key], idx['item'])
                        tg[key][idx['item']] = add_subdict(
                            tg[key][idx['item']], i_type)
                        tg[key][idx['item']][i_type] = add_subdict(
                            tg[key][idx['item']][i_type], idx[i_type])
                        tg[key][idx['item']][i_type][idx[i_type]][fld] = val
    return tg


def add_subdict(d, s):

    '''
    add key to empty subdict if not yet part of dict
    
    Args:
      d dict
      s key
    
    Returns:
      d incl key spointing to empty subdict
    '''

    if not (s in d):
        d[s] = {}
    return d


def o_tg(tg, fil):
    h = open(fil, mode='w', encoding='utf-8')
    idt = '    '
    fld = tg_fields()
    # head
    if tg['format'] == 'long':
        h.write("File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n")
        h.write("xmin = {}\n".format(tgv(tg['head']['xmin'], 'xmin')))
        h.write("xmax = {}\n".format(tgv(tg['head']['xmax'], 'xmax')))
        h.write("tiers? <exists>\n")
        h.write("size = {}\n".format(tgv(tg['head']['size'], 'size')))
    else:
        h.write("File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n")
        h.write("{}\n".format(tgv(tg['head']['xmin'], 'xmin')))
        h.write("{}\n".format(tgv(tg['head']['xmax'], 'xmax')))
        h.write("<exists>\n")
        h.write("{}\n".format(tgv(tg['head']['size'], 'size')))

    # item
    if (tg['format'] == 'long'):
        h.write("item []:\n")

    for i in sorted_keys(tg['item']):
        # subkey := intervals or points?
        if re.search(tg['item'][i]['class'], 'texttier', re.I):
            subkey = 'points'
        else:
            subkey = 'intervals'
        if tg['format'] == 'long':
            h.write("{}item [{}]:\n".format(idt, i))
        for f in fld['item']:
            if tg['format'] == 'long':
                if f == 'size':
                    h.write("{}{}{}: size = {}\n".format(
                        idt, idt, subkey, tgv(tg['item'][i]['size'], 'size')))
                else:
                    h.write("{}{}{} = {}\n".format(
                        idt, idt, f, tgv(tg['item'][i][f], f)))
            else:
                h.write("{}\n".format(tgv(tg['item'][i][f], f)))

        # empty tier
        if subkey not in tg['item'][i]:
            continue
        for j in sorted_keys(tg['item'][i][subkey]):
            if tg['format'] == 'long':
                h.write("{}{}{} [{}]:\n".format(idt, idt, subkey, j))
            for f in fld[subkey]:
                if (tg['format'] == 'long'):
                    myv = tgv(tg['item'][i][subkey][j][f], f)
                    h.write(
                        "{}{}{}{} = {}\n".format(idt, idt, idt, f, myv))
                else:
                    myv = tgv(tg['item'][i][subkey][j][f], f)
                    h.write("{}\n".format(myv))
    h.close()


def tg_fields():

    ''' returns field names of TextGrid head and items '''
        
    return {'head': ['xmin', 'xmax', 'size'],
            'item': ['class', 'name', 'xmin', 'xmax', 'size'],
            'points': ['time', 'mark'],
            'intervals': ['xmin', 'xmax', 'text']}


def tgv(v, a):

    '''
    rendering of TextGrid values
    Args:
    s value
    s attributeName
    Returns:
    s renderedValue
    '''
    
    if re.search(r'(xmin|xmax|time|size)', a):
        return v
    else:
        return "\"{}\"".format(v)


def tg_tier(tg, tn):

    '''
    returns tier subdict from TextGrid
    Args:
       tg: dict by i_tg()
       tn: name of tier
    Returns:
       t: dict tier (deepcopy)
    '''

    if tn not in tg['item_name']:
        return {}
    return copy.deepcopy(tg['item'][tg['item_name'][tn]])


def tg_tn(tg):

    '''
    returns list of TextGrid tier names
    Args:
      tg: textgrid dict
    Returns:
      tn: sorted list of tiernames
    '''

    return sorted(list(tg['item_name'].keys()))


def tg_tierType(t):

    '''
    returns tier type
    Args:
      t: tg tier (by tg_tier())
    Returns:
      typ: 'points'|'intervals'|''
    '''

    for x in ['points', 'intervals']:
        if x in t:
            return x
    return ''


def tg_txtField(typ):

    '''
    returns text field name according to tier type
    Args:
      typ: tier type returned by tg_tierType(myTier)
    Returns:
      'points'|<'text'>
    '''
    
    if typ == 'points':
        return 'mark'
    return 'text'


def tg_tier2tab(t, opt={}):

    '''
    transforms TextGrid tier to 2 arrays
    point -> 1 dim + lab
    interval -> 2 dim (one row per segment) + lab
    
    Args:
      t: tg tier (by tg_tier())
      opt dict
          .skip <""> regular expression for labels of items to be skipped
                if empty, only empty items will be skipped
    
    Returns:
      x: 1- or 2-dim array of time stamps
      lab: corresponding labels
    REMARK:
      empty intervals are skipped
    '''
    
    opt = opt_default(opt, {"skip": ""})
    if len(opt["skip"]) > 0:
        do_skip = True
    else:
        do_skip = False
    x = []
    lab = []
    if 'intervals' in t:
        for i in sorted_keys(t['intervals']):
            z = t['intervals'][i]
            if len(z['text']) == 0:
                continue
            if do_skip and re.search(opt["skip"], z["text"]):
                continue

            x.append([z['xmin'], z['xmax']])
            lab.append(z['text'])
    else:
        for i in sorted_keys(t['points']):
            z = t['points'][i]
            if do_skip and re.search(opt["skip"], z["mark"]):
                continue
            x.append(z['time'])
            lab.append(z['mark'])

    x = np.array(x)
    return x, lab


def tg_tab2tier(t, lab, specs):

    '''
    transforms table to TextGrid tier
    
    Args:
       t - numpy 1- or 2-dim array with time info
       lab - list of labels <[]>
       specs['class'] <'IntervalTier' for 2-dim, 'TextTier' for 1-dim>
            ['name']
            ['xmin'] <0>
            ['xmax'] <max tab>
            ['size'] - will be determined automatically
            ['lab_pau'] - <''>
    
    Returns:
       dict tg tier (see i_tg() subdict below myItemIdx)
    for 'interval' tiers gaps between subsequent intervals will be bridged
    by lab_pau
    '''

    tt = {'name': specs['name']}
    nd = len(t.shape)
    # 2dim array with 1 col
    if nd == 2:
        nd = t.shape[1]
    # tier class
    if nd == 1:
        tt['class'] = 'TextTier'
        tt['points'] = {}
    else:
        tt['class'] = 'IntervalTier'
        tt['intervals'] = {}
        # pause label for gaps between intervals
        if 'lab_pau' in specs:
            lp = specs['lab_pau']
        else:
            lp = ''
    # xmin, xmax
    if 'xmin' not in specs:
        tt['xmin'] = 0
    else:
        tt['xmin'] = specs['xmin']
    if 'xmax' not in specs:
        if nd == 1:
            tt['xmax'] = t[-1]
        else:
            tt['xmax'] = t[-1, 1]
    else:
        tt['xmax'] = specs['xmax']
    # point tier content
    if nd == 1:
        for i in np.arange(0, len(t)):
            # point tier content might be read as [[x],[x],[x],...]
            # or [x,x,x,...]
            if listType(t[i]):
                z = t[i, 0]
            else:
                z = t[i]
            if len(lab) == 0:
                myMark = "x"
            else:
                myMark = lab[i]
            tt['points'][i+1] = {'time': z, 'mark': myMark}
        tt['size'] = len(t)
    # interval tier content
    else:
        j = 1
        # initial pause
        if t[0, 0] > tt['xmin']:
            tt['intervals'][j] = {'xmin': tt['xmin'],
                                  'xmax': t[0, 0], 'text': lp}
            j += 1
        for i in np.arange(0, len(t)):
            # pause insertions
            if ((j-1 in tt['intervals']) and
                    t[i, 0] > tt['intervals'][j-1]['xmax']):
                tt['intervals'][j] = {'xmin': tt['intervals'][j-1]['xmax'],
                                      'xmax': t[i, 0], 'text': lp}
                j += 1
            if len(lab) == 0:
                myMark = "x"
            else:
                myMark = lab[i]
            tt['intervals'][j] = {'xmin': t[i, 0],
                                  'xmax': t[i, 1], 'text': myMark}
            j += 1
        # final pause
        if tt['intervals'][j-1]['xmax'] < tt['xmax']:
            tt['intervals'][j] = {'xmin': tt['intervals'][j-1]['xmax'],
                                  'xmax': tt['xmax'], 'text': lp}
            j += 1  # so that uniform 1 subtraction for size
        # size
        tt['size'] = j-1
    return tt


def tg_add(tg, tier, opt={'repl': True}):

    '''
    add tier to TextGrid
    
    Args:
      tg dict from i_tg(); can be empty dict
      tier subdict to be added:
          same dict form as in i_tg() output, below 'myItemIdx'
      opt
         ['repl'] <True> - replace tier of same name
    
    Returns:
      tg updated
    
    REMARK:
      !if generated from scratch head xmin and xmax are taken over from the tier
       which might need to be corrected afterwards!
    '''
    
    # from scratch
    if 'item_name' not in tg:
        fromScratch = True
        tg = {'name': '', 'format': 'long', 'item_name': {}, 'item': {},
              'head': {'size': 0, 'xmin': 0, 'xmax': 0, 'type': 'ooTextFile'}}
    else:
        fromScratch = False

    # tier already contained?
    if (opt['repl'] and (tier['name'] in tg['item_name'])):
        i = tg['item_name'][tier['name']]
        tg['item'][i] = tier
    else:
        # item index
        ii = sorted_keys(tg['item'])
        if len(ii) == 0:
            i = 1
        else:
            i = ii[-1]+1
        tg['item_name'][tier['name']] = i
        tg['item'][i] = tier
        tg['head']['size'] += 1

    if fromScratch and 'xmin' in tier:
        for x in ['xmin', 'xmax']:
            tg['head'][x] = tier[x]

    return tg


def sorted_keys(x):

    '''
    returns sorted list keys
    
    Args:
      x: (dict)
    
    Returns:
      (list) of sorted keys
    '''

    return sorted(x.keys())


def opt_default(c, d):

    '''
    recursively adds default fields of dict d to dict c
       if not yet specified in c
    
    Args:
     c someDict
     d defaultDict
    
    Returns:
     c mergedDict (defaults added to c)
    '''

    if c is None:
        c = {}
    
    for x in d:
        if x not in c:
            c[x] = d[x]
        elif type(c[x]) is dict:
            c[x] = opt_default(c[x], d[x])
    return c


def listType(y):

    '''
    returns True if input is numpy array or list; else False
    '''

    if (type(y) == np.ndarray or type(y) == list):
        return True
    return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="augmentation test collection")
    parser.add_argument('-t', '--task', help='which task ("ref": create reference,\n' \
                        '"apply": apply copasul and compare with reference,\n ' \
                        '"stereo": run on larger stereo file; not public)',
                        type=str, required=True)
    kwargs = vars(parser.parse_args())
    diagnosis = tests_aug(**kwargs)

