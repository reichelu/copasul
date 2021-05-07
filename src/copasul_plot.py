
# author: Uwe Reichel, Budapest, 2016

import mylib as myl
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import sys
import numpy as np
import math
import copy as cp

# browsing through copa
# calling plot_main for selected feature sets
# do clst first (single plot)
def plot_browse(copa):
    
    c = copa['data']
    
    opt = copa['config']
    o = opt['plot']['browse']

    # not for online usage
    if o['time'] != 'final': return

    ### clustering ###########
    if o['type']['clst'] and o['type']['clst']['contours']:
        plot_main({'call':'browse','state':'final','type':'clst','set':'contours','fit':copa},opt)

    ### stylization ##########
    ## domains
    for x in sorted(o['type'].keys()):
        if x=='clst': continue
        ## featsets
        for y in o['type'][x]:
            if not o['type'][x][y]: continue
            ## files
            for ii in myl.numkeys(c):
                ## channels
                for i in myl.numkeys(c[ii]):
                    
                    # check grouping constraints ("and"-connected)
                    if "grp" in o:
                        do_plot = True
                        for g in o["grp"]:
                            if ((g not in c[ii][i]["grp"]) or
                                (c[ii][i]["grp"][g] != o["grp"][g])):
                                do_plot = False
                                break
                        if not do_plot:
                            continue
                        
                    plot_browse_channel(copa,x,y,ii,i)

# channelwise processing of plot_browse()
# IN:
#   copa
#   typ type  'glob'|'loc'|'rhy_f0'|'complex'|...
#   s   set   'decl'|'acc'|'rhy'|'superpos'|'gestalt'|'bnd'...
#   ii fileIdx
#   i channelIdx
def plot_browse_channel(copa,typ,s,ii,i):
    
    c = copa['data']
    po = copa['config']['plot']['browse']
    # time, f0, f0-residual
    t = c[ii][i]['f0']['t']
    y = c[ii][i]['f0']['y']
    if 'r' in c[ii][i]['f0']:
        r = c[ii][i]['f0']['r']
    else:
        r = c[ii][i]['f0']['y']

    # for all complex plots range over global segments
    if typ == 'complex':
        if re.search('^bnd',s):
            dom = 'bnd'
        else:
            dom = 'glob'
    else:
        dom = typ

    # file stem to be displayed
    myStm = c[ii][i]['fsys']['f0']['stm']

    ## segments
    for j in myl.numkeys(c[ii][i][dom]):
        # verbose to find plot again
        if myl.ext_true(po,'verbose'):
            print("file_i={}, channel_i={}, segment_i={}".format(ii,i,j))
        # skip all but certain segment is reached
        if ('single_plot' in po) and myl.ext_true(po['single_plot'],'active'):
            if (ii != po['single_plot']['file_i'] or
                i != po['single_plot']['channel_i'] or
                j != po['single_plot']['segment_i']):
                continue

        if typ != 'complex':
            if re.search('^rhy_',typ):
                ## tiers
                for k in myl.numkeys(c[ii][i][dom][j]):
                    myFit = c[ii][i][dom][j][k][s]
                    myInfx = "{}-{}-{}-{}".format(ii,i,c[ii][i][dom][j][k]['tier'],k)
                    myTim = c[ii][i][dom][j][k]['t']
                    myTier = c[ii][i][dom][j][k]['tier']
                    myLoc = "{}:{}:[{} {}]".format(myStm,myTier,myTim[0],myTim[1])
                    obj = {'call':'browse','state':'final',
                           'fit':myFit,'type':typ,'set':s,
                           'infx':myInfx,'local':myLoc}

                    ## sgc on
                    #sgc_rhy_wrp(obj,c[ii][i]['rhy_en'][j][k][s],
                    #            c[ii][i]['rhy_f0'][j][k][s],copa['config'])
                    #continue
                    ## sgc off

                    plot_main(obj,copa['config'])
            else:
                myFit = cp.deepcopy(c[ii][i][dom][j][s])
                myLoc = plot_loc(c,ii,i,dom,j,myStm)

                if dom == "glob" and s == "decl" and "eou" in c[ii][i][dom][j]:
                    myFit["eou"] = c[ii][i][dom][j]["eou"]
                
                if s=='acc':
                    ys = myl.copa_yseg(copa,dom,ii,i,j,t,r)
                else:
                    ys = myl.copa_yseg(copa,dom,ii,i,j,t,y)
                    
                obj = {'call':'browse','state':'final',
                       'fit':myFit,'type':typ,'set':s,'y':ys,
                       'infx':"{}-{}-{}".format(ii,i,j),'local':myLoc}
                plot_main(obj,copa['config'])
        else:
            if re.search('^bnd',s):
                # get key depending on s
                if s=='bnd':
                    z='std'
                elif s=='bnd_win':
                    z='win'
                else:
                    z='trend'
                ## tiers
                for k in myl.numkeys(c[ii][i][dom][j]):
                    if z not in c[ii][i][dom][j][k] or 'plot' not in c[ii][i][dom][j][k][z]:
                        continue
                    myObj = c[ii][i][dom][j][k][z]['plot']
                    if 'lab' in c[ii][i][dom][j][k][z]:
                        myLab = c[ii][i][dom][j][k][z]['lab']
                    else:
                        myLab = ''
                    myInfx = "{}-{}-{}-{}".format(ii,i,c[ii][i][dom][j][k]['tier'],k)
                    myTim = c[ii][i][dom][j][k]['t']
                    myTier = c[ii][i][dom][j][k]['tier']
                    myLoc = "{}:{}:[{} {}]".format(myStm,myTier,myTim[0],myTim[1])
                    obj = {'call': 'browse', 'state': 'final',
                           'type': 'complex', 'set': s,
                           'fit': myObj['fit'],
                           'y': myObj['y'],
                           't': myObj['t'],
                           'infx': myInfx,
                           'local': myLoc}
                    plot_main(obj,copa['config'])
            else:
                myLoc = plot_loc(c,ii,i,dom,j,myStm)
                if 'lab' in c[ii][i][dom][j]:
                    myLab = c[ii][i][dom][j]['lab']
                else:
                    myLab = ''
                obj = {'call':'browse','state':'final','fit':copa,'type':'complex',
                       'set':s,'i':[ii,i,j],'infx':"{}-{}-{}".format(ii,i,j),'local':myLoc,
                       'lab': myLab, 't_glob': c[ii][i]['glob'][j]['to'], 'stm': myStm}
                plot_main(obj,copa['config'])

    return

def plot_loc(c,ii,i,dom,j,myStm):
    myTim = c[ii][i][dom][j]['t']
    if 'tier' in c[ii][i][dom][j]:
        myTier = c[ii][i][dom][j]['tier']
        myLoc = "{}:{}:[{} {}]".format(myStm,myTier,myTim[0],myTim[1])
    else:
        myLoc = "{}:[{} {}]".format(myStm,myTim[0],myTim[1])
    return myLoc


# wrapper around plotting
# IN:
#   obj dict depending on caller function
#      'call' - 'browse'|'grp'
#             'browse': iterate through copa file x channel x segment. 1 Plot per iter step
#                       + 1 plot for clustering
#             'grp': mean contours etc according to groupings, 1 Plot per Group) 
#      'state' - 'online'|'final'|'group' for unique file names
#      'type' - 'glob'|'loc'|'clst'|'rhy_f0'|'rhy_en'|'complex'
#      'set' - 'decl'|'acc'|'rhy'|'superpos'|'gestalt'
#         ... restrictions in combining 'type' and 'set'
#         ... for several types ('clst', 'rhy_*') only one set is available,
#             thus set spec just serves the purpose of uniform processing
#             but is not (yet) used in plot_* below
#      'fit' - fit object depending on styl domain (can be entire copa dict)
#         ... depending on calling function, e.g. to generate unique file names
#   opt copa['config']
def plot_main(obj,opt):
    po = opt['plot']
    
    if plot_doNothing(obj,opt): return
    
    if obj['call']=='browse':

        # display location
        if 'local' in obj:
            print(obj['local'])
        elif 'infx' in obj:
            print(obj['infx'])

        if obj['type']=='clst':
            fig = plot_clst(obj,opt)
        elif re.search('^(glob|loc)$',obj['type']):
            fig = plot_styl_cont(obj,opt['fsys']['pic'])
        elif re.search('^rhy',obj['type']):
            fig = plot_styl_rhy(obj,opt)
        elif obj['type']=='complex':
            if re.search('(superpos|gestalt)',obj['set']):
                fig = plot_styl_complex(obj,opt)
            elif re.search('^bnd',obj['set']):
                fig = plot_styl_bnd(obj,opt)

        # save plot
        if (po['browse']['save'] and fig):
            # output file name
            fs = opt['fsys']['pic']
            fb = "{}/{}".format(fs['dir'],fs['stm'])
            if 'infx' in obj:
                fo = "{}_{}_{}_{}_{}.png".format(fb,obj['state'],obj['type'],obj['set'],obj['infx'])
            else:
                fo = "{}_{}_{}_{}.png".format(fb,obj['state'],obj['type'],obj['set'])
            fig.savefig(fo)

    elif obj['call']=='grp':
        plot_grp(obj,opt)

    return

# checks type-set compliance
# IN:
#   obj - object passed to plot_main() by caller
#   opt - copa[config]
# OUT:
#   True if type/set not compliant, else False
def plot_doNothing(obj,opt):
    if not opt['navigate']['do_plot']:
        return True
    po = opt['plot']
    # final vs online
    if ((obj['call'] != 'grp') and (po[obj['call']]['time'] != obj['state'])):
        return True
    # type not specified
    if not po[obj['call']]['type'][obj['type']]:
        return True
    # type/set not compliant
    if obj['set'] not in po[obj['call']]['type'][obj['type']]:
        return True
    # type-set set to 0 in config
    if not po[obj['call']]['type'][obj['type']][obj['set']]:
        return True

    #### customized func to skip specified data portions ##########
    # ! comment if not needed !
    #return plot_doNothing_custom_senta_coop(obj,opt)
    ###############################################################
    
    return False

def plot_doNothing_custom_senta_coop(obj,opt):
    stm, t =  '16-02-116-216-Coop', 33.4
    if not obj['stm'] == stm:
        return True
    if obj['t_glob'][0] < t:
        return True
    return False


def plot_doNothing_custom_senta_comp(obj,opt):
    if not obj['stm'] == '10-04-110-210-Comp':
        return True
    if obj['t_glob'][0] < 98.3:
        return True
    return False


def plot_doNothing_custom2(obj,opt):
    cond = 'Coop'
    if not re.search(cond,obj['local']):
        return True
    if not re.search('RW',obj['lab']):
        return True
    return False

def plot_doNothing_custom(obj,opt):
    if not re.search('hun003',obj['local']):
        return True

    return False

# to be customized by user: criteria to skip certain segments
def plot_doNothing_custom1(obj,opt):
    if (('local' in obj) and (not re.search('fra',obj['local']))):
        return True

    return False

# plot 3 declination objects underlying boundary features
def plot_styl_bnd(obj,opt):

    if 'fit' not in obj:
        return

    # new figure
    fig = plot_newfig()
    
    # segment a.b
    bid = plot_styl_cont({'fit':obj['fit']['ab'],'type':'glob','set':'decl','y':obj['y']['ab'],
                          't':obj['t']['ab'],'tnrm':False,'show':False,'newfig':False},opt)
    # segment a
    bid = plot_styl_cont({'fit':obj['fit']['a'],'type':'glob','set':'decl',
                          't':obj['t']['a'],'tnrm':False,'show':False,'newfig':False},opt)
    
    # segment b
    bid = plot_styl_cont({'fit':obj['fit']['b'],'type':'glob','set':'decl',
                          't':obj['t']['b'],'tnrm':False,'show':False,'newfig':False},opt)

    plt.show()
    return fig

# plotting mean contours by grouping
# only supported for glob|loc
# principally not supported for varying number of variables, since
# then mean values cannot be derived (e.g. gestalt: n AGs per IP)
def plot_grp(obj,opt):
    dom = obj['type']
    mySet = obj['set']
    if not re.search('^(glob|loc)$',dom):
        sys.exit("plot by grouping is supported only for glob and loc feature set")
    c = obj['fit']['data']
    h = plot_harvest(c,dom,mySet,opt['plot']['grp']['grouping'])

    # output file stem
    fb = "{}/{}".format(opt['fsys']['pic']['dir'],opt['fsys']['pic']['stm'])

    # normalized time
    t = np.linspace(opt['styl'][dom]['nrm']['rng'][0],opt['styl'][dom]['nrm']['rng'][1],100)

    ## over groupings
    # pickle file storing the underlying data
    #    as dict:
    #      'x': normalized time
    #      'ylim': [minF0, maxF0] over all groupings for unifying the plots ylims 
    #      'y':
    #         myGroup:
    #           if mySet=="acc":
    #              F0 array
    #           if mySet=="decl":
    #              'bl'|'ml'|'tl':
    #                 F0 array
    # getting ylim over all groups for unified yrange
    fo_data = "{}_{}_{}_{}_grpPlotData.pickle".format(fb,obj['call'],obj['type'],obj['set'])
    plot_data = {'x': t, 'ylim': [], 'y': {}}
    all_y = []
    for x in h:
        if mySet=='acc':
            y = np.polyval(np.mean(h[x],axis=0),t)
            plot_data["y"][x] = y
            all_y.extend(y)
        elif mySet=='decl':
            plot_data["y"][x] = {}
            for reg in ['bl','ml','tl']:
                y = np.polyval(np.mean(h[x][reg],axis=0),t)
                plot_data["y"][x][reg] = y
                all_y.extend(y)
    plot_data['ylim'] = [np.min(all_y)-0.2, np.max(all_y)+0.2]

    # again over all groups, this time plotting
    for x in h:
        fo = "{}_{}_{}_{}_{}.png".format(fb,obj['call'],obj['type'],obj['set'],x)
        if mySet=='acc':
            fig = plot_styl_cont({'fit':{'tn':t, 'y':plot_data["y"][x]},
                                  'type':dom,'set':mySet,'show':False,
                                  'ylim':plot_data['ylim']},opt)
            fig.savefig(fo)
            plt.close()
        elif mySet=='decl':
            o = {'fit':{'tn':t},'type':dom,'set':mySet,'show':False,'ylim':plot_data['ylim']}
            for reg in ['bl','ml','tl']:
                o['fit'][reg]={}
                o['fit'][reg]['y'] = plot_data["y"][x][reg]
            fig = plot_styl_cont(o,opt)
            fig.savefig(fo)
            plt.close()

    myl.output_wrapper(plot_data,fo_data,'pickle')

# returns dict with feature matrices of glob|loc etc for each grouping key
# IN:
#   c copa['data']
#   dom 'glob'|'loc'|...
#   mySet 'acc'|'decl'|...
#   grp list of grouping fields in opt['plot']['grp']['grouping']
# OUT:
#   h[myGrpKey] -> myFeatMatrix or mySubdict
#      ... mySet=='decl' (for dom: glob and loc)
#             -> ['bl'|'ml'|'tl'] -> [coefMatrix]
#          mySet=='acc'
#             -> [coefMatrix]
def plot_harvest(c,dom,mySet,grp):
    h={}
    # files
    for ii in myl.numkeys(c):
        # channels
        for i in myl.numkeys(c[ii]):
            # segments
            for j in myl.numkeys(c[ii][i][dom]):
                # grouping key
                gk = copa_grp_key(grp,c,dom,ii,i,j)
                # decl set (same for dom='glob'|'loc')
                if mySet == 'decl':
                    # new key
                    if gk not in h:
                        h[gk]={}
                        for x in ['bl','ml','tl']:
                            h[gk][x]=np.asarray([])
                    for x in ['bl','ml','tl']:
                        h[gk][x] = myl.push(h[gk][x],c[ii][i][dom][j][mySet][x]['c'])
                elif mySet == 'acc':
                    # new key
                    if gk not in h:
                        h[gk]=np.asarray([])
                    h[gk] = myl.push(h[gk],c[ii][i][dom][j][mySet]['c'])
    return h

# grouping key string
# IN:
#   grp list of grouping fields
#   c  copa['data']
#  dom 'glob'|'loc'...
#  ii fileIdx
#  i channelIdx
#  j segmentIdx
# OUT:
#   string of groupingValues separated by '_'
#  variable values can be taken from:
#    1. 'file': ii
#    2. 'channel': i
#    3. 'lab'|'lab_(pnt|int)': c[ii][i][dom][j]['lab']
#    4. 'lab_next': c[ii][i][dom][j+1]['lab'] if available, else '#'
#    5. myVar: c[ii][i]['grp'][myVar]
# includes no grouping (if grp==[])
def copa_grp_key(grp,c,dom,ii,i,j):
    key = []
    for x in grp:
        if x=='file':
            key.append(str(ii))
        elif x=='channel':
            key.append(str(i))
        else:
            if (x=='lab' or re.search('lab_(pnt|int)',x)):
                z = c[ii][i][dom][j]
            elif x=='lab_next':
                if j+1 in c[ii][i][dom]:
                    z = c[ii][i][dom][j+1]
                else:
                    z = {}
            else:
                z = c[ii][i]['grp']
            if x in z:
                key.append(str(z[x]))
            else:
                key.append('#')
    return "_".join(key)



# init new figure with onclick->next, keypress->exit
# figsize can be customized
# IN:
#   fs tuple <()>
# OUT:
#   figure object
# OUT:
#   figureHandle
def plot_newfig(fs=()):
    if len(fs)==0:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=fs)
    cid1 = fig.canvas.mpl_connect('button_press_event', onclick_next)
    cid2 = fig.canvas.mpl_connect('key_press_event', onclick_exit)
    return fig


def plot_newfig_big():
    fig = plt.figure(figsize=(15,15))
    cid1 = fig.canvas.mpl_connect('button_press_event', onclick_next)
    cid2 = fig.canvas.mpl_connect('key_press_event', onclick_exit)
    return fig

def plot_newfig_verybig():
    fig = plt.figure(figsize=(25,25))
    cid1 = fig.canvas.mpl_connect('button_press_event', onclick_next)
    cid2 = fig.canvas.mpl_connect('key_press_event', onclick_exit)
    return fig

#  plots IP-AG superposition or gestalt
# IN:
#   obj dict
#     'fit' -> copa
#     'i' -> [myFileIdx myChannelIdx myGlobSegIdx]
#     'infx' -> myFileIdx_myChannelIdx_myGlobSegIdx
#   opt
def plot_styl_complex(obj,opt):
    c = obj['fit']['data']
    ii,i,j = obj['i']

    # global segment
    gs = c[ii][i]['glob'][j]

    # orig time and f0
    t = c[ii][i]['f0']['t']
    y = c[ii][i]['f0']['y']
    tg = c[ii][i]['glob'][j]['t']
    yi = myl.find_interval(t,tg)
    ys = y[yi]

    # unnormalized time
    ttg = np.linspace(tg[0],tg[1],len(ys))

    # new figure
    fig = plot_newfig()

    # plot F0 and global styl
    bid = plot_styl_cont({'fit':gs['decl'],'type':'glob','set':'decl','y':ys,
                          't':ttg,'tnrm':False,'show':False,'newfig':False},opt)

    # register to be added to loc contours
    reg = gs['decl'][opt['styl']['register']]['y']

    # over local segments
    for jj in c[ii][i]['glob'][j]['ri']:
        ls = cp.deepcopy(c[ii][i]['loc'][jj])
        tl = ls['t'][0:2]
        ttl = np.linspace(tl[0],tl[1],len(ls['acc']['y']))
        if obj['set']=='superpos':

            # add/denormFor register
            if re.search('^(bl|ml|tl)$',opt['styl']['register']):
                # part in local segment
                regs = np.asarray(reg[myl.find_interval(ttg,tl)])
                while len(regs) < len(ttl):
                    regs = myl.push(regs,regs[-1])
                ls['acc']['y'] = ls['acc']['y']+regs
                
            elif opt['styl']['register'] == 'rng':
                bl_seg = gs['decl']['bl']['y'][myl.find_interval(ttg,tl)]
                tl_seg = gs['decl']['tl']['y'][myl.find_interval(ttg,tl)]
                yy = ls['acc']['y']
                zz = np.asarray([])
                for u in range(len(bl_seg)):
                    zz = myl.push(zz,bl_seg[u]+yy[u]*(tl_seg[u]-bl_seg[u]))
                ls['acc']['y'] = zz
            bid = plot_styl_cont({'fit':ls['acc'],'type':'loc','set':'acc','t':ttl,
                                  'tnrm':False,'show':False,'newfig':False},opt)
        else:
            ls['decl']['t']=np.linspace(ls['t'][0],ls['t'][1],len(ls['acc']['y']))
            bid = plot_styl_cont({'fit':ls['decl'],'type':'loc','set':'decl','t':ttl,
                                  'tnrm':False,'show':False,'newfig':False},opt)
    plt.show()
    return fig


# plotting register and local contours
# IN:
#   OBJ
#      .fit dict returned by styl_decl_fit or styl_loc_fit
#      .type 'glob'|'loc'
#      .y   original y values
#   OPT copa['config']
def plot_styl_cont(obj,opt):

    # to allow embedded calls (e.g. complex-superpos/gestalt)
    for x in ['show','newfig','tnrm']:
        if x not in obj:
            obj[x]=True

    if obj['newfig']:
        fig = plot_newfig()
    else:
        fig = 0

    if obj['tnrm']:
        tn = obj['fit']['tn']
    else:
        tn = obj['t']

    # type=='glob' --> 'set'=='decl'
    if obj['set']=='decl':
        
        bl = obj['fit']['bl']['y']
        ml = obj['fit']['ml']['y']
        tl = obj['fit']['tl']['y']

        tbl,ybl = phal(tn,bl)
        tml,yml = phal(tn,ml)
        ttl,ytl = phal(tn,tl)

        # + original f0
        if 'y' in obj:
            myTn,myY = phal(tn,obj['y'])
            plt.plot(myTn,myY,'k.',linewidth=1)

        # color specs
        if ("plot" in opt and myl.ext_true(opt['plot'],'color')) or myl.ext_true(opt,'color'):
            cc = ['-g','-r','-c']
            lw = [4,4,4]
        else:
            #cc = ['--k','-k','--k']
            cc = ['-k','-k','-k']
            if obj['type']=='glob':
                #cc = ['--k','-k','--k']
                lw = [3,4,3]
            else:
                #cc = ['-k','-k','-k']
                lw = [5,5,5]
        plt.plot(tbl,ybl,cc[0],linewidth=lw[0])
        plt.plot(tml,yml,cc[1],linewidth=lw[1])
        plt.plot(ttl,ytl,cc[2],linewidth=lw[2])
        #plt.plot(tbl,ybl,cc[0],tml,yml,cc[1],ttl,ytl,cc[2],linewidth=4)


        # plot line crossings #!v
        #if "eou" in obj["fit"]:
        #    z = obj["fit"]["eou"]
        #    plt.plot([z["tl_ml_cross_t"],z["tl_bl_cross_t"],z["ml_bl_cross_t"]],
        #             [z["tl_ml_cross_f0"],z["tl_bl_cross_f0"],z["ml_bl_cross_f0"]],"or",linewidth=20)

    else:
        if 'y' in obj:
            myTn,myY = phal(tn,obj['y'])
            plt.plot(myTn,myY,'k.',linewidth=1)
        myTn,myY = phal(tn,obj['fit']['y'])

        # color specs
        if ("plot" in opt and myl.ext_true(opt['plot'],'color')) or myl.ext_true(opt,'color'):
            cc = '-b'
        else:
            cc = '-k'

        plt.plot(myTn,myY,cc,linewidth=6)

    # ylim
    if 'ylim' in obj:
        plt.ylim((obj['ylim'][0], obj['ylim'][1]))
        
    if obj['tnrm']:
        plt.xlabel('time (nrm)')
    else:
        plt.xlabel('time (s)')
    plt.ylabel('f (ST)')
    if obj['show']:
        plt.show()
    
    return fig
    

# hacky length adjustment
def phal(t,y):
    return myl.hal(cp.deepcopy(t),cp.deepcopy(y))
    

# speech rhythm verbose
# x - frequency
# y - coef
def plot_styl_rhy(obj,opt):
    rhy = obj['fit']
    if len(rhy['f'])==0 or len(rhy['c'])==0:
        return
    rb = opt['styl'][obj['type']]['rhy']['wgt']['rb']

    # color
    if opt['plot']['color']:
        doco = True
    else:
        doco = False
    
    #if 'SYL_2' in rhy['wgt']:    #!csl
    #    del rhy['wgt']['SYL_2']  #!csl

    fig, spl = plt.subplots(len(rhy['wgt'].keys()),1,squeeze=False)
    cid1 = fig.canvas.mpl_connect('button_press_event', onclick_next)
    cid2 = fig.canvas.mpl_connect('key_press_event', onclick_exit)
    fig.subplots_adjust(hspace=0.8)

    # domain-influence window
    i=0
    c_sum = sum(abs(rhy['c']))
    # tiers
    for x in sorted(rhy['wgt'].keys()):
        if doco:
            spl[i,0].stem(rhy['f'],abs(rhy['c'])/c_sum)
        else:
            mla,sla,bla = spl[i,0].stem(rhy['f'],abs(rhy['c'])/c_sum, '-.')
            plt.setp(sla, 'color', 'k', 'linewidth', 2)

        tit = x
        #tit = 'influence on f0' #!csl
        #spl[i,0].title.set_text(tit, fontsize=18)
        spl[i,0].set_title(tit, fontsize=18)
        r = rhy['wgt'][x]['rate']
        b = [max([0,r-rb]), r+rb]
        w = myl.intersect(myl.find(rhy['f'],'>=',b[0]),
                          myl.find(rhy['f'],'<=',b[1]))
        if len(w)==0:
            continue
        ml,sl,bl = spl[i,0].stem(rhy['f'][w],abs(rhy['c'][w])/c_sum)
        if doco:
            plt.setp(sl, 'color', 'r', 'linewidth', 3)
        else:
            plt.setp(sl, 'color', 'k', 'linewidth', 4)

        # local maxima (green lines)
        #if 'f_lmax' in rhy:
        #    for fm in rhy['f_lmax']:
        #        spl[i,0].plot([fm,fm],[0,rhy['c_cog']/c_sum],'-g',linewidth=5)

        # 1st spectral moment (thick black vertical line)
        spl[i,0].plot([rhy['sm'][0],rhy['sm'][0]],[0,rhy['c_cog']/c_sum],'-k',linewidth=5)

        #plt.ylim([0,0.4]) #!csl

        spl[i,0].set_xlabel('f (Hz)', fontsize=18)
        spl[i,0].set_ylabel('|coef|', fontsize=18)
        i+=1

    plt.show()

    return fig


# IN:
#   copa dict
#   opt copa['config']
# OUT:
#   fig object
#   pickle file output to "opt[fsys][dir]/opt[fsys][stm]_clstPlotData.pickle"
#     that stores dict of the following structure:
#       {"glob"|"loc"}
#         "ylim": ylim for uniform y-range in all subplots
#         "x": array of normalized time
#         "y":
#           myClassIndex: F0 Array
def plot_clst(obj,opt):
    copa = obj['fit']

    ## return
    if (('cntr' not in copa['clst']['glob']) or
        ('cntr' not in copa['clst']['loc'])):
        print('no clustering result to plot. Apply clustering first!')
        return False

    ## color
    if opt['plot']['color']:
        cc = 'b'
    else:
        cc = 'k'

    ## glob
    # time
    rg = copa['config']['styl']['glob']['nrm']['rng'];
    tg = np.linspace(rg[0],rg[1],100)
    
    # coefs
    cg = copa['clst']['glob']['cntr']
    ## loc
    # time
    rl = copa['config']['styl']['loc']['nrm']['rng'];
    tl = np.linspace(rl[0],rl[1],100)
    # coefs
    cl = copa['clst']['loc']['cntr']

    # number of subplots/rows, columns
    nsp = len(cl)+1
    nrow, ncol = nn_subplots(nsp)
    
    fig, spl = plt.subplots(nrow,ncol) #,figsize=(15,60))

    cid1 = fig.canvas.mpl_connect('button_press_event', onclick_next)
    cid2 = fig.canvas.mpl_connect('key_press_event', onclick_exit)

    #fig.tight_layout()
    fig.subplots_adjust(hspace=0.8)

    i_row=0
    i_col=0

    # glob and loc y values
    (yg, yl) = ([], [])
    for i in range(len(cg)):
        yg = myl.push(yg,np.polyval([cg[i,:],0],tg))
    for i in range(len(cl)):
        yl = myl.push(yl,np.polyval(cl[i,:],tl))
    
    ylim_g = [int(math.floor(np.min(yg))),
              int(math.ceil(np.max(yg)))]
    ylim_l = [int(math.floor(np.min(np.min(yl)))),
              int(math.ceil(np.max(np.max(yl))))]

    plot_data = {"glob": {"ylim": ylim_g, "x": tg, "y": {}},
                 "loc": {"ylim": ylim_l, "x": tl, "y": {}}}
    
    for i in range(nsp):
        if i==0:
            for j in range(len(yg)):
                spl[i_row,i_col].plot(tg,yg[j,:],cc,label="{}".format(j+1))
                plot_data["glob"]["y"][j] = yg[j,:]
            spl[i_row,i_col].set_title("g_*")
            spl[i_row,i_col].set_ylim(ylim_g)
            spl[i_row,i_col].set_xlabel('time (nrm)')
            spl[i_row,i_col].set_ylabel('f (ST)')
            #spl[i_row,i_col].legend(loc='best', fontsize=8)
        else:
            spl[i_row,i_col].plot(tl,yl[i-1,:],cc)
            spl[i_row,i_col].set_title("l_{}".format(i-1))
            #spl[i_row,i_col].set_ylim(ylim_l)
            plot_data["loc"]["y"][i-1] = yl[i-1,:]
            #if i>1:
            #    spl[i_row,i_col].set_xticks([])
            #    spl[i_row,i_col].set_yticks([])
        if i_col==ncol-1:
            i_row+=1
            i_col=0
        else:
            i_col+=1
    
    fo_data = "{}/{}_clstPlotData.pickle".format(opt["fsys"]["pic"]["dir"],
                                                 opt["fsys"]["pic"]["stm"])
    myl.output_wrapper(plot_data,fo_data,"pickle")
    plt.show()
    return fig

# returns optimal nrow, ncol depending on num of subplots (inpur)
def nn_subplots(n):
    if n <= 4:
        ncol = 2
        nrow = 2
    else:
        ncol = 3
        nrow = int(math.ceil(n/ncol))
    return nrow, ncol


# klick on plot -> next one
def onclick_next(event):
    plt.close()

# press key -> exit
def onclick_exit(event):
    sys.exit()


############# customized functions for publications etc #################

# for slovak game corpus visualization only!
# syl rate influence on energy and f0 contour
#  tier SYL_1 only
def sgc_rhy_wrp(obj,rhy_en,rhy_f0,opt):

    if not re.search('turns',obj['local']):
        return

    fig, spl = plt.subplots(1,2)
    cid1 = fig.canvas.mpl_connect('button_press_event', onclick_next)
    cid2 = fig.canvas.mpl_connect('key_press_event', onclick_exit)
    fig.subplots_adjust(hspace=0.8)

    # energy
    rhy = rhy_en
    if len(rhy['f'])==0 or len(rhy['c'])==0:
        return
    c_sum = sum(abs(rhy['c']))
    r = rhy['wgt']['SYL_1']['rate']
    mla,sla,bla = spl[0].stem(rhy['f'],abs(rhy['c'])/c_sum, '-.')
    plt.setp(sla, 'color', 'k', 'linewidth', 3)
    spl[0].title.set_text('influence on energy')
    rb = opt['styl']['rhy_en']['rhy']['wgt']['rb']
    b = [max([0,r-rb]), r+rb]
    w = myl.intersect(myl.find(rhy['f'],'>=',b[0]),
                      myl.find(rhy['f'],'<=',b[1]))
    if len(w)==0:
        return
    ml,sl,bl = spl[0].stem(rhy['f'][w],abs(rhy['c'][w])/c_sum)
    plt.setp(sl, 'color', 'k', 'linewidth', 4)
    spl[0].set_xlabel('f (Hz)')
    spl[0].set_ylabel('|coef|')

    # f0
    rhy = rhy_f0
    if len(rhy['f'])==0 or len(rhy['c'])==0:
        return
    c_sum = sum(abs(rhy['c']))
    r = rhy['wgt']['SYL_1']['rate']
    mla,sla,bla = spl[1].stem(rhy['f'],abs(rhy['c'])/c_sum, '-.')
    plt.setp(sla, 'color', 'k', 'linewidth', 3)
    spl[1].title.set_text('influence on F0')
    rb = opt['styl']['rhy_f0']['rhy']['wgt']['rb']
    b = [max([0,r-rb]), r+rb]
    w = myl.intersect(myl.find(rhy['f'],'>=',b[0]),
                      myl.find(rhy['f'],'<=',b[1]))
    if len(w)==0:
        return
    ml,sl,bl = spl[1].stem(rhy['f'][w],abs(rhy['c'][w])/c_sum)
    plt.setp(sl, 'color', 'k', 'linewidth', 4)
    spl[1].set_xlabel('f (Hz)')
    spl[1].set_ylabel('|coef|')


    plt.show()
