# CHANGELOG

## Version 1.2.1 (2023-10-21)
* added: `copasul_utils.robust_corrcoef()`

## Version 1.2.0 (2023-10-20)
* general re-factoring
* bugfix in copasul_export.upd_suma_feat(): IQR calculation corrected (before it was STD; affects all `*iqr` columns `summary.csv` output)
* `copasul_sigproc.specmom()`: replaced `sum()` by `np.sum()` leading to slightly different spectral moment values (differences are smaller than 1e-06)
* `copasul_augment.aug_prep_copy()`: keep original timestamps for automatically detected accents instead of those rounded to 2nd decimal.

## Version 1.1.0 (2023-10-03)
* bugfix in copasul_styl.styl_voice_feat(): sorting pulse time stamps
* `sigFunc.py` renamed to `copasul_sigproc.py`
* added progress bars for stylization steps
* ongoing refactoring

## Version 1.0.6 (2023-09-30)
* all code pep8-ified
* docstrings unified

## Version 1.0.5 (2023-09-24)
* `copasul_init.py`, `config.json`: for boolean parameters replaced `0, 1` by `false, true`. Backward compatible to previous config-s.

## Version 1.0.4 (2023-08-28)
* `copasul_preproc.pp_loc_merge()`: fixed numpy `VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences`

## Version 1.0.3 (2022-12-23)
* `copasul_styl.styl_styl_std_feat()`: added feature "maxpos" (relative position of f0 or energy maximum)

## Version 1.0.2 (2022-11-27)
* `copasul_styl.styl_loc_fit()`: added feature "rmsd" (area under polyval)

## Version 1.0.1 (2021-05-07)
* `copasul.copasul()`: now allowing for warm start feature extraction

## Version 1.0.0 (2021-03-20)
* `Copasul()` class added (please see `example_call.py` how to integrate it into python code)
* added: `copa["export"][myFeatSet] = pd.DataFrame()` so that feature output is stored in generated copa dict. myFeatSet: `glob, loc, bnd, voice, gnl_f0, gnl_en, gnl_f0_file, gnl_en_file, rhy_f0, rhy_en, rhy_f0_file, rhy_en_file`
* output table columns (and dataframe columns in `copa["export"]`) are now alphanumerically sorted
* dict `config.plot.browse.grp`, which is processed in `copasul_plot.plot_browse()`. This dict contains zero or more `myGroupingKey-myGroupingValue` pairs. Each `myGroupingKey` should match one of the strings in `config.fsys.grp.lab`. By this the user can select to plot images with (a combination of) certain grouping values only.

## Version 0.8.28 (2020-08-26)
* augmentation log messages redirected to log file

## Version 0.8.27 (2020-08-09)
* f0 contour is now horizontally extrapolated till 0.00 and not 0.01 By this summary statistics of the `glob` and `gnl` feature not contain the same values.
* `copasul_styl.styl_polyfit()`: polyfit made more robust

## Version 0.8.26 (2020-08-07)
* filter variables `is_init(_chunk)` and `is_fin(_chunk)` now treated as factors in R output

## Version 0.8.25 (2020-07-11)
* new voice quality features
  * **jitter:** `m, sd, m_nrm, sd_nrm`: dur mean and std (+ normalized over file); `jit_nrm`: normalized over file
  * **shimmer:** `m, sd, m_nrm, sd_nrm`: amplitude mean and std (+ normalized over file); `shim_nrm`: normalized over file

## Version 0.8.24 (2020-06-20)
* default config file removed; now hard-coded in `copasul_init.py`

## Version 0.8.23 (2020-06-12)
* default config file moved to `src/` directory
* documentation update: installation within virtual environment added

## Version 0.8.22 (2019-09-09)
* uniform `ylim` in contour grouping and clustering plots
* data underlying the grouping and plotting plots is stored in pickle files in the `opt.fsys.pic` subdirectory (see `copasul_plot.plot_grp()` and `copasul_plot.plot_clst()` for details)

## Version 0.8.21 (2019-08-24)
* improved syllable extraction

## Version 0.8.20 (2019-05-10)
* new energy gnl feature: `r_en_f0`, i.e. correlation between energy and f0 contour

## Version 0.8.19 (2019-03-23)
* new parameter: `opt.styl.glob.align` for median window alignment center (default), left (no lookback), right (no lookahead)
* new parameter subdict: `opt.preproc.interp` with element `.kind`. If set then `scipy.interpolate.interp1d()` is used with `kind` argument instead of default linear interpolation by `numpy.interp()`

## Version 0.8.18 (2019-03-22)
* increasing robustness against vector length variation in `copasul_preproc.pp_zp()`

## Version 0.8.17 (2019-03-16)
* new glob features stored in `copa.data.fi.ci.glob.si.eou` as
  * `.{tl|ml|bl|rng}_drop`: actual f0 drop in IP, i.e. dur * rate
  * `.tl_ml_cross_t|f0`, `.tl_bl_cross_t|f0`, `.ml_bl_cross_t|f0`: time and f0 of crossing point of respective line pair
  * content is in glob table columns `{tl|ml|bl|rng}_drop` and `tl_ml_cross_t|f0`
* **Initial commit on Gitub**

## Version 0.8.16 (2019-03-01)
* new options `.fsys.augment.{syl|loc}.force = <0>|1`. Dependencies: if `.fsys.export.summary==1`, both `.force` are set to 1. If `loc.force==1`, also `syl.force` is set to 1. If `syl.force` is 1, at least one syllable nucleus is to be found per parent tier segment. In case no nucleus is extracted within a segment, a nucleus is placed at the energy maximum. If `loc.force` is 1, at least one pitch accent is to be found in a file. In case no accent is extracted, an accent is placed at the energy maximum. Both ensures, that for all files (also e.g. very short or noisy ones) all selected feature sets can be extracted, which is needed for `.fsys.export.summary==1`, so that all columns have the same number of rows.
* `copasul_augment.aug_loc_fv()`: bugfix; adjusting number of return arguments

## Version 0.8.15 (2019-01-16)
* new loc features: `{bl|ml|tl|rng}_d`; mean distance between local regression line and corrsponding part of global regression line. The higher, the more the local is above the global line.

## Version 0.8.14 (2018-10-29)
* `copasul_augment.aug_batch_ps()`: bugfix; adjusting number of return arguments of `aug_glob_fv()`

## Version 0.8.13 (2018-10-28)
* `copasul_augment.<divers>()`: increasing robustness.

## Version 0.8.12 (2018-10-19)
* `copasul_styl.styl_discont()`: for each register representation 2 more features of use e.g. for measuring downstep, namely `d_o`: onset difference, and `d_m`: difference of means

## Version 0.8.11 (2018-09-24)
* `copasul_preproc.pp_loc_merge()`: more robust treatment of deviating numbers of accents and accent groups. Covered by new option `preproc:loc_align`

## Version 0.8.10 (2018-09-14)
* `copasul_styl.styl_discont()`: 9 boundary featured added for each register representation: `corrD, corrD_pre, corrD_post, rmsR, rmsR_pre, rmsR_post, aicR, aicR_pre, aicR_post`

## Version 0.8.9 (2018-09-10)
* `copasul_styl.styl_discont()`: 3 slope diff boundary features added for each register representation: `sd_prepost, sd_pre, sd_post`

## Version 0.8.8 (2018-09-07)
* `futureScipy.py` removed
* `scipy version >= 0.15.1` required (that contains `scipy.signal.savgol_filter()`)

## Version 0.8.7 (2018-08-08)
* `copasul_styl.styl_discont()` updates for `do_styl_bnd_win`, and `do_styl_bnd_trend`: no empty segment warning at chunk ends resulting from option `cross_chunk=0`, but only NaN output in the output tables
* `bnd`, `bnd_win`, and `bnd_trend` stylizations can also be plotted AFTER stylization (`plot.browse.time='final'`). Not backward compatible with previously obtained pickle files.

## Version 0.8.6 (2018-08-07)
* bugfix in `copasul_styl.sdw_robust()`: occasionally affected boundary rms features

## Version 0.8.5 (2018-08-02)
* `styl.register` default changed from `'bl'` to `'ml'`
* boundary features now can also be extracted on f0 residual (i.e. after removing global register) ! usually not interpretable across global segment boundaries due to different global register subtraction

## Version 0.8.4 (2018-07-31)
* bugfix in `copasul_preproc.pp_channel()` for calculating `is_fin` column of `loc` featset

## Version 0.8.3 (2018-07-25)
* bugfix in `copasul_export.export_summary()`: now also working for stereo files. Summary statistics are now given per file and analysis tier (before: per file and channel)

## Version 0.8.2 (2018-07-11)
* `config/copasul_default_config.json`: default for `config['styl']['gnl_en']['domain']` set to 'time' since 'freq' is time-expensive

## Version 0.8.1 (2018-07-02)
* voice feature set added: `copasul_styl.styl_voice()`

## Version 0.7.12 (2018-06-26)
* `copasul_preproc.pp_channelfeatures()`: `'is_init'` and `'is_fin'` keys added (features of same name in `loc` output table). Describe position of local segment in global one.

## Version 0.7.11 (2018-06-19)
* `sigFunc.pau_detector_merge()`: merging of chunks across too short pauses no comes before merging pauses across too short chunks (expected to work better for dialogs with short backchannel chunks)

## Version 0.7.10 (2018-06-08)
* chunk-final syllable boundary time stamp added
* new phrase boundary and accent bootstrapping method working without previous word segmentation: `'seed_minmax'`

## Version 0.7.9 (2018-04-05)
* pre-emphasis for spectral balance calculation now possible in time and frequency domain (the latter according to Fant et al, 2000)
* new sub-dictionary `opt['styl']['gnl_en']['sb']` for spectral balance settings:
  * `'domain'`: `'time'` or `'freq'`; in which domain to do pre-emphasis and SPLH-SPL calculation
  * `'alpha'`: for time-domain pre-emphasis. If `>1` it is interpreted as the lower boundary frequency from which on pre-emphasis is to be carried out.
  * `'win'`: length of center window in sec within which spectral balance is to be calculated; default: -1 (length of segment)
  * `'btype'`: filter type to specify frequency band relevant for spectral balance calculation: `<'none'>|'low'|'high'|'band'`
  * 'f': cutoff frequency/ies for `'btype'` (default: -1, i.e. no filtering)
* `styl:gnl\_en:alpha` is deprecated and replaced by `styl:gnl\_en:sb:alpha`

## Version 0.7.8 (2018-03-24)
* `copasul_styl.styl_std_quot()` now robust against empty input vectors
* `copasul_augment.pau_detector_red()`: bugfix for `opt['fbnd']==True`

## Version 0.7.7 (2018-01-30)
* output tables: column separator now can freely be chosen (default ',')

## Version 0.7.6 (2018-01-25)
* chunking: silence margins can be set at chunk starts and ends

## Version 0.7.5 (2018-01-23)
* bugfix: automatic accent annotation now works also without AG definition (`copasul_augment.aug_loc_fv()`)
* chunking: changing default for `e_rel` to 0.1 fallback RMS over entire channel content now calculated on abs amplitude values > median. This prevents too many extracted chunks in signals that consist mainly of speech pauses.

## Version 0.7.4 (2018-01-19)
* event tier support for global segments: time points are treated as right boundaries

## Version 0.7.3 (2018-01-18)
* default values for outlier removal changed
  * `preproc:out:f = 2` (unchanged)
  * `preproc:out:m = 'mean'`
* median-based outlier definition changed: now Tukey's fences used, i.e. outliers defined wrt 1st and 3rd quartiles

## Version 0.7.2 (2018-01-12)
* `sigFunc.dct_peak()`: maximum finder now robust against multiple maxima

## Version 0.7.1 (2018-01-10)
* new features added to gnl feature set for both energy and f0: `qi, qf, qb, qm, c0, c1, c2`
* `copasul_preproc.pp_channel()`: for feature sets `loc`, and `loc_ext` missing local segments are not anymore replaced by the corresponding global segment. So far a global segment without any contained local one was treated as a global segment containing one local one of the same size.

## Version 0.6.4 (2017-12-21)
* informative error message for zero-only f0 input

## Version 0.6.3 (2017-12-11)
* new `rhy_f0/en` features from DCT spectrum:
  * Number of peaks in absolute DCT spectrum (n peak)
  * corresponding frequency for coefficient with amplitude maximum (f max)
  * frequency difference for each selected prosodic event rate to f max (dgm) and to the nearest peak (dlm).
* new config parameter peak_prct for DCT peak candidates
* bugfix: `styl.register == 'none'` now works with `navigate.do_clst_loc`

## Version 0.6.2 (2017-11-20)
* new features: mean values for base-, mid-, topline, and range (`bl_m, ml_m, tl_m, rng_m`)
* bugfix in `copasul.py` at call of `coin.copa_opt_init()` (bug since version `0.6.1`)

## Version 0.5.2 (2017-11-17)
* more robust processing of TextGrids with empty tiers

## Version 0.5.1 (2017-10-11)
* summary output added: mean and variation measures for all extracted features (unigram entropy for contour classes) per file

## Version 0.4.5 (2017-10-10)
* bugfix: annotation augmentation now works also from scratch, i.e. without pre-given annotation files

## Version 0.4.4 (2017-09-21)
* bugfix in spectral moment calculation in `sigFunc.py:specmom()`

## Version 0.4.3 (2017-09-05)
* time stamps (`t_on, t_off`) added to boundary feature output

## Version 0.4.2 (2017-07-18)
* some `plot:browse` options added for index verbose and selected plotting

## Version 0.4.1 (2017-05-22)
* added global segment features: `{bl|ml|tl|rng}_rate`, i.e. base/mid/topline/range rate (ST or Hz per sec)
* added extended local segment features: `{bl|ml|tl|rng}_{c0|c1|rate}`, i.e. base/mid/topline/range intercept, slope, and rate (ST or Hz per sec)
* `cost:styl_rhy()`
  * the proportion of event-rate-related DCT coefficients is calculated relative to the coefficients with frequencies between lb and ub (before: all coefficients, which makes comparisons across different signal lengths hard to interprete)
  * same for mean absolute error: IDCT by the event-rate-related DCT coefficients is compared with the IDCT by the coefficients with frequencies between lb and ub

## Version 0.3.12 (2017-05-16)
* fixed a bug resulting from version update 0.3.7. In `cost:styl_std_feat()` calls from now on not only segment but event tiers are correctly processed.
* bugfix: for `loc` feature set extraction customized options `preproc:loc:{point|nrm}_win` not anymore overwritten by default `preproc:{point|nrm}_win` values of 0.5 and 1s, respectively
* `coag:aug_pho_nrm()`: vowel duration normalization by median devision replaced by robust z-scores

## Version 0.3.11 (2017-05-02)
* `coag:aug_syl()`: now robust if time info in annotation exceeds signal file length
* `coag:aug_annot_upd()`: now also outputting (empty) tiers if no event extracted. Before a missing tier lead to an error if needed for stylization. Empty tier info is written to log file.
* `copp:pp_tier_time_to_tab()`: adjusted to cope with empty tiers
* `copp:pp_f_rate()`: adjusted to cope with empty tiers
* `cost:styl_std_feat()`: now robust against empty signal input
* `config fsys:export:fullpath`: +/- fullpath switch to csv tables in R code output

## Version 0.3.10 (2017-04-05)
* bugfix: file part omission in R template output file stems containing a '.' are not further split into stem and extension

## Version 0.3.9 (2017-03-28)
* file path to csv omitted in R template output

## Version 0.3.8 (2017-03-27)
* index column removed from csv output to ease reading with R

## Version 0.3.7 (2017-03-21)
* `cost:styl_std_feat()`: duration `dur` features now calcaluated more precisely based on original segment on- and offsets. So far it was derived from the number of f0 samples in the resp. segment

## Version 0.3.6 (2017-03-06)
* bugfix in `coag:aug_annot_init()`: empty string default for `timeStamp`
* `cost:styl_glob()`: eval for N=0 no robust NaN instead of error

## Version 0.3.5 (2017-02-24)
* bugfix in `coag:aug_fv_mrg`: higher numpy versions do not support column concat with empty arrays. append() calls corrected accordingly.

## Version 0.3.3 (2017-02-16)
* bugfix in `copl.plot_browse_channel`: declination plot now possible also without preceding local contour stylization

## Version 0.3.2 (2017-02-06)
* bugfix in `cost.styl_rhy_file`: now working for stereo files

## Version 0.3.1 (2017-02-03)
* `pho` featset now also usable for augmentation with `unit=file` (so far the combination of `unit=file` and `wgt:pho=1` lead to an non-infomative error)

## Version 0.2.2 (2017-01-30)
* bugfix `coex.export_rhy`: now working for stereo files

## Version 0.2.1 (2017-01-25)
* bugfix in `cost.styl_gnl_file`: accent batch augmentation in stereo files fixed
