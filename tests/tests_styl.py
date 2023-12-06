import argparse
import audeer
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


def tests(task, feature_set, column_name):
    ''' copasul test collection

    Args:
    task: (str) "ref" - create reference tables
                "apply" - create output tables and compare with reference
                "diagnose" - output file, start, end of mismatches
    feature_set: (str) to be specified for diagnose
    column_name: (str) to be specified for diagnose

    Returns:
    diagnosis:
       task "ref": None
       task "apply": (dict)
         keys:
             "pass": bool, False if error in any of tables/columns
             feature set names ("bnd", "glob", etc.)
         values:
              {"pass": bool, False if error
               "diff_columns": list of columns for which different items have
                       been found
               "miss_columns": list of missing columns in answer}
       task "diagnosis": (pd.DataFrame)
          index: [stm, t_on, t_off]
          columns: ans, ref
          rows: all rows of reference and answer dataframes for which values
                of column_name differ
    '''

    # config, featext
    f_config = os.path.normpath(f"{cwd}/minex/config/minex.json")
    with open(f_config, 'r') as h:
        opt = json.load(h)

    # example reference feature table
    f_feat = os.path.normpath(f"{cwd}/minex/reference/test.gnl.csv")
        
    #!x expand relative paths to absolute paths relative to config location
    ref_path = os.path.dirname(f_config)
    for x in opt["fsys"].keys():
        if type(opt["fsys"][x]) is dict and "dir" in opt["fsys"][x]:
            jn = os.path.join(ref_path, opt["fsys"][x]["dir"])
            opt["fsys"][x]["dir"] = os.path.abspath(jn)
            
    fex = copasul.Copasul()

    assert task in ["ref", "apply", "diagnose"], \
        "task must be 'ref', 'apply', or 'diagnose'"

    if task == "ref":

        # create reference tables

        # do not overwrite
        assert (not os.path.isfile(f_feat)), \
            "Reference tables in minex/reference already generated and " \
            "cannot be overwritten. Delete them first if intended."

        # set export directory in config to reference.
        d_exp = os.path.normpath(
            os.path.join(ref_path, "../reference"))
        _ = audeer.mkdir(d_exp)
        opt["fsys"]["export"]["dir"] = d_exp

        # process
        copa = fex.process(config=opt)

        print(f"Output written to {d_exp}. Done.")
        diagnosis = None

    elif task == "apply":

        # apply copasul and compare to reference

        # no reference
        assert os.path.isfile(f_feat), \
            "No reference tables. To be generated with tests.py -t ref"

        # process
        copa = fex.process(config=opt)
        diagnosis = {"pass": True}

        # factor variables
        fac = ['class', 'ci', 'fi', 'si', 'gi', 'stm', 'tier', 'spk',
               'is_init', 'is_fin', 'is_init_chunk', 'is_fin_chunk', 'lab',
               'lab_next', "f0_is_init", "f0_is_fin", "f0_is_init_chunk",
               "f0_is_fin_chunk", "en_is_init", "en_is_fin", "en_is_init_chunk",
               "en_is_fin_chunk", "lab_acc", "lab_ag"]
        
        # test each feature set
        fsets = ["bnd", "glob", "gnl", "gnl_file", "loc", "rhy", "rhy_file",
                 "summary", "voice", "voice_file"]
        for fset in fsets:
            f_fset = os.path.normpath(f"{cwd}/minex/reference/test.{fset}.csv")
            f_ans = os.path.normpath(f"{cwd}/minex/output/test.{fset}.csv")
            df_ref = pd.read_csv(f_fset, header=0)
            df_ans = pd.read_csv(f_ans, header=0)
            diagnosis[fset] = {
                "pass": True,
                "diff_cols": [],
                "miss_cols": []
            }
            for c in df_ref.columns:
                if c not in df_ans.columns:
                    diagnosis[fset]["miss_cols"].append(c)
                    continue
                r = df_ref[c].to_numpy()
                a = df_ans[c].to_numpy()
                if c in fac or re.search(r"^(grp|tier)_", c):
                    # categorical variables
                    if np.any(r != a):
                        diagnosis[fset]["diff_cols"].append(c)
                else:
                    # numeric variables
                    b = np.allclose(r, a, equal_nan=True)
                    if not b:
                        diagnosis[fset]["diff_cols"].append(c)

            if max(len(diagnosis[fset]["diff_cols"]),
                   len(diagnosis[fset]["miss_cols"])) > 0:
                diagnosis[fset]["pass"] = False

        for fset in fsets:
            if diagnosis[fset]["pass"]:
                print(fset, ": ok.")
            else:
                print(fset)
                if len(diagnosis[fset]["miss_cols"]) > 0:
                    print("\tmissing columns:",
                          diagnosis[fset]["miss_cols"])
                if len(diagnosis[fset]["diff_cols"]) > 0:
                    print("\tdifferent values in columns:",
                          diagnosis[fset]["diff_cols"])

    elif task == "diagnose":
        assert (not (feature_set is None or column_name is None)), \
            "for task 'diagnose', specify both feature_set and column_name."

        diagnosis = {"stm": [], "t_on": [], "t_off": [], "ref": [], "ans": []}
        f_ref = os.path.normpath(f"{cwd}/minex/reference/test.{feature_set}.csv")
        f_ans = os.path.normpath(f"{cwd}/minex/output/test.{feature_set}.csv")
        df_ref = pd.read_csv(f_ref, header=0)
        df_ans = pd.read_csv(f_ans, header=0)
        r = df_ref[column_name].to_numpy()
        a = df_ans[column_name].to_numpy()
        if np.any(r != a):

            ii = np.where(r != a)[0]
            for i in ii:
                if np.isfinite(r[i]) or np.isfinite(a[i]):
                    for u in ["stm", "t_on", "t_off"]:
                        diagnosis[u].append(df_ref[u].iloc[i])
                    diagnosis["ref"].append(r[i])
                    diagnosis["ans"].append(a[i])

        diagnosis = pd.DataFrame(diagnosis)
        diagnosis.set_index(["stm", "t_on", "t_off"], inplace=True)

        if diagnosis.shape[0] > 0:
            print(diagnosis)
        else:
            print("ok.")

    return diagnosis


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="stylization tests")
    parser.add_argument('-t', '--task', help='which task ("ref": create reference, '
                        '"apply": apply copasul and compare with reference, '
                        '"diagnose": diagnosis of specific feature set and column)',
                        type=str, required=True)
    parser.add_argument('-fset', '--feature_set', help='feature set name (glob, loc, ...) '
                        'for diagnosis', type=str, required=False, default=None)
    parser.add_argument('-col', '--column_name', help='column name in feature set table '
                        'for diagnosis', type=str, required=False, default=None)
    kwargs = vars(parser.parse_args())
    diagnosis = tests(**kwargs)

