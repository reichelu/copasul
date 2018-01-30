import os
import re

global python_path

### adjust: Python3 interpreter call ###########
# Linux, Mac: something like
python_path = '#!/usr/bin/env python3'
# Windows: something like
#python_path = '#!C:\my\Path\to\python3.exe'
# Empty string if not needed
#python_path = ''
################################################


# returns files incl full path as list (recursive dir walk)
def file_collector(d,e):
    ff=[]
    for root, dirs, files in os.walk(cwd):
        files.sort()
        for f in files:
            if f.endswith(e):
                ff.append(os.path.join(root, f))
    return ff

# insert current working directory of this make file into
# python scripts for module localization
def insert_cwd(f):
    d = []
    if re.search('copasul_root\.py',f):
        d.append("{}\n".format(python_path))
        d.append("import os\n")
        d.append("import sys\n")
        d.append("sys.path.insert(0,'{}/')\n".format(cwd))
        d.append("def copa_root(): return '{}/'".format(cwd))
    else:
        with open(f, encoding='utf-8') as h:
            for z in h:
                if re.search('#!.*python3',z):
                    if len(python_path) > 0:
                        d.append("{}\n".format(python_path))
                    d.append("import sys\n")
                    d.append("sys.path.insert(0,'{}/')\n".format(cwd))
                elif re.search('^import sys',z):
                    continue
                else:
                    d.append(z)
    o = open(f,mode='w',encoding='utf-8')
    o.write(''.join(d))
    o.close()
    return

# insert cwd in all python scripts
cwd = os.getcwd()
for f in file_collector(cwd,'py'):
    if re.search('\/make\.py',f): continue
    print('adjusting {}'.format(f))
    insert_cwd(f)
    
print('done!')
            

#print(pyf)

#sys.path.insert(0, '/homes/reichelu/repos/repo_pl/src/copasul_py/')
