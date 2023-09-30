import os
import json
import copasul

# load minimal example config
pth = os.path.dirname(os.path.abspath(__file__))
with open("{}/../minex/config/minex.json".format(pth), "r") as h:
    opt = json.load(h)

# init Copasul() object
fex = copasul.Copasul()

# process data specified in config
copa = fex.process(config=opt)

# access output, e.g. local contour feature set
print("local contour features:")
print(copa["export"]["loc"])

