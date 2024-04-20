import os
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(cwd, '..')
sys.path.append(root)
from copasul import praat_utils

# Praat pitch extractor
piex = praat_utils.PraatPitchExtractor(
    hopsize=0.01, minfreq=75, maxfreq=600)

# process mono files
piex.process_mono_files(
    input_dir=os.path.join(root, "tests", "minex", "input"),
    output_dir=os.path.join(cwd, "cache_pitch"),
    input_ext="wav",
    output_ext="f0"
)

# in order to process stereo files, use:
# piex.process_stereo_files()

# Praat pulse extractor
puex = praat_utils.PraatPulseExtractor(
    hopsize=0.01, minfreq=75, maxfreq=400)

# process mono files
puex.process_mono_files(
    input_dir=os.path.join(root, "tests", "minex", "input"),
    output_dir=os.path.join(cwd, "cache_pulse"),
    input_ext="wav",
    output_ext="pulse"
)

# in order to process stereo files, use:
# puex.process_stereo_files()
