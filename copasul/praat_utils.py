import audeer
import parselmouth
import numpy as np
import os
import sys
from tqdm import tqdm


class PraatPulseExtractor(object):

    def __init__(
            self,
            hopsize: float = 0.01,
            minfreq: float = 75.0,
            maxfreq: float = 600,
            max_number_of_candidates: int = 15,
            silence_threshold: float = 0.03,
            voicing_threshold: float = 0.45,
            octave_cost: float = 0.01,
            octave_jump_cost: float = 0.35,
            voiced_unvoiced_cost: float = 0.14
    ):
        r"""Praat-Parselmouth pulse extractor

        Args:
            hopsize: hop size
            minfreq: minimum allowed frequency
            maxfreq: maximum allowed frequency
        """

        super().__init__()
        self.hopsize = hopsize
        self.minfreq = minfreq
        self.maxfreq = maxfreq
        self.max_number_of_candidates = max_number_of_candidates
        self.silence_threshold = silence_threshold
        self.voicing_threshold = voicing_threshold
        self.octave_cost = octave_cost
        self.octave_jump_cost = octave_jump_cost
        self.voiced_unvoiced_cost = voiced_unvoiced_cost

    def process_mono_files(
            self,
            input_dir: str,
            output_dir: str,
            input_ext: str = "wav",
            output_ext: str = "pulse" 
    ) -> None:

        f"""process mono audio files. Outputs per audio file
        a pulse plain text file with pulse time stamps in seconds,
        one per row

        Args:
            input_dir: name of input directory
            output_dir: name of output directory
            input_ext: audio file extension
            output_ext: f0 file extension
        
        """

        _ = audeer.mkdir(output_dir)
        ff = os.listdir(input_dir)
        for i in tqdm(range(len(ff)), desc="pulse extraction"):

            if not ff[i].endswith(input_ext):
                continue
            
            sound = parselmouth.Sound(os.path.join(input_dir, ff[i]))
            pitch = sound.to_pitch_cc(
                time_step=self.hopsize,
                pitch_floor=self.minfreq,
                max_number_of_candidates=self.max_number_of_candidates,
                silence_threshold=self.silence_threshold,
                voicing_threshold=self.voicing_threshold,
                octave_cost=self.octave_cost,
                octave_jump_cost=self.octave_jump_cost,
                voiced_unvoiced_cost=self.voiced_unvoiced_cost,
                pitch_ceiling=self.maxfreq
            )
            pulses = parselmouth.praat.call(
                [sound, pitch], "To PointProcess (cc)")
            n_pulses = parselmouth.praat.call(
                pulses, "Get number of points")
            with open(os.path.join(
                    output_dir, os.path.splitext(
                        ff[i])[0] + "." + output_ext), 'w') as f:
                for j in range(1, n_pulses + 1):
                    t = parselmouth.praat.call(
                        pulses, "Get time from index", j)
                    f.write(f"{round(t, 8)}\n")
                    
    def process_stereo_files(
            self,
            input_dir: str,
            output_dir: str,
            input_ext: str = "wav",
            output_ext: str = "pulse" 
    ) -> None:

        f"""process stereo audio files. Outputs per audio file
        a pulse plain text file with pulse time stamps for the left and
        right channel in seconds

        Args:
            input_dir: name of input directory
            output_dir: name of output directory
            input_ext: audio file extension
            output_ext: f0 file extension
        
        """
        
        _ = audeer.mkdir(output_dir)
        ff = os.listdir(input_dir)
        for i in tqdm(range(len(ff)), desc="pulse extraction"):

            if not ff[i].endswith(input_ext):
                continue
            
            sound = parselmouth.Sound(os.path.join(input_dir, ff[i]))
            sound1 = sound.extract_left_channel()
            sound2 = sound.extract_right_channel()
            pitch1 = sound1.to_pitch_cc(
                time_step=self.hopsize,
                pitch_floor=self.minfreq,
                max_number_of_candidates=self.max_number_of_candidates,
                silence_threshold=self.silence_threshold,
                voicing_threshold=self.voicing_threshold,
                octave_cost=self.octave_cost,
                octave_jump_cost=self.octave_jump_cost,
                voiced_unvoiced_cost=self.voiced_unvoiced_cost,
                pitch_ceiling=self.maxfreq
            )
            pitch2 = sound2.to_pitch_cc(
                time_step=self.hopsize,
                pitch_floor=self.minfreq,
                max_number_of_candidates=self.max_number_of_candidates,
                silence_threshold=self.silence_threshold,
                voicing_threshold=self.voicing_threshold,
                octave_cost=self.octave_cost,
                octave_jump_cost=self.octave_jump_cost,
                voiced_unvoiced_cost=self.voiced_unvoiced_cost,
                pitch_ceiling=self.maxfreq)
            
            pulses1 = parselmouth.praat.call(
                [sound1, pitch1], "To PointProcess (cc)")
            pulses2 = parselmouth.praat.call(
                [sound2, pitch2], "To PointProcess (cc)")
            n_pulses1 = parselmouth.praat.call(
                pulses1, "Get number of points")
            n_pulses2 = parselmouth.praat.call(
                pulses2, "Get number of points")
            n_max = max(n_pulses1, n_pulses2)
            with open(os.path.join(
                    output_dir, os.path.splitext(
                        ff[i])[0] + "." + output_ext), 'w') as f:
                for j in range(1, n_max + 1):
                    t1, t2 = 0.0, 0.0
                    if j <= n_pulses1:    
                        t1 = parselmouth.praat.call(
                            pulses1, "Get time from index", j)
                    if j <= n_pulses2:
                        t2 = parselmouth.praat.call(
                            pulses2, "Get time from index", j)
                    f.write(f"{round(t1, 8)} {round(t2, 8)}\n")
    

class PraatPitchExtractor(object):

    def __init__(
            self,
            hopsize: float = 0.01,
            minfreq: float = 75.0,
            maxfreq: float = 600
    ):

        r"""Praat-Parselmouth pitch extractor

        Args:
            hopsize: hop size
            minfreq: minimum allowed frequency
            maxfreq: maximum allowed frequency
        """

        super().__init__()
        self.hopsize = hopsize
        self.minfreq = minfreq
        self.maxfreq = maxfreq

        
    def process_mono_files(
            self,
            input_dir: str,
            output_dir: str,
            input_ext: str = "wav",
            output_ext: str = "f0"
    ) -> None:

        f"""process mono audio files. Outputs per audio file
        an f0 plain text file with time f0 pairs

        Args:
            input_dir: name of input directory
            output_dir: name of output directory
            input_ext: audio file extension
            output_ext: f0 file extension
        
        """

        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)
        _ = audeer.mkdir(output_dir)

        timeprec = round(abs(np.log10(self.hopsize)))
        ff = os.listdir(input_dir)
        for i in tqdm(range(len(ff)), desc="f0 extraction"):
            if not ff[i].endswith(input_ext):
                continue
            sound = parselmouth.Sound(os.path.join(input_dir, ff[i]))
            pitch = sound.to_pitch(
                time_step=self.hopsize,
                pitch_floor=self.minfreq,
                pitch_ceiling=self.maxfreq
            )
            times = pitch.xs()
            f0s = pitch.selected_array['frequency']
            with open(os.path.join(
                    output_dir,
                    os.path.splitext(ff[i])[0] + "." + output_ext), 'w') as f:
                for time, f0 in zip(times, f0s):
                    if np.isnan(f0):
                        f.write(f"{round(time, timeprec)} 0\n")
                    else:
                        f.write(f"{round(time, timeprec)} {f0}\n")

                        
    def process_stereo_files(
            self,
            input_dir: str,
            output_dir: str,
            input_ext: str = "wav",
            output_ext: str = "f0"
    ) -> None:

        f"""process mono audio files. Outputs per audio file
        an f0 plain text file with time f0-left f0-right triplets

        Args:
            input_dir: name of input directory
            output_dir: name of output directory
            input_ext: audio file extension
            output_ext: f0 file extension
        
        """

        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)
        _ = audeer.mkdir(output_dir)
        
        timeprec = round(abs(np.log10(self.hopsize)))
        
        ff = os.listdir(input_dir)
        for i in tqdm(range(len(ff)), desc="f0 extraction"):

            if not ff[i].endswith(input_ext):
                continue
            
            sound = parselmouth.Sound(os.path.join(input_dir, ff[i]))
            sound1 = sound.extract_left_channel()
            sound2 = sound.extract_right_channel()
            pitch1 = sound1.to_pitch(
                self.hopsize, self.minfreq, self.maxfreq
            )
            pitch2 = sound2.to_pitch(
                self.hopsize, self.minfreq, self.maxfreq
            )
            
            nof = pitch1.get_number_of_frames()
            f0_values = []
            for iof in range(1, nof+1):
                time = pitch1.get_time_from_frame_number(iof)
                y1 = pitch1.get_value_in_frame(iof)
                y2 = pitch2.get_value_in_frame(iof)
                y1 = 0 if y1 is None else y1
                y2 = 0 if y2 is None else y2
                f0_values.append((time, y1, y2))
                
            with open(os.path.join(
                    output_dir,
                    os.path.splitext(ff[i])[0] + "." + output_ext), 'w') as f:
                for time, y1, y2 in f0_values:
                    if np.isnan(y1):
                        y1 = 0.0
                    if np.isnan(y2):
                        y2 = 0.0
                    f.write(f"{round(time, timeprec)} {y1} {y2}\n")


