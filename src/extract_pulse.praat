###################################################
## pulse extraction ###############################
###################################################

# directory-wise pulse extraction from mono audio files
# output: 1 column table <timestamp>
# call:
# > extract_pulse.praat minfrequ maxfrequ inputDir outputDir \
#                       audioExtension pulseFileExtension
# example:
# > extract_pulse.praat 50 400 /my/Audio/Dir/ /my/Pulse/Dir/ wav pulse

form input
     real minfrequ
     real maxfrequ
     word dir
     word diro
     word aud_ext
     word pulse_ext
endform

# list of audio files
Create Strings as file list... list 'dir$'/*.'aud_ext$'
# last element's index
nos = Get number of strings
for ios to nos
  select Strings list
  fileName$ = Get string... ios
  # generate output file name: e.g. dir$/a.wav --> fo$/a.pulse
  call get_of
  # delete already existing output files
  filedelete 'fo$'
  echo 'fileName$' --> 'fo$'

  # read audio
  Read from file... 'dir$'/'fileName$'
  mySound$ = selected$ ("Sound") 

  # pulse
  call get_pulse
endfor


## outputs pulse time stamps, one per row ###########################
procedure get_pulse

  # pitch
  To Pitch (cc)... 0.01 minfrequ 15 no 0.03 0.45 0.01 0.35 0.14 maxfrequ

  select Sound 'mySound$'
  plus Pitch 'mySound$'

  To PointProcess (cc)
  #Save as short text file... 'fo$'

  nop = Get number of points
  for iop to nop
    time = Get time from index... iop
    # time value to string
    times$ = fixed$('time', 8)
    fileappend 'fo$' 'times$''newline$'
  endfor
endproc

## gets name of output file (exchanges extension to ext$) ###########
procedure get_of
  fstem$=left$(fileName$,rindex(fileName$,"."))
  fo$="'diro$'/'fstem$''pulse_ext$'"
endproc


