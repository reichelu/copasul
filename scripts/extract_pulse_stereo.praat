###################################################
## pulse extraction ###############################
###################################################

# directory-wise pulse extraction from stereo audio files
# output: 2 column table <timestamps ch. 1> <timestamps ch. 2>
#   right -1 padding in shorter column
# call:
# > extract_pulse_stereo.praat minfrequ maxfrequ inputDir outputDir \
#                       audioExtension pulseFileExtension
# example:
# > extract_pulse_stereo.praat 50 400 myAudioDir/ myPulseDir/ wav pulse

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
  aud = Read from file... 'dir$'/'fileName$'
  mySound$ = selected$ ("Sound") 

  # pulse
  call get_pulse_stereo
  call clean_up
endfor


## outputs pulse time stamps, one per row ###########################
procedure get_pulse_stereo

  ## channel 1
  Extract one channel... 1
  myCh1$ = selected$ ("Sound")
  To Pitch (cc)... 0.01 minfrequ 15 no 0.03 0.45 0.01 0.35 0.14 maxfrequ
  
  select Sound 'myCh1$'
  plus Pitch 'myCh1$'

  To PointProcess (cc)
  myPnt1$ = selected$ ("PointProcess")
  nop1 = Get number of points

  ## channel 2
  select Sound 'mySound$'
  Extract one channel... 2
  myCh2$ = selected$ ("Sound")
  To Pitch (cc)... 0.01 minfrequ 15 no 0.03 0.45 0.01 0.35 0.14 maxfrequ
  
  select Sound 'myCh2$'
  plus Pitch 'myCh2$'

  To PointProcess (cc)
  myPnt2$ = selected$ ("PointProcess")
  nop2 = Get number of points

  nop = nop1
  if nop2>nop1
    nop=nop1
  endif

  for iop to nop
    if iop <= nop1
      select PointProcess 'myPnt1$'
      time = Get time from index... iop
      time1$ = fixed$('time',8)
    else
      time1$ = fixed$(-1,0)
    endif
    
    if iop <= nop2
      select PointProcess 'myPnt2$'
      time = Get time from index... iop
      time2$ = fixed$('time',8)
    else
      time2$ = fixed$(-1,0)
    endif

    fileappend 'fo$' 'time1$' 'time2$''newline$'
  endfor
endproc

## gets name of output file (exchanges extension to ext$) ###########
procedure get_of
  fstem$=left$(fileName$,rindex(fileName$,"."))
  fo$="'diro$'/'fstem$''pulse_ext$'"
endproc


#New Praat script
#Read from file... /home/reichelu/data/hunGameCorpus/minimalExample/wav/gr1_game1_bl3.wav
#Extract one channel... 1
#To Pitch (cc)... 0 75 15 no 0.03 0.45 0.01 0.35 0.14 600
#select Sound gr1_game1_bl3_ch1
#plus Pitch gr1_game1_bl3_ch1
#To PointProcess (cc)
