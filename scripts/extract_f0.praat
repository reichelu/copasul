################################################
## f0 extraction ###############################
################################################

# directory-wise f0 extraction from audio files
# output: 2 column table <frame timestamp> <f0 value>
# call:
# > extract_f0.praat framelength minfrequ maxfrequ inputDir outputDir \
#                    audioExtension f0FileExtension
# example:
# > extract_f0.praat 0.01 50 400 myAudioDir/ myF0Dir/ wav f0

form input
     real framelength
     real minfrequ
     real maxfrequ
     word dir
     word diro
     word aud_ext
     word f0_ext
endform

## settings ###########################################################

# precision of time output
timeprec=round(abs(log10(framelength)))

## main ###############################################################

# list of audio files
Create Strings as file list... list 'dir$'/*.'aud_ext$'
# last element's index
nos = Get number of strings
for ios to nos
  select Strings list
  fileName$ = Get string... ios
  # generate output file name: e.g. dir$/a.wav --> fo$/a.f0
  call get_of
  # delete already existing output files
  filedelete 'fo$'
  echo 'fileName$' --> 'fo$'
  # read audio into sound object
  aobj = Read from file... 'dir$'/'fileName$'
  call get_f0
  call clean_up
endfor

## removing current objects #########################################
procedure clean_up
  select aobj
  Remove
  select pobj
  Remove
endproc

## gets name of output file (exchanges extension to ext$) ###########
procedure get_of
  fstem$=left$(fileName$,rindex(fileName$,"."))
  fo$="'diro$'/'fstem$''f0_ext$'"
endproc

## extracts and outputs f0 contour ##################################
procedure get_f0
  # from sound object to pitch object pobj
  pobj = To Pitch... framelength minfrequ maxfrequ
  # frame count
  nof = Get number of frames
  for iof to nof
    # time of current frame
    time = Get time from frame... iof
    # time value to string
    times$ = fixed$('time', 'timeprec')
    # get f0 value of current frame
    pitch = Get value in frame... iof Hertz
    # if no f0 value (pause, supposed measurement error)
    if pitch = undefined
      # append row <time> 0 to fo$
      fileappend 'fo$' 'times$'  0'newline$'
    else
      # append row <time> <pitch> to fo$
      fileappend 'fo$' 'times$'  'pitch''newline$'
    endif
  endfor
endproc
