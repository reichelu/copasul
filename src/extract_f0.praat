################################################
## f0 extraction ###############################
################################################

# directory-wise f0 extraction from audio files
# output: 2 column table <frame timestamp> <f0 value>
# call:
# > extract_f0.praat framelength minfrequ maxfrequ inputDir outputDir \
#   audioExtension f0FileExtension
# example:
# > extract_f0.praat 0.01 50 400 /my/Audio/Dir/ /my/F0/Dir/ wav f0

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

## framesize for f0 extraction:
#framelength=0.01
## min and max allowed f0 (below or higher --> 0)
#minfrequ=50
#maxfrequ=500
## precision of time output
timeprec=round(abs(log10(framelength)))
## Input/Output-Directories
#dir$="/homes/reichelu/data/inton_hu2012/sound"
# Output-Directory (anzupassen)
#diro$="/homes/reichelu/data/inton_hu2012/sound"
# Extension der Signal-Files (Input)
#aud_ext$="wav"
# Extension der f0-Files (Output)
#f0_ext$="f0"


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
  Read from file... 'dir$'/'fileName$'
  call get_f0
endfor


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
