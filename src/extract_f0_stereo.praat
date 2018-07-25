################################################
## f0 extraction ###############################
################################################

# directory-wise f0 extraction from stereo audio files
# output: 3 column table <frame timestamp> <f0 value ch. 1> <f0 value ch. 2>
# call:
# > extract_f0_steret.praat framelength minfrequ maxfrequ inputDir outputDir \
#   audioExtension f0FileExtension
# example:
# > extract_f0_stereo.praat 0.01 50 400 /my/Audio/Dir/ /my/F0/Dir/ wav f0
# output file:
#   /my/F0/Dir/inputStem.f0 

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

## precision of time output
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
  # pitch objects
  call get_pitchObj
  # output
  call get_f0
  call clean_up
endfor


## removing current objects #########################################
procedure clean_up
  select aud
  Remove
  select channel1
  Remove
  select channel2
  Remove
  select pitch1
  Remove
  select pitch2
  Remove
endproc

## extract pitch object for each channel ############################

procedure get_pitchObj
  # read audio into sound object
  aud = Read from file... 'dir$'/'fileName$'
  channel1 = Extract one channel... 1
  pitch1 = To Pitch... framelength minfrequ maxfrequ
  select aud
  channel2 = Extract one channel... 2
  pitch2 = To Pitch... framelength minfrequ maxfrequ
endproc

## gets name of output file (exchanges extension to ext$) ###########

procedure get_of
  fstem$=left$(fileName$,rindex(fileName$,"."))
  fo$="'diro$'/'fstem$''f0_ext$'"
endproc


## extracts and outputs f0 contour ##################################

procedure get_f0
  # frame count
  select pitch1
  nof = Get number of frames

  for iof to nof
    # time of current frame
    time = Get time from frame... iof
    # time value to string
    times$ = fixed$('time', 'timeprec')
    # get f0 values of current frame
    select pitch1
    y1 = Get value in frame... iof Hertz
    select pitch2
    y2 = Get value in frame... iof Hertz
    if y1 = undefined
      y1 = 0
    endif
    if y2 = undefined
      y2 = 0
    endif
    fileappend 'fo$' 'times$' 'y1' 'y2''newline$'
  endfor
endproc
