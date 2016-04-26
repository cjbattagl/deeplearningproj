# casey battaglino
# add model-generated labels to video

import numpy as np
from numpy import loadtxt
import subprocess

ffmpeg = "ffmpeg"
#inVid = "./ann.avi"
#outVid = "./out.avi"


rnn = np.genfromtxt('labels_rnn.txt', names=True, delimiter=' ', dtype=None)
tcnn = np.genfromtxt('labels_tcnn.txt', names=True, delimiter=' ', dtype=None)

numvids = min(rnn.size,tcnn.size)
for i in range(0,numvids):
  pred_action = rnn[i][1]  #predicted action
  real_action = rnn[i][0].split('/')[0]  #real action
  if (pred_action == real_action):
    rnncolor = 'green'
  else:
    rnncolor = 'red'
  real_action_tcnn = tcnn[i][0].split('/')[0]
  pred_action_tcnn = tcnn[i][1]
  if real_action_tcnn == pred_action_tcnn:
    tcnncolor = 'green'
  else:
    tcnncolor = 'red'

  inVid = ('./UCF-101/' + rnn[i][0])
  outVid = ('./UCF-101/' + rnn[i][0].split('.')[0] + '_ann.avi')
  #mypred_action = "lrcn pred " + pred_action
  #mypred_action_tcnn = "tcnn pred: " + pred_action_tcnn
  #ffmpeg -i test_in.avi -vf "[in]drawtext=fontsize=20:fontcolor=White:fontfile='/Windows/Fonts/arial.ttf':text='onLine1':x=(w)/2:y=(h)/2, drawtext=fontsize=20:fontcolor=White:fontfile='/Windows/Fonts/arial.ttf':text='onLine2':x=(w)/2:y=((h)/2)+25, drawtext=fontsize=20:fontcolor=White:fontfile='/Windows/Fonts/arial.ttf':text='onLine3':x=(w)/2:y=((h)/2)+50[out]" -y test_out.avi
  #  subprocess.Popen(ffmpeg + " -y -i " + inVid + ''' -vf "format=yuv444p, drawbox=y=ih/PHI:color=black@0.4:width=iw:height=48:t=max, drawtext=fontfile=/Windows/Fonts/arial.ttf:text='ApplyMakeup':fontcolor=white:fontsize=24:x=(w-tw)/2:y=(h/PHI)+th,format=yuv420p" ''' + outVid , shell=True) 
#  subprocess.Popen(ffmpeg + opts + inVid + ''' -vf "format=yuv444p, drawbox=y=ih/PHI:color=black@0.4:width=iw:height=48:t=max, [in]drawtext=fontfile=/Windows/Fonts/arial.ttf:text='''' + pred_action + '''':fontcolor=''' + rnncolor + ''':fontsize=24:x=(w-tw)/2:y=(h/PHI)+th,drawtext=fontfile=/Windows/Fonts/arial.ttf:text='''' + pred_action_tcnn + '''':fontcolor=''' + tcnncolor + ''':fontsize=24:x=(w-tw)/2:y=(h/PHI)+th[out],format=yuv420p" ''' + outVid , shell=True)
  opts = " -y -i "
  firsttext = ''' -vf "[in]format=yuv444p, drawbox=y=ih/PHI:color=black@0.4:width=iw:height=48:t=max,drawtext=fontfile=/Windows/Fonts/arial.ttf:text=''' + pred_action + ''':fontcolor=''' + rnncolor + ''':fontsize=20:x=(w-tw)/2:y=(h/PHI)'''
  ftpt2 = ''':fontsize=24:x=(w-tw)/2:y=(h/PHI)'''
  secondtext = ''',drawtext=fontfile=/Windows/Fonts/arial.ttf:text=''' + pred_action_tcnn + ''':fontcolor=''' + tcnncolor + ''':fontsize=20:x=(w-tw)/2:y=(h/PHI)+th,format=yuv420p[out]" '''
  subprocess.Popen(ffmpeg + opts + inVid + firsttext + secondtext + outVid , shell=True)  
