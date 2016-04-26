#code for a single annotation
import subprocess

ffmpeg = "ffmpeg"
inVid = "./ann.avi"
outVid = "./out.avi"

subprocess.Popen(ffmpeg + " -y -i " + inVid + ''' -vf "format=yuv444p, drawbox=y=ih/PHI:color=black@0.4:width=iw:height=48:t=max, drawtext=fontfile=/Windows/Fonts/arial.ttf:text='ApplyMakeup':fontcolor=white:fontsize=24:x=(w-tw)/2:y=(h/PHI)+th,format=yuv420p" ''' + outVid , shell=True)
