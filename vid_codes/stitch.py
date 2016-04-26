# casey battaglino
# create a list of the files to stitch together for ffmpeg
import subprocess
out = subprocess.check_output(["/usr/bin/find", "./UCF-101", "-name",  "*ann.avi"]).splitlines()
with open('stitches.txt', 'a') as the_file:
  for i in range(0,len(out)):
    writestr = "file '"+ out[i] + "'\n"
    the_file.write(writestr)
