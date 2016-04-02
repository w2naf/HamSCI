#!/usr/bin/env python
import sys
import os
import glob
from subprocess import call

outFile = 'movie.mp4'

try:
  os.remove(outFile)
except:
  pass

def cleanHouse():
  oldFiles = glob.glob('*-symlink*')
  if len(oldFiles) > 0:
    for file in oldFiles: os.remove(file)

cleanHouse()

if len(sys.argv) == 1:
  files = glob.glob('*.png')
else:
  files = sys.argv[1:-1]

files.sort()

inx=0
for ff in files:
  path,ext = os.path.splitext(ff)
  if ext.lower() != '.png':
    call("convert -verbose -density 400 +matte "+ff+" "+path+".png",shell=True)
    ff = path+'.png'
  newName = ('%06d' % inx) +'-symlink.png'
  inx = inx + 1
  os.symlink(ff,newName)
  print ff + ' --> ' + newName

#call("avconv -qscale 10 -r 10 -b 9600 -i %06d-symlink.png "+outFile,shell=True)
call("avconv -qscale 10 -r 10 -b 20000 -i %06d-symlink.png "+outFile,shell=True)
#call("avconv -mbd rd -flags +mv4+aic -trellis 2 -cmp 2 -subcmp 2 -g 300 -qscale 4 -pass 1 -strict experimental -i %06d-symlink.png "+outFile,shell=True)

cleanHouse()
