import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import cm

fontP = FontProperties()
fontP.set_size('xx-small')
#legend([plot1], "title", prop = fontP)


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)
plt.clf()
models=('NIN/1L/','NIN/2L/1D/h20-50-250/','NIN/2L/1D/h32-64-256/','NIN/2L/2D/h20-50-250/','ResNet/1L/','ResNet/2L/')
modelnames=('NIN-1L','NIN-2L-1D-20-50-250','NIN-2L-1D-32-64-256','NIN-2L-2D-20-50-250','ResNet-1L','ResNet-2L')
styles=('k','k','r','r','b','b','g','g','c','c','m','m','y','y','k','k','r.','r','b.','b','g.','g','c.','c')
count = 0
numcols = 6
color=iter(cm.rainbow(np.linspace(0,1,numcols)))
plt.yticks(np.arange(0, 101, 5.0))
for mod in models:
  c = next(color)
  modname = modelnames[count/2]
  testdata = np.genfromtxt(mod + 'test.log')
  traindata = np.genfromtxt(mod + 'train.log')
  numiters = max(len(testdata),len(traindata))
  t1 = np.arange(0, len(testdata), 1)
  t2 = np.arange(0, len(traindata), 1)
  plt.figure(1)
  plt.plot(t1, testdata, c=c, label=modname+'/test',linewidth=2)
  plt.plot(t2, traindata, c=c, label=modname+'/train', linewidth=1)
  count+=2
plt.legend(loc=8,ncol=4,prop=fontP)
#loc='upper center', bbox_to_anchor=(0.5, 1.05)
plt.savefig("out.png", bbox_inches="tight")
