import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import rcParams
rcParams['figure.figsize'] = 9,4

fontP = FontProperties()
fontP.set_size('xx-small')
#legend([plot1], "title", prop = fontP)


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)
plt.clf()
models=('NIN/','ResNet/')
modelnames=('NIN','ResNet')
styles=('k','k','r','r','b','b','g','g','c','c','m','m','y','y','k','k','r.','r','b.','b','g.','g','c.','c')
count = 0
#plt.figure(num=None, figsize=(8, 6), dpi=80)
plt.figure(1)
plt.yticks(np.arange(0, 101, 5.0))
for mod in models:
  modname = modelnames[count/2]
  testdata = np.genfromtxt(mod + 'test.log')
  traindata = np.genfromtxt(mod + 'train.log')
  numiters = max(len(testdata),len(traindata))
  t1 = np.arange(0, len(testdata), 1)
  t2 = np.arange(0, len(traindata), 1)
  plt.plot(t1, testdata, styles[count], label=modname+'/test',linewidth=2)
  plt.plot(t2, traindata, styles[count+1], label=modname+'/train', linewidth=1)
  count+=2
plt.legend(loc=8,ncol=4,prop=fontP)
#loc='upper center', bbox_to_anchor=(0.5, 1.05)
plt.savefig("out.png", bbox_inches="tight")
