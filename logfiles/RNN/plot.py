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
models=('NIN/LSTM/Hidden_256/','NIN/LSTM/Hidden_512_256/','ResNet/LSTM/Hidden_1024/','ResNet/LSTM/Hidden_1024_512/','ResNet/LSTM/Hidden_1024_512_256/','ResNet/LSTM/Hidden_256/','ResNet/LSTM/Hidden_512/','VGG8/LSTM/Hidden_1024_256/','VGG8/LSTM/Hidden_1024_512_256/','VGG8/LSTM/Hidden_256/')
modelnames=('NIN-256','NIN-512-256','ResNet-1024','ResNet-1024-512','ResNet-1024-512-256','ResNet-256','ResNet-512','VGG8-1024-256','VGG8-1024-512-256','VGG8-256')
styles=('k','k','r','r','b','b','g','g','c','c','m','m','y','y','k','k','r','r','b','b','g','g','c','c')
count = 0

numcols = 10
color=iter(cm.rainbow(np.linspace(0,1,numcols)))

for mod in models:
  c=next(color)
  modname = modelnames[count/2]
  testdata = np.genfromtxt(mod + 'test.log')
  traindata = np.genfromtxt(mod + 'train.log')
  numiters = max(len(testdata),len(traindata))
  t1 = np.arange(0, len(testdata), 1)
  t2 = np.arange(0, len(traindata), 1)
  plt.figure(1)
  plt.plot(t1, testdata,c=c, label=modname+'/test',linewidth=2)
  plt.plot(t2, traindata,c=c, label=modname+'/train', linewidth=1)
  count+=2
plt.legend(loc=8,ncol=4,prop=fontP)
#loc='upper center', bbox_to_anchor=(0.5, 1.05)
plt.savefig("out.png", bbox_inches="tight")
