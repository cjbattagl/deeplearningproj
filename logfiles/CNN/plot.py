import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)
plt.clf()
models=('NIN/','ResNet/')
styles=('k--','k','r--','r')
count = 0
for mod in models:
  testdata = np.genfromtxt(mod + 'test.log')
  traindata = np.genfromtxt(mod + 'train.log')
  numiters = max(len(testdata),len(traindata))
  t1 = np.arange(0, len(testdata), 1)
  t2 = np.arange(0, len(traindata), 1)
  plt.figure(1)
  plt.plot(t1, testdata, styles[count], label=mod+'test',linewidth=1)
  plt.plot(t2, traindata, styles[count+1], label=mod+'train', linewidth=3)
  count+=2
plt.legend()
plt.savefig("out.png", bbox_inches="tight")
