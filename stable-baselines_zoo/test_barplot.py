import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# dfdict = {'ID': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
#       'quarter': ['2015 2Q', '2016 1Q', '2015 2Q', '2016 1Q', '2015 2Q',
#                   '2016 1Q', '2015 2Q', '2016 1Q'],
#       'Percent': [0.851789, 0.333333, 0.355240, 0.167224, 1.533220,
#                   0.333333, 0.170358, 0.000000],
#       'AgrCoullLower': [ 0.378046, 0.057962,  0.061850, -0.027515,
#                          0.866025, 0.057962, -0.028012, -0.092614],
#       'AgrCoullUpper': [1.776511, 1.054612, 1.123492, 0.810851,
#                         2.645141, 1.054612, 0.825960, 0.541513]}
# df = pd.DataFrame(dfdict)

# print(df)

# errLo = df.pivot(index='ID', columns='quarter', values='AgrCoullLower')

# print(errLo)

# df.pivot(index='ID', columns='quarter', values='Percent').plot(kind='bar', yerr=errLo)

# plt.show()



df = pd.DataFrame([[4,6,1,3], [5,7,5,2]], columns = ['mean1', 'mean2', 'std1', 'std2'], index=['A', 'B'])
print(df)

df[['mean1', 'mean2']].plot(kind='bar', yerr=df[['std1', 'std2']].values.T, alpha = 0.5, error_kw=dict(ecolor='k'))
plt.show()





means   = [26.82,26.4,61.17,61.55]           # Mean Data 
stds    = [(0,0,0,0), [4.59,4.39,4.37,4.38]] # Standard deviation Data
peakval = ['26.82','26.4','61.17','61.55']   # String array of means

ind = np.arange(len(means))
width = 0.35
colours = ['red','blue','green','yellow']

plt.figure()
plt.bar(ind, means, width, color=colours, align='center', yerr=stds, ecolor='k')
plt.xticks(ind,('Young Male','Young Female','Elderly Male','Elderly Female'))

plt.show()
