"""Visualize power line data against one of the context files"""

from __future__ import print_function    # (at top of module)
import sys
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from utils.utils import *
from config import config

sys.path.append("../")

#from utils.utils import *

#Define the aggregation interval (can be 1H, 1D, 1W)
interval = '1D'
interval = '1H'

def d_resample(df):
    if interval:
        return df.resample(interval).mean()
    else:
        return df

scale=lambda x: MinMaxScaler().fit_transform(x)

target = d_resample(pd.read_pickle(config.data_folder + '/target.pkl'))

evtf = d_resample(pd.read_pickle(config.features_folder + '/evtf_states.pkl')).fillna(method='ffill').apply(scale)
dmop = d_resample(pd.read_pickle(config.features_folder + '/dmop_count_1h.pkl')).fillna(method='ffill').apply(scale)
saaf = d_resample(pd.read_pickle(config.features_folder + '/saaf.pkl')).fillna(method='ffill').apply(scale)
ltdata = d_resample(pd.read_pickle(config.features_folder + '/ltdata.pkl')).fillna(method='ffill').apply(scale)
ftl = d_resample(pd.read_pickle(config.features_folder + '/ftl_states.pkl')).fillna(method='ffill').resample('1D').mean()



#Define which context variable should be visualized, and what keyword to be used for title
context=dmop
context_title="DMOP"




#Sort target columns by standard deviations
cols=config.target_cols
cols=sorted(cols, key=lambda col:target[col].std(), reverse=True)

print( "Standard deviation by power line:")
for col in cols:
    print( col,',', target[col].std())


#Calculate spearman rank correlation of context variables with top2 target variables
print( "Spearman correlation for top two lines")
print( "feature,", cols[0], ',', cols[1])
for c_col in context.columns:
    c1s=spearmanr(context[c_col], target[cols[0]][context.index])[0]
    c2s=spearmanr(context[c_col], target[cols[1]][context.index])[0]
    print( c_col, ',', c1s, ',', c2s)


dmop_nrow, dmop_ncol = dmop.shape
    
#Prepare the visualization
fig, axs = plt.subplots(3, 1, sharex=True,figsize=(13,13))

if context_title=="DMOP":
    col=cols[0]
    axs[0].plot(target[col].values, label='True Power ' + col + 'STD:' +str(target[col].std()))
    axs[0].legend()
    col=cols[1]
    axs[1].plot(target[col].values, label='True Power ' + col + 'STD:' +str(target[col].std()))
    axs[1].legend()
    axs[2].imshow(dmop.values.T, aspect='auto', interpolation='nearest')
    if interval == '1D':
        axs[2].set_xticks(range(0,dmop_nrow,30))#np.linspace(0.5,dmopsorted_sub[i].columns.size+0.5,1.0))
        axs[2].set_xticklabels(list(dmop.index[0::(30)].strftime('%Y-%m-%d')),rotation='vertical')
    elif(interval == '1H'):
        axs[2].set_xticks(range(0,dmop_nrow,24))#np.linspace(0.5,dmopsorted_sub[i].columns.size+0.5,1.0))
        axs[2].set_xticklabels(list(dmop.index[0::(24)].strftime('%Y-%m-%d')),rotation='vertical')
    axs[2].set_yticks(np.linspace(0.5,dmop_ncol-0.5,dmop_ncol))#np.linspace(0.5,dmopsorted_sub[i].columns.size+0.5,1.0))
    axs[2].set_yticklabels(list(dmop.columns))

    
else:

    col=cols[0]
    axs[0].plot(target.index, target[col].values, label='True Power ' + col + 'STD:' +str(target[col].std()))
    axs[0].legend()
    col=cols[1]
    axs[1].plot(target.index, target[col].values, label='True Power ' + col + 'STD:' +str(target[col].std()))
    axs[1].legend()
    lines=axs[2].plot(context.index, context)
    box = axs[2].get_position()
    #axs[2].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axs[2].legend(lines, context.columns,loc='upper center', bbox_to_anchor=(0.5, 1.1),
                      ncol=5, fancybox=True, shadow=True)

#axs[5].plot(ftl.index, ftl, label='Saaf')

    # axs[2].imshow(saaf.values.T, aspect='auto')
    # plt.yticks(np.arange(0.5, len(saaf.index), 1), saaf.index)
    # plt.xticks(np.arange(0.5, len(saaf.columns), 1), saaf.columns)
plt.show()
