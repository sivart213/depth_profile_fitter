# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 12:36:55 2021.

@author: j2cle
"""

# %% import section
import numpy as np
import pandas as pd
import sims_fit_cls3 as sfc
import General_functions as gf
import warnings
import dill

warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings("ignore")


# %% Import data
mypath = "C:\\Users\\j2cle\\Work Docs\\Data\\Analysis\\SIMS\\"
figpath = 'C:\\Users\\j2cle\\Work Docs\\Data\\Analysis\\SIMS\\Fig_fits'
df_log = pd.read_excel(f'{mypath}/Sample Log for fitting.xlsx',
                       index_col=0, skiprows=1).dropna(axis=0, how='all')

if 0:
    df_log.drop(['R-60 SF ROI'], inplace=True)

# sample_all = {x: sfc.DataProfile(df_log.loc[x, :]) for x in df_log.index}

# %% Create the necesary objects
# obj = pd.DataFrame()
obj = {}
for x in range(18, 19): # 15 is all, 18 is 90
    sample = df_log.index[x].split(' ')[0]
    obj[sample] = {}
    # obj.loc[df_log.index[x].split(' ')[0], 'SIMS'] = sample_all[df_log.index[x]]
    obj[sample]['SIMS'] = sfc.DataProfile(df_log.iloc[x, :])

    obj[sample]['Prime'] = sfc.MatrixOps(obj[sample]['SIMS'],
                                         'FitProfile',
                                         xrange=['depth', 0, 75, 'index'],
                                         yrange=['depth', 0, 75, 'index'],
                                         size=75, min_range=1,
                                         log_form=False)

    obj[sample]['Prime'].error_calc(get_best=False,
                                    log_form=False,
                                    instr='none',
                                    use_sample_w=False,
                                    )

    obj[sample]['Ops'] = obj[sample]['Prime'].obj_operator

    obj[sample]['Eval'] = sfc.Analysis(obj[sample]['Prime'])

    obj[sample]['Plot'] = sfc.Plotter(obj[sample]['Prime'])


del x
# dill.dump_session('3_25_dill.pkl')
# dill.load_session('3_25_dill.pkl')


# %%
check = 'R-90'
label_num = check.split('-')[1]
res_init_df = obj[sample]['Eval'].family_df

# %% Generate error weight stats and plot
obj[check]['Prime'].error_calc(get_best=False,
                               log_form=True,
                               instr='none',
                               use_sample_w=True,
                               )


res_df = obj[sample]['Eval'].family_df

# stats_df = res_df.pivot_table(values='stats', columns='start_loc', index='index_range')
# error_df = res_df.pivot_table(values='error',columns='start_index',index='stop_index')
# diff_df = res_df.pivot_table(values='diff',columns='start_loc',index='index_range')

# err_matrix = obj['R-90'][ 'Ops'].matrix(info=['stop_loc', 'depth_range', 'error'])
# alt_matrix = unpickled.loc['R-90', 'Ops'].matrix(info=['start_index', 'stop_index', 'diff'])
# %% In index form to find points
obj[check]['Plot'].map_plot(name=str(label_num+': Error Evaluation'),  # log10(Data) & Weighted
                             info=['start_index', 'index_range', 'error'],
                             # matrix=error_df,
                             # conv=[1, 1e4],
                             # xscale='log',
                             # yscale='linear',
                             zscale='linlog',
                             xlimit=[0, 50],
                             ylimit=[0, 50],
                             # zlimit=[1e-2, 0.5],
                             zlimit=[5e-4, 0.01],
                             levels=5,
                             xname='Start index',  # u=\u03BC
                             yname='Range (# of data-points)',
                             zname='Error',
                             # zlimit=[1e-2, 1],
                             # ztick={'base':10},
                             ztick={'base': 10,
                                    'labelOnlyBase': False,
                                    'minor_thresholds': (2, 2e-5),
                                    'linthresh': 1},
                             corner_mask=True,
                             )

# %% In range form to show
obj[check]['Plot'].map_plot(name=str(label_num+': Error Evaluation'),  # log10(Data) & Weighted
                             info=['start_loc', 'depth_range', 'error'],
                             # matrix=error_df,
                             # conv=[1, 1e4],
                             # xscale='log',
                             # yscale='linear',
                             zscale='linlog',
                             xlimit=[0, 2.5],
                              ylimit=[0, 2.5],
                               zlimit=[5e-4, 0.01],
                             # zlimit=[1e-2, 0.5],
                             levels=5,
                             xname='Start location (\u03BCm)',  # u=\u03BC
                             yname='Range (\u03BCm)',
                             zname='Error',
                             # zlimit=[1e-2, 1],
                             # ztick={'base':10},
                             ztick={'base': 10,
                                    'labelOnlyBase': False,
                                    'minor_thresholds': (2, 2e-5),
                                    'linthresh': 1e-1},
                             corner_mask=True,
                             )
# %% Get Profile
std_vars = ['start_index', 'stop_index', 'index_range', 'start_loc',
            'stop_loc', 'diff', 'conc', 'error', 'stats']
pair_labels = ['start_index', 'index_range']
pairs=[(2, 22), (2, 26), (2, 30), (2, 36), (10, 14), (10, 18), (10, 22), (10, 28),
       (18, 10), (18, 16), (18, 25), (18, 39), (23, 14), (29, 17), (33, 15), (35, 23)]

focus = obj[sample]['Eval'].focus(pairs=pairs, pair_names=pair_labels, var=std_vars) # var=['depth','sims','pred' pairs=None, pair_names

# low_err = obj[sample]['Eval'].focus(pair_names=['start_index','error'])

obj[check]['Plot'].prof_plot(name=str(label_num+': Diffusivity'),
                             data_in=focus,
                                x='stop_loc',
                                y='diff',
                                xscale='linear',
                                yscale='log',
                                xlimit=[0,3],
                                ylimit=[1e-16,1e-14],
                                xname='Depth',
                                yname='Diffusivity',
                                hline=False,
                                hue='start_loc',
                                legend=True,
                                palette='kindlmann',
                                hue_norm=(0, 1.5),
                                )

# %%

prof = obj[sample]['Eval'].focus(pairs=pairs, pair_names=pair_labels, var=['data']) # var=['depth','sims','pred' pairs=None, pair_names
loc=1666
profile = obj[check]['Ops']._family[loc].prof.data
print(obj[check]['Ops']._family[loc].prof.start_index,obj[check]['Ops']._family[loc].prof.stop_index)

# %% evaluate results
lim = 1e-3
limited_df = res_df[res_df['error'] < lim]
limited_df = limited_df.drop_duplicates(subset='start_index',keep='last')

# res_df = obj[sample]['Eval'].family_df
# stats_df = res_df.pivot_table(values='stats',columns='start_loc',index='index_range')
# error_df = res_df.pivot_table(values='error',columns='start_loc',index='index_range')
# diff_df = res_df.pivot_table(values='diff',columns='start_loc',index='index_range')

# diff_list = list()
# for y in range(len(stats_df.columns)):
#     x = 2
#     while error_df.iloc[x,y] < 1e-3:
#         x += 1
#     diff_list.append([error_df.index[x],error_df.columns[y],diff_df.iloc[x,y],stats_df.iloc[x,y]])
#     y=0

# diff_array=np.array(diff_list)
# diff_df=pd.DataFrame(diff_list,columns=['index_range','start_loc','diff','error'])

# inflect = [0, 10, 16, 24, 199]
# man = [(inflect[x], inflect[x+1]) for x in reversed(range(len(inflect)-1))]

# plot_obj = sfc.Plotter()
# plot_obj.prof_plot(name='Diffusivity vs Depth',
#                    data_in=diff_df,
#                    x='start_index',
#                    y='diff',
#                    xscale='linear',
#                    yscale='log',
#                    xlimit=[0,40],
#                    ylimit=[1e-16,1e-12],
#                    xname='Depth',
#                    yname='Diffusivity',
#                    hline=False,
#                    hue='error',
#                    legend=False,
#                    # palette='kindlmann',
#                    hue_norm=(0, len(inflect)),
#                    )

# %% Review Matrix and Plot
res_df2 = obj[check]['Ops'].gen_df(stat_type='normal_test')
# stats_matrix = obj['R-90'][ 'Ops'].matrix(info=['start_index', 'index_range', 'stats.ks_test'])

obj[check]['Plot'].map_plot(name=str(label_num+': Error Evaluation'),  # log10(Data) & Weighted
                             info=['start_index', 'index_range', 'stats.shap_test'],
                             # matrix=error_df,
                             # conv=[1, 1e4],
                             # xscale='log',
                             # yscale='linear',
                             zscale='linlog',
                             xlimit=[0, 50],
                             ylimit=[0, 50],
                              zlimit=[2.5e-2, 1],
                             levels=5,
                             xname='Start location (\u03BCm)',  # u=\u03BC
                             yname='Range (\u03BCm)',
                             zname='P-value',
                             # zlimit=[1e-2, 1],
                             # ztick={'base':10},
                              ztick={'base': 10,
                                     'labelOnlyBase': False,
                                     'minor_thresholds': (2, 2e-5),
                                     'linthresh': 1},
                             corner_mask=True,
                             )
# %% Reisual analysis
# evaluation = obj['R-90']['Eval']
# check =obj[sample]['Ops']._family[loc]
# inflect = [1, 5, 24, 50, 199]
# man = [(inflect[x], inflect[x+1]) for x in reversed(range(len(inflect)-1))]
# evaluation.stitcher(obj['R-90']['SIMS'], man)
# res = evaluation.stitched_res

# plot_obj = sfc.Plotter(res)
# plot_obj.prof_plot(name='Residual Plot',
#                    x='depth',
#                    y='ESR',
#                    xscale='linear',
#                    yscale='linear',
#                    xlimit=None,
#                    ylimit=[-2.5,2.5],
#                    xname='Depth',
#                    yname='ESR',
#                    hline=True,
#                    hue='Range number',
#                    legend=False,
#                    # palette='kindlmann',
#                    hue_norm=(0, len(inflect)),
#                    )
