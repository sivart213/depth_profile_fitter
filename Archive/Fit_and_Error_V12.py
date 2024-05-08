# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 12:36:55 2021.

@author: j2cle
"""

# %% import section
import numpy as np
import pandas as pd
import sims_fit_cls7 as sfc
import General_functions as gf
import Units_Primary3 as up
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
for x in range(10,11): # 0-4, tset, 0=90, 4-6 eset, 6-10 local, 10-14 init EVA, 15-end MIMS
    sample = df_log.index[x].split(' ')[0]
    obj[sample] = {}
    # obj.loc[df_log.index[x].split(' ')[0], 'SIMS'] = sample_all[df_log.index[x]]
    obj[sample]['SIMS'] = sfc.DataProfile(df_log.iloc[x, :])

    obj[sample]['Prime'] = sfc.MatrixOps(obj[sample]['SIMS'],
                                         'FitProfile',
                                         # xrange=['depth', 0, 75, 'index'],
                                         # yrange=['depth', 0, 75, 'index'],
                                         # size=75, min_range=1,
                                         xrange=['depth', 0, up.Length(6,'um').cm, 'lin'],
                                         yrange=['depth', 0, up.Length(6,'um').cm, 'lin'],
                                         size=75, min_range=1,
                                         diff_pred=pd.Series((-18,-15,-12),index=('low','mid','high')),
                                         log_form=False)

    obj[sample]['Prime'].error_calc(get_best=False,
                                    log_form=True,
                                    instr='none',
                                    use_sample_w=True,
                                    )

    obj[sample]['Ops'] = obj[sample]['Prime'].obj_operator

    obj[sample]['Plot'] = sfc.Plotter(obj[sample]['Prime'])

del x
del sample
# dill.dump_session('3_25_dill.pkl')
# dill.load_session('3_25_dill.pkl')


# %%
check = 'A-75'
label_num = check.split('-')[1]
res_init_df = obj[check]['Plot'].family_df

# # %% Regenerate Error
# obj[check]['Prime'].error_calc(get_best=False,
#                                 log_form=True,
#                                 instr='none',
#                                 use_sample_w=True,
#                                 )
# %% In range form to show
obj[check]['Plot'].map_plot(name=str(check+': Error Evaluation'),  # log10(Data) & Weighted
                             info=['start_loc', 'depth_range', 'error'],
                             zscale='linlog',
                             xlimit=[0, 6],
                             ylimit=[0, 6],
                             zlimit=[5e-4, 1],
                             levels=10,
                             xname='Start location (\u03BCm)',  # u=\u03BC
                             yname='Range (\u03BCm)',
                             zname='Error',
                             ztick={'base': 10,
                                    'labelOnlyBase': False,
                                    'minor_thresholds': (2, 2e-5),
                                    'linthresh': 100},
                             corner_mask=True,
                             )
obj[check]['Plot'].map_plot(name=str(check+': Coupled Diffusivity'),  # log10(Data) & Weighted
                             info=['start_loc', 'depth_range', 'diff'],
                             zscale='linlog',
                             xlimit=[0, 6],
                             ylimit=[0, 6],
                             # zlimit=[5e-4, 1],
                             levels=10,
                             xname='Start location (\u03BCm)',  # u=\u03BC
                             yname='Range (\u03BCm)',
                             zname='Diffusivity (cm2/s)',
                             ztick={'base': 10,
                                    'labelOnlyBase': False,
                                    'minor_thresholds': (2, 2e-5),
                                    'linthresh': 100},
                             corner_mask=True,
                             )
obj[check]['Plot'].map_plot(name=str(check+': Surface Conc'),  # log10(Data) & Weighted
                             info=['start_loc', 'depth_range', 'conc'],
                             zscale='linlog',
                             xlimit=[0, 6],
                             ylimit=[0, 6],
                             # zlimit=[5e-4, 1],
                             levels=10,
                             xname='Start location (\u03BCm)',  # u=\u03BC
                             yname='Range (\u03BCm)',
                             zname='Concentration (atoms/cm3)',
                             ztick={'base': 10,
                                    'labelOnlyBase': False,
                                    'minor_thresholds': (2, 2e-5),
                                    'linthresh': 1e21},
                             corner_mask=True,
                             )


# %% Peak generator
topo_pairs, mins_pairs, mins_maxs_df, mins_limits_df = obj[check]['Plot'].peaks_solver(
    num_of_peaks=5,
    stop_max=1.5, # 2.5 for other tof
    start_max=1, # 2 for other tof
    start_min=2, # 2 for other tof
    overlap=True,
    )

focus_df_auto = pd.DataFrame(columns=['count',
                                      'start',
                                      'min index',
                                      'stop',
                                      'start loc',
                                      'min loc',
                                      'stop loc',
                                      'error',
                                      'error std',
                                      'diff',
                                      'diff std',
                                      'conc',
                                      'conc std',
                                      ], index=mins_limits_df.index, dtype=float)


focus_df_auto['min index'] = mins_limits_df['start_index']
focus_df_auto['min loc'] = [obj[check]['Plot'].depth[x] for x in focus_df_auto['min index']]
focus_df_auto = focus_df_auto.fillna(0)
focus_df_auto = sfc.peak_cycles(obj[check]['Plot'], focus_df_auto, min_range=3)
focus_df = focus_df_auto.copy()

# %% Peak analyzer
peak=0
focus_stats, focii = obj[check]['Plot'].peaks(
    peak=peak,
    min_range=3,
    start=3,
    stop=22,
    full_stop=75,
    pair_set='all',
    )

focus_df.loc[peak, 'count'] = focus_stats.loc['count', 'conc']
focus_df.loc[peak, 'start'] = focii.index.get_level_values(0).min()
focus_df.loc[peak, 'stop'] = focii['stop_index'].max()
focus_df.loc[peak, 'start loc'] = focus_stats.loc['min', 'start_loc']
focus_df.loc[peak, 'stop loc'] = focus_stats.loc['max', 'stop_loc']
focus_df.loc[peak, 'error'] = focus_stats.loc['mean', 'error']
focus_df.loc[peak, 'error std'] = focus_stats.loc['std', 'error']
focus_df.loc[peak, 'diff'] = focus_stats.loc['mean', 'diff']
focus_df.loc[peak, 'diff std'] = focus_stats.loc['std', 'diff']
focus_df.loc[peak, 'conc'] = focus_stats.loc['mean', 'conc']
focus_df.loc[peak, 'conc std'] = focus_stats.loc['std', 'conc']
test=focii.reset_index()
focus_df_full = sfc.peak_cycles(obj[check]['Plot'], focus_df.copy(), min_range=2, stop=40, full_stop=40)

#%%
peak=0
focus_stats, focii = obj[check]['Plot'].peaks(
    peak=peak,
    min_start=0,
    max_start=21,
    min_range=4,
    peak_range=15,
    max_range=15,
    # pair_set='all',
    # old_range=True,
    )
focus_df.loc[peak, 'count'] = focus_stats.loc['count', 'conc']
focus_df.loc[peak, 'start'] = focii.index.get_level_values(0).min()
focus_df.loc[peak, 'stop'] = focii['stop_index'].max()
focus_df.loc[peak, 'start loc'] = focus_stats.loc['min', 'start_loc']
focus_df.loc[peak, 'stop loc'] = focus_stats.loc['max', 'stop_loc']
focus_df.loc[peak, 'error'] = focus_stats.loc['mean', 'error']
focus_df.loc[peak, 'error std'] = focus_stats.loc['std', 'error']
focus_df.loc[peak, 'diff'] = focus_stats.loc['mean', 'diff']
focus_df.loc[peak, 'diff std'] = focus_stats.loc['std', 'diff']
focus_df.loc[peak, 'conc'] = focus_stats.loc['mean', 'conc']
focus_df.loc[peak, 'conc std'] = focus_stats.loc['std', 'conc']
test=focii.reset_index()
# focus_df_full = sfc.peak_cycles(obj[check]['Plot'], focus_df.copy(), min_range=2, stop=40, full_stop=40)

# %% In index form to find points
plot_obj = sfc.Plotter(obj[check]['Prime']) # obj[check]['Plot']
obj[check]['Plot'].map_plot(name=str(label_num+': Error Evaluation'),  # log10(Data) & Weighted
                             info=['start_index', 'index_range', 'error'],
                             # matrix=error_df,
                             # conv=[1, 1e4],
                             # xscale='log',
                             # yscale='linear',
                             zscale='linlog',
                             # xlimit=[0, 50],
                             # ylimit=[0, 50],
                             # xlimit=[0, 600],
                             # ylimit=[0, 600],
                             # zlimit=[1e-2, 0.5],
                             # zlimit=[5e-5, 0.01],
                             levels=25,
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
                             # xlimit=[0, 2.5],
                             # ylimit=[0, 2.5],
                             zlimit=[7e-4, 0.07],
                             # zlimit=[1e-2, 0.5],
                             levels=10,
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
# %% Get Profile stats for specified pairs
# std_vars = ['start_index', 'stop_index', 'index_range', 'start_loc',
#             'stop_loc', 'diff', 'conc', 'error', 'stats']
pair_labels = ['start_index', 'index_range']
# pairs=[(2, 22), (2, 26), (2, 30), (2, 36), (10, 14), (10, 18), (10, 22), (10, 28),
#        (18, 10), (18, 16), (18, 25), (18, 39), (23, 14), (29, 17), (33, 15), (35, 23)]
# pairs=[(2, 22), (2, 26), (2, 30), (2, 36), (10, 14), (10, 18), (10, 22), (10, 28),
#        (18, 10), (18, 16), (18, 25), (18, 39), (23, 14), (29, 17), (33, 15), (35, 23)]
pairs = [(0,10)]
focus = obj[check]['Plot'].focus(pairs=pairs, pair_names=pair_labels) # var=['depth','sims','pred' pairs=None, pair_names

# low_err = obj[check]['Eval'].focus(pair_names=['start_index','error'])

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






# %% Review P type stats and stat options
# res_df2 = obj[check]['Ops'].gen_df(stat_type='normal_test')
# stats_matrix = obj['R-90'][ 'Ops'].matrix(info=['start_index', 'index_range', 'stats.ks_test'])
obj[check]['Ops'].set_attr('stats_settings', val=['log_form', True])
obj[check]['Ops'].set_attr('stats_settings', val=['resid_type', 'base'])

stat_type = 'shap_test'
obj[check]['Plot'].map_plot(
    name=str(label_num+': '+stat_type+' Evaluation'),  # log10(Data) & Weighted
    info=['start_index', 'index_range', str('stats.'+stat_type)],
    # matrix=error_df,
    # conv=[1, 1e4],
    # xscale='log',
    # yscale='linear',
    zscale='linlog',
    xlimit=[0, 75],
    ylimit=[0, 75],
    # zlimit=[2.5e-2, 1],
    zlimit=[5e-2, 1],
    levels=50,
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





# %% Residual analysis of peaks
# evaluation = obj['R-90']['Eval']
# check =obj[check]['Ops']._family[loc]
# inflect = [1, 5, 24, 50, 199]
# man = [(inflect[x], inflect[x+1]) for x in reversed(range(len(inflect)-1))]
# evaluation.stitcher(obj['R-90']['SIMS'], man)
# res = evaluation.stitched_res
# pair_labels = ['start_index', 'index_range']
# pairs = [(3, 8), (3, 19), (22, 4), (22, 11)] # 90
# pairs = [(18, 4), (18, 8)] # 80
# pairs = [(4, 4), (4, 18), (22, 4), (22, 15)] # 70

pair_labels = ['start_index', 'stop_index']
# pairs = [(3, 22), (22, 40)] # 90
# pairs = [(6, 10), (10, 16), (16, 20)] # 80
pairs = [(3, 12), (8, 13)] # 70

obj[check]['Ops'].set_attr('stats_settings', val=['log_form', True])
obj[check]['Ops'].set_attr('stats_settings', val=['resid_type', 'base'])

prof = obj[check]['Plot'].focus(pairs=pairs, pair_names=pair_labels, var=['data'])
# %% Residual analysis
pt = 1

prof_data = pd.DataFrame(prof.iloc[pt-1,0])
prof_data['depth'] = gf.fromcm(prof_data['depth'],'um')

plot_obj = sfc.Plotter()
plot_obj.prof_plot(name=str(label_num+': '+'Residual Plot'),
                   data_in=prof_data,
                    x='depth',
                    y='pred',
                    xscale='linear',
                    yscale='log',
                    xlimit=None,
                    ylimit=[1e16,1e20],
                    xname='Depth',
                    yname='Residuals',
                    # hline=True,
                    # hue='Range number',
                    legend=False,
                    # palette='kindlmann',
                    # hue_norm=(0, len(inflect)),
                    )
