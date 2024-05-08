# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:24:26 2022

@author: j2cle
"""
# %% import section
import numpy as np
import pandas as pd
import General_functions as gf
from itertools import islice,combinations
import Units_Primary3 as up
import matplotlib.pyplot as plt
import seaborn as sns
import abc
import re
from scipy.special import erfc
from scipy import stats
from sklearn import metrics
from functools import partial
from dataclasses import field, fields, astuple, dataclass, InitVar

from scipy.optimize import curve_fit
import warnings


warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings("ignore")

def lin_test(x, y, lim=0.025):
    """Return sum of squared errors (pred vs actual)."""
    line_info = np.array([np.polyfit(x[-n:], y[-n:], 1) for n in range(1, len(x))])

    delta = np.diff(line_info[int(.1*len(x)):, 0])
    delta = delta/max(abs(delta))
    bounds = np.where(delta < -lim)[0]
    if bounds[0]+len(bounds)-1 == bounds[-1]:
        bound = len(x)-bounds[0]
    else:
        bound = len(x)-[bounds[n] for n in range(1, len(bounds)) if bounds[n-1]+1 != bounds[n]][-1]
    return bound, x[bound]

def depth_conv(data_in, unit, layer_act, layer_meas):
    """Return sum of squared errors (pred vs actual)."""
    data_out = data_in
    if unit != 's':
        if not pd.isnull(layer_meas):
            data_out[data_in < layer_meas] = data_in[data_in < layer_meas]*layer_act/layer_meas
            data_out[data_in >= layer_meas] = ((data_in[data_in >= layer_meas]-layer_meas) *
                                               ((max(data_in)-layer_act) /
                                                (max(data_in)-layer_meas))) + layer_act

            # if col(A)< a2 ? col(A)*a1/a2 : (col(A)-a2)*((max(A)-a1)/(max(A)-a2))+a1
        if unit != 'cm':
            data_out = gf.tocm(data_in, unit)

    return data_out

def linear(x, coeffs):
    """Return sum of squared errors (pred vs actual)."""
    return coeffs[1] + coeffs[0] * x


class ImportFunc:
    func: str = ''
    args: list = field(default_factory=list)

    # def csv(self, *args):
    #     with open(args[0], 'r') as file:
    #         lines = []
    #         data=[]
    #         delims=[]
    #         for line in file:
    #             lines.append(line.strip())
    #             data.append(re.sub('([\s,;:])',',',line.strip()).split(','))
    #             delims.append(re.sub('([^\s,;:])','',line.strip()))
    #     data = [[m for m in n if m != ''] for n in data]
    #     data_info = []
    #     data_num = []
    #     data_lines = []
    #     for n in range(len(data)):
    #         if data[n] != []:
    #             try:
    #                 data_num.append(np.array([m for m in data[n] if m != ''], dtype=float))
    #             except ValueError:
    #                 data_info.append([m for m in data[n] if m != ''])
    #             else:
    #                 data_lines.append(n)
    #     start = 0
    #     while not all([len(x) == len(data_num[start]) for x in data_num]) and start < 50:
    #         start += 1

    #     if delims[data_lines[0]].count(delims[data_lines[0]][0]) == len(delims[data_lines[0]]):
    #         delim = delims[data_lines[0]][0]

    #     clean_line = [[m for m in n.split(delim) if m != ''] for n in lines[:]]

    #     data_raw = pd.DataFrame(data_num)
    #     return data_raw

    def nrel_d(self, *args):
        data_raw = pd.read_excel(args[0], sheet_name=args[1], usecols=args[2]).dropna()
        return data_raw

    def asu_raw(self, *args):
        header_in = pd.read_csv(args[0], delimiter='\t', header=None,skiprows=14, nrows=2).dropna(axis=1, how='all')
        header_temp = header_in.iloc[0,:].dropna().to_list() + header_in.iloc[0,:].dropna().to_list()
        header_in.iloc[0,:len(header_temp)] = sorted(header_temp, key=lambda y: header_temp.index(y))
        header_in = header_in.dropna(axis=1)
        headers = [header_in.iloc[0, x] + ' ' + header_in.iloc[1, x]
                    for x in range(header_in.shape[1])]
        data_raw = pd.read_csv(args[0], delimiter='\t', header=None, names=headers, index_col=False, skiprows=16).dropna().astype(float)
        return data_raw

    def rice_treated(self, *args):
        header_in = pd.read_csv(args[0], delimiter='\t', header=None,skiprows=2, nrows=3).dropna(axis=1, how='all')
        header_in = header_in.fillna(method='ffill', axis=1)
        headers = [header_in.iloc[0, x] + ' ' + header_in.iloc[2, x]for x in range(header_in.shape[1])]
        data_raw = pd.read_csv(args[0], delimiter='\t',header=None, names=headers, index_col=False, skiprows=5)
        return data_raw

    def rice_raw(self, *args):
        data_raw = pd.read_csv(args[0], delimiter='\s+', header=None, index_col=[0,1,2], names=['x','y','z','intens'], skiprows=10, dtype='int')
        return data_raw


class ConvFunc:
    def __init__(self, raw_data, params, func=''):
        self.data = pd.DataFrame(np.ones((len(raw_data), 2)), columns=['Depth', 'Na'])
        self.params = params
        self.func = func
        self.func(raw_data)


    @property
    def func(self):
        """Calculate constant, may shift to use the depth range instead."""
        return self._func

    @func.setter
    def func(self, value):
        if hasattr(self, value):
            self._func = getattr(self, value)
        else:
            self._func = getattr(self, 'error')

    def error(self, *args):
        print('available func not set')
        return

    def gen_col(self, df, a, b):
        return df[df.columns[[(a in x.lower()) and (b in x.lower()) for x in df.columns]]].to_numpy(copy=True)

    def nrel_d(self, raw):
        self.data['Depth'] = depth_conv(self.gen_col(raw,'na','x'),
                                        self.params['X unit'],
                                        self.params['Layer (actual)'],
                                        self.params['Layer (profile)'])
        self.data['Na'] = self.gen_col(raw,'na','y')/self.gen_col(raw,'12c','y').mean()*self.params['RSF']
        return

    def asu_raw(self, raw):
        rate = self.params['Max X']/self.gen_col(raw,'na','time').max()
        self.data['Depth'] = gf.tocm(self.gen_col(raw,'na','time')*rate, self.params['X unit'])
        self.data['Na'] = self.gen_col(raw,'na','c/s')/self.gen_col(raw,'12c','c/s').mean()*self.params['RSF']

        return

    def rice_semi_treated(self, raw):
        self.data['Depth'] = gf.tocm(self.gen_col(raw,'depth','dep'), self.params['X unit'])

        if 'counts' in self.params['Y unit'] and not np.isnan(self.params['RSF']):
            self.data['Na'] = self.gen_col(raw,'na+','intens') / self.gen_col(raw,'c_2h_5','intens').mean() * self.params['RSF']
        elif 'counts' in self.params['Y unit'] and not np.isnan(self.params['SF']):
            self.data['Na'] = self.gen_col(raw,'na+','intens') * self.params['SF']
        else:
            self.data['Na'] = self.gen_col(raw,'na+','intens')

        return

    def rice_treated(self, raw):
        self.data['Depth'] = gf.tocm(self.gen_col(raw,'depth','dep'), self.params['X unit'])
        self.data['Na'] = self.gen_col(raw,'na+','conc')
        return

    def rice_raw(self, *args):
        data_raw = pd.read_csv(args[0], delimiter='\s+', header=None, index_col=[0,1,2], names=['x','y','z','intens'], skiprows=10, dtype='int')
        return data_raw

            # elif 'matrix' in self.params['Type'].lower():
            #     if col.lower() == 'z':
            #         self.data['Depth'] = gf.tocm(
            #             data_raw[col].to_numpy(copy=True), self.params['X unit'])
            #     if col == self.params['Measurement']+' '+ str(int(self.params['Sample'][-1])-1) :
            #         na_col = col

class PrimeImport:
    def __init__(self, call):
        # call=['R-90','EVA_A','2-1']

        prime_path = 'C:\\Users\\j2cle\\Work Docs\\Data\\Analysis\\SIMS\\'
        active_log = pd.read_excel(f'{prime_path}Active Log.xlsx', index_col=0, header=0).dropna(axis=0, how='all').fillna('')
        called = active_log[active_log.iloc[:,:5].isin(call)].dropna(axis=1, how='all').dropna(axis=0, how='any')
        self.df_log = active_log.loc[called.index,:]

        self.not_called = [x for x in call if x not in self.df_log.to_numpy()]

        folders = pd.Series([f'{x[0]}\\{x[1]}' for x in self.df_log[['Source','Folder']].to_numpy()], index=self.df_log.index)
        files = pd.Series([f'{prime_path}{x[0]}\\{x[1]}\\Files{x[2]}\\{x[3]}'
                           for x in self.df_log[['Source','Folder','Sub Folder','File']].to_numpy()], index=self.df_log.index)


        for logs in folders.unique():
            self.params = pd.read_excel(f'{prime_path}{logs}\Sample Log.xlsx', index_col=0,  skiprows=1).dropna(axis=0, how='all')

        self.raws = {}
        for sample in self.df_log.index:
            func = getattr(self, self.df_log['Import Info'][sample])
            self.raws[sample] = func(files[sample], self.df_log.loc[sample,'Tab'],self.df_log.loc[sample,'Columns'])

    @property
    def params(self):
        """Calculate constant, may shift to use the depth range instead."""
        return self._params

    @params.setter
    def params(self, value):
        ind = value.index.join(self.df_log.index, how='inner')
        value = value.loc[ind,:]
        if hasattr(self, '_params'):
            self._params = pd.concat([self._params,value], join='outer')
        else:
            self._params = value

    def nrel_d(self, *args):
        data_raw = pd.read_excel(args[0], sheet_name=args[1], usecols=args[2]).dropna()
        return data_raw

    def asu_raw(self, *args):
        header_in = pd.read_csv(args[0], delimiter='\t', header=None,skiprows=14, nrows=2).dropna(axis=1, how='all')
        header_temp = header_in.iloc[0,:].dropna().to_list() + header_in.iloc[0,:].dropna().to_list()
        header_in.iloc[0,:len(header_temp)] = sorted(header_temp, key=lambda y: header_temp.index(y))
        header_in = header_in.dropna(axis=1)
        headers = [header_in.iloc[0, x] + ' ' + header_in.iloc[1, x]
                   for x in range(header_in.shape[1])]
        data_raw = pd.read_csv(args[0], delimiter='\t', header=None, names=headers, index_col=False, skiprows=16).dropna().astype(float)
        return data_raw

    def rice_treated(self, *args):
        header_in = pd.read_csv(args[0], delimiter='\t', header=None,skiprows=2, nrows=3).dropna(axis=1, how='all')
        header_in = header_in.fillna(method='ffill', axis=1)
        headers = [header_in.iloc[0, x] + ' ' + header_in.iloc[2, x]for x in range(header_in.shape[1])]
        data_raw = pd.read_csv(args[0], delimiter='\t',header=None, names=headers, index_col=False, skiprows=5)
        return data_raw

    def rice_semi_treated(self, *args):
        return self.rice_treated(*args)

    def rice_raw(self, *args):
        data_raw = pd.read_csv(args[0], delimiter='\s+', header=None, index_col=[0,1,2], names=['x','y','z','intens'], skiprows=10, dtype='int')
        return data_raw


test=PrimeImport(['R-90','EVA_A','2-1'])
# test.raws['R-90'][test.raws['R-90'].columns[[('Depth' in x) for x in test.raws['R-90'].columns]]]
# class DataProfile:
#     """Return sum of squared errors (pred vs actual)."""

#     def __init__(self, slog, limit=False, loc=None, even=False, **kwargs):
#         self.params = slog

#         self.data_treatment()

#         if not np.isnan(self.params['Layer (actual)']):
#             self.a_layer_cm = gf.tocm(self.params['Layer (actual)'], self.params['A-Layer unit'])
#         else:
#             self.a_layer_cm = 0

#         if not np.isnan(self.params['Fit depth/limit']):
#             self.fit_depth_cm = gf.tocm(self.params['Fit depth/limit'], self.params['Fit Dep unit'])
#         else:
#             self.params['Fit depth/limit'] = lin_test(self.data['Depth'].to_numpy(),
#                                                       self.data['Na'].to_numpy(), 0.05)[1]
#             self.params['Fit Dep unit'] = 'cm'

#         if not np.isnan(self.params['Layer (profile)']):
#             self.p_layer_cm = gf.tocm(self.params['Layer (profile)'], self.params['P-Layer unit'])

#         self.data_bgd = pd.Series()

#         self.limit_test()
#         # if 'tof' in self.params['Type'].lower():
#         self.regress_test(**kwargs)

#         if limit:
#             if loc is None:
#                 self.data = self.data.iloc[self.data_bgd['bgd_ave'],:]
#             elif isinstance(loc, (int, np.integer)):
#                 self.data = self.data.iloc[loc,:]
#             else:
#                 self.data = self.data[self.data['Depth'] < loc]

#         if even:
#             self.data['Depth'] = np.linspace(self.data['Depth'].min(),
#                                              self.data['Depth'].max(),
#                                              len(self.data['Depth']))


#     def data_treatment(self):
#         """Return sum of squared errors (pred vs actual)."""
#         if self.params['Type'] == 'NREL MIMS':
#             data_raw = pd.read_excel(self.params['Data File Location'],sheet_name=self.params['Tab'],usecols=self.params['Columns']).dropna()
#         elif 'matrix' in self.params['Type'].lower():
#             data_raw = pd.read_excel(self.params['Data File Location'], header=[0,1], index_col=0)
#             data_raw.columns = data_raw.columns.map('{0[0]} {0[1]}'.format)
#             data_raw = data_raw.reset_index()
#         elif 'TOF' in self.params['Type']:
#             header_in = pd.read_csv(self.params['Data File Location'], delimiter='\t', header=None,skiprows=2, nrows=3).dropna(axis=1, how='all')
#             header_in = header_in.fillna(method='ffill', axis=1)
#             headers = [header_in.iloc[0, x] + ' ' + header_in.iloc[2, x]for x in range(header_in.shape[1])]
#             data_raw = pd.read_csv(self.params['Data File Location'], delimiter='\t',header=None, names=headers, index_col=False, skiprows=5)
#         elif 'DSIMS' in self.params['Type']:
#             header_in = pd.read_csv(
#                 self.params['Data File Location'], delimiter='\t', header=None,
#                 skiprows=14, nrows=2).dropna(axis=1, how='all')
#             header_temp = header_in.iloc[0,:].dropna().to_list() + header_in.iloc[0,:].dropna().to_list()
#             header_in.iloc[0,:len(header_temp)] = sorted(header_temp, key=lambda y: header_temp.index(y))
#             header_in = header_in.dropna(axis=1)
#             headers = [header_in.iloc[0, x] + ' ' + header_in.iloc[1, x]
#                        for x in range(header_in.shape[1])]
#             data_raw = pd.read_csv(self.params['Data File Location'], delimiter='\t', header=None, names=headers, index_col=False, skiprows=16).dropna().astype(float)



#         self.data = pd.DataFrame(np.ones((len(data_raw), 2)), columns=['Depth', 'Na'])
#         data_cols = list(data_raw.columns)

#         if 'atoms' in self.params['Y unit'].lower():
#             col_type = 'conc'
#         elif 'dsims' in self.params['Type'].lower():
#             col_type = 'i [c/s]'
#         else:
#             col_type = 'inten'

#         for col in data_cols:
#             if self.params['Type'] == 'NREL MIMS':
#                 if 'x' in col.lower() or 'depth' in col.lower():
#                     self.data['Depth'] = depth_conv(data_raw[col].to_numpy(copy=True),
#                                                     self.params['X unit'],
#                                                     self.params['Layer (actual)'],
#                                                     self.params['Layer (profile)'])
#                 if 'na' in col.lower():
#                     na_col = col
#                 if self.params['Matrix'] in col:
#                     data_matrix = data_raw[col].to_numpy()
#             elif 'matrix' in self.params['Type'].lower():
#                 if col.lower() == 'z':
#                     self.data['Depth'] = gf.tocm(
#                         data_raw[col].to_numpy(copy=True), self.params['X unit'])
#                 if col == self.params['Measurement']+' '+ str(int(self.params['Sample'][-1])-1) :
#                     na_col = col
#             elif 'tof' in self.params['Type'].lower():
#                 if 'x' in col.lower() or 'depth' in col.lower():
#                     self.data['Depth'] = gf.tocm(
#                         data_raw[col].to_numpy(copy=True), self.params['X unit'])
#                 if self.params['Ion'].lower() in col.lower() and col_type in col.lower():
#                     na_col = col
#                 if self.params['Matrix'] in col and 'inten' in col.lower():
#                     data_matrix = data_raw[col].to_numpy()
#             elif 'dsims' in self.params['Type'].lower():
#                 if 'na time' in col.lower():
#                     self.data['Depth'] = gf.tocm(
#                         data_raw[col].to_numpy(copy=True)*self.params['Max X']/data_raw[col].max(), self.params['X unit'])
#                 if self.params['Ion'].lower() in col.lower() and col_type in col.lower():
#                     na_col = col
#                 if ' '.join([self.params['Matrix'].lower(), col_type]) in col.lower():
#                     data_matrix = data_raw[col].to_numpy()


#         if 'counts' in self.params['Y unit'] and not np.isnan(self.params['RSF']):
#             self.data['Na'] = data_raw[na_col].to_numpy() / np.mean(data_matrix)*self.params['RSF']
#         elif 'counts' in self.params['Y unit'] and not np.isnan(self.params['SF']):
#             self.data['Na'] = data_raw[na_col].to_numpy()*self.params['SF']
#         else:
#             self.data['Na'] = data_raw[na_col].to_numpy()

#     def limit_test(self, thresh=0.025):
#         """Return sum of squared errors (pred vs actual)."""
#         lin_loc, lin_lim = lin_test(
#             self.data['Depth'].to_numpy(), self.data['Na'].to_numpy(), thresh)

#         if lin_lim > self.fit_depth_cm*1.1 or lin_lim < self.fit_depth_cm*0.9:
#             self.data_bgd['bgd_lim'] = gf.find_nearest(self.data['Depth'].to_numpy(),
#                                                        self.fit_depth_cm)
#         else:
#             self.data_bgd['bgd_lim'] = lin_loc

#     def regress_test(self, alpha=0.05, ind_range=10, **kwargs):
#         """Return sum of squared errors (pred vs actual)."""
#         stop = len(self.data['Depth'])
#         cng = int(len(self.data['Depth'])*0.02)
#         perc = 0
#         while perc < 0.2 and stop > len(self.data['Depth'])*0.5:
#             self.p = np.ones(stop-10)
#             for x in range(stop-10):
#                 coeff = stats.linregress(self.data['Depth'].to_numpy()[x:stop],
#                                          self.data['Na'].to_numpy()[x:stop], **kwargs)[:2]
#                 resid = (self.data['Na'].to_numpy()[x:stop] -
#                          linear(self.data['Depth'].to_numpy()[x:stop], coeff))
#                 self.p[x] = stats.normaltest(resid)[1]
#             stop -= cng
#             perc = len(self.p[self.p > alpha])/len(self.p)

#         itr = 0
#         while self.p[itr] < alpha and itr < int((len(self.data['Na'])-10)*.75):
#             itr += 1
#         self.data_bgd['bgd_max'] = itr

#         ind = 0
#         while ind < ind_range and itr < int((len(self.data['Na'])-10)*.9):
#             ind += 1
#             itr += 1
#             if self.p[itr] < alpha:
#                 ind = 0
#         self.data_bgd['bgd_min'] = itr-ind

#         self.data_bgd['bgd_ave'] = int((self.data_bgd['bgd_max']+self.data_bgd['bgd_min'])/2)
#         coeff = stats.linregress(self.data['Depth'].to_numpy()[self.data_bgd['bgd_ave']:],
#                                  self.data['Na'].to_numpy()[self.data_bgd['bgd_ave']:],
#                                  **kwargs)[:2]
#         self.data_bgd['P-value'] = self.p[self.data_bgd['bgd_ave']]
#         self.data_bgd['slope'] = coeff[0]
#         self.data_bgd['intercept'] = coeff[1]

#     @property
#     def thick_cm(self):
#         """Return sum of squared errors (pred vs actual)."""
#         return gf.tocm(self.params['Thick'], self.params['Thick unit'])

