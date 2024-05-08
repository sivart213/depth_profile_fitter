# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 12:36:55 2021.

@author: j2cle
"""

# %% import section
import numpy as np
import pandas as pd
import General_functions as gf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import seaborn as sns
import abc
import statsmodels.api as sm
from scipy.special import erfc
from scipy import stats
from functools import partial
# import lmfit
# from lmfit import Model
# from scipy.stats.distributions import  t
# import uncertainties as unc
# import uncertainties.unumpy as unp

from scipy.optimize import curve_fit
# from statsmodels.stats.weightstats import DescrStatsW
import warnings
from sklearn.metrics import mean_absolute_percentage_error as MAPE
warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings("ignore")

sns.set_style('dark')


# %% Functions
def c_np(depth, diff, c_0, thick, temp, e_app, time):
    """
    Calculate NP.

    Takes preforms a full calc of the NP equation
    """
    if diff < 0:
        diff = 10**diff
        c_0 = 10**c_0
    mob = diff/(gf.KB_EV*temp)
    term_B = erfc(-mob*e_app*time/(2*np.sqrt(diff*time)))
    return (c_0/(2*term_B)) * (erfc((depth - mob*e_app * time)/(2*np.sqrt(diff*time))) +
                               erfc(-(depth-2*thick+mob*e_app*time)/(2*np.sqrt(diff*time))))


def depth_conv(data_in, unit, layer_act, layer_meas):
    """
    Calculate.

    generic discription
    """
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


def lin_test(x, y, lim=0.025):
    """
    Calculate.

    generic discription
    """
    line_info = np.array([np.polyfit(x[-n:], y[-n:], 1) for n in range(1, len(x))])
    delta = np.diff(line_info[int(.1*len(x)):, 0])
    delta = delta/max(abs(delta))
    bounds = np.where(delta < -lim)[0]
    if bounds[0]+len(bounds)-1 == bounds[-1]:
        bound = len(x)-bounds[0]
    else:
        bound = len(x)-[bounds[n] for n in range(1, len(bounds)) if bounds[n-1]+1 != bounds[n]][-1]
    return bound, x[bound]


# %% Classes
class Component(metaclass=abc.ABCMeta):
    """
    Treat initial data.

    Imports
    """

    @abc.abstractmethod
    def set_error(self, **kwargs):
        """
        Treat initial data.

        Imports
        """
        pass

    @abc.abstractmethod
    def set_best_error(self, **kwargs):
        """
        Treat initial data.

        Imports
        """
        pass


class Composite(Component):
    """
    Treat initial data.

    Imports
    """

    def __init__(self, limit=False):
        self._children = list()
        self._family = list()
        self.limit = limit

    def set_error(self, limit=False, **kwargs):
        """
        Treat initial data.

        Imports
        """
        self.limit = limit
        for work in (self.chores):
            work.set_error(**kwargs)

    def set_best_error(self, limit=False, **kwargs):
        """
        Treat initial data.

        Imports
        """
        self.limit = limit
        for work in (self.chores):
            work.set_best_error(**kwargs)

    def set_attr(self, attr='x0_index', num=0, limit=False, **kwargs):
        """
        Treat initial data.

        Imports
        """
        self.limit = limit
        for work in (self.chores):
            setattr(work, attr, num)

    def get_attr(self, attr='error', limit=False, **kwargs):
        """
        Treat initial data.

        Imports
        """
        work_attr = list()
        self.limit = limit
        for work in (self.chores):
            work_attr.append(getattr(work, attr))
        return work_attr

    def del_attr(self, attr='error', limit=False, **kwargs):
        """
        Treat initial data.

        Imports
        """
        work_attr = list()
        self.limit = limit
        for work in (self.chores):
            if hasattr(work, attr):
                delattr(work, attr)


    def add(self, component):
        """
        Treat initial data.

        Imports
        """
        self._children.append(component)
        if component not in self._family:
            self._family.append(component)

    def drop(self, component):
        """
        Treat initial data.

        Imports
        """
        if component in self._children:
            self._children.remove(component)

    @property
    def chores(self):
        """
        Treat initial data.

        Imports
        """
        if self.limit:
            self._worker = self._children
        else:
            self._worker = self._family

        return self._worker


class DataProfile:
    """
    Treat initial data.

    Imports
    """

    def __init__(self, slog):
        self.params = slog

        self.Data()

        if not np.isnan(self.params['Layer (actual)']):
            self.a_layer_cm = gf.tocm(self.params['Layer (actual)'], self.params['A-Layer unit'])
        else:
            self.a_layer_cm = 0

        if not np.isnan(self.params['Fit depth/limit']):
            self.fit_depth_cm = gf.tocm(self.params['Fit depth/limit'], self.params['Fit Dep unit'])
        else:
            self.params['Fit depth/limit'] = lin_test(self.data['Depth'].to_numpy(),
                                                      self.data['Na'].to_numpy(), 0.05)[1]
            self.params['Fit Dep unit'] = 'cm'

        if not np.isnan(self.params['Layer (profile)']):
            self.p_layer_cm = gf.tocm(self.params['Layer (profile)'], self.params['P-Layer unit'])

        self.Lim()

    def Data(self):
        """
        Calculate.

        generic discription
        """
        if self.params['Type'] == 'NREL MIMS':
            data_raw = pd.read_excel(
                self.params['Data File Location'],
                sheet_name=self.params['Tab'],
                usecols=self.params['Columns']).dropna()
        elif 'TOF' in self.params['Type']:
            header_in = pd.read_csv(
                self.params['Data File Location'], delimiter='\t', header=None,
                skiprows=2, nrows=3).dropna(axis=1, how='all')
            header_in = header_in.fillna(method='ffill', axis=1)
            headers = [header_in.iloc[0, x] + ' ' + header_in.iloc[2, x]
                       for x in range(header_in.shape[1])]
            data_raw = pd.read_csv(self.params['Data File Location'], delimiter='\t',
                                   header=None, names=headers, index_col=False,
                                   skiprows=5)

        self.data = pd.DataFrame(np.ones((len(data_raw), 2)), columns=['Depth', 'Na'])
        data_cols = list(data_raw.columns)

        for col in data_cols:
            if self.params['Type'] == 'NREL MIMS':
                if 'x' in col.lower() or 'depth' in col.lower():
                    self.data['Depth'] = depth_conv(data_raw[col].to_numpy(copy=True),
                                                    self.params['X unit'],
                                                    self.params['Layer (actual)'],
                                                    self.params['Layer (profile)'])
                if 'na' in col.lower():
                    na_col = col
                if self.params['Matrix'] in col:
                    data_matrix = data_raw[col].to_numpy()
            elif self.params['Type'] == 'TOF':
                if 'x' in col.lower() or 'depth' in col.lower():
                    self.data['Depth'] = gf.tocm(
                        data_raw[col].to_numpy(copy=True), self.params['X unit'])
                if 'na+' in col.lower() and 'conc' in col.lower():
                    na_col = col
                if self.params['Matrix'] in col and 'inten' in col.lower():
                    data_matrix = data_raw[col].to_numpy()
            elif self.params['Type'] == 'TOF Local':
                if 'x' in col.lower() or 'depth' in col.lower():
                    self.data['Depth'] = gf.tocm(
                        data_raw[col].to_numpy(copy=True), self.params['X unit'])
                if 'na+' in col.lower() and 'inten' in col.lower():
                    na_col = col
                if self.params['Matrix'] in col and 'inten' in col.lower():
                    data_matrix = data_raw[col].to_numpy()

        if 'counts' in self.params['Y unit'] and not np.isnan(self.params['RSF']):
            self.data['Na'] = data_raw[na_col].to_numpy() / \
                np.mean(data_matrix)*self.params['RSF']
        elif 'counts' in self.params['Y unit'] and not np.isnan(self.params['SF']):
            self.data['Na'] = data_raw[na_col].to_numpy()*self.params['SF']
        else:
            self.data['Na'] = data_raw[na_col].to_numpy()

    def Lim(self, thresh=0.025):
        """
        Calculate.

        generic discription
        """
        lin_loc, lin_lim = lin_test(
            self.data['Depth'].to_numpy(), self.data['Na'].to_numpy(), thresh)
        if lin_lim > self.fit_depth_cm*1.1 or lin_lim < self.fit_depth_cm*0.9:
            self.lim_loc = gf.find_nearest(self.data['Depth'].to_numpy(), self.fit_depth_cm)
        else:
            self.lim_loc = lin_loc

    @property
    def thick_cm(self):
        """
        Calculate.

        call and use the private variable
        """
        return gf.tocm(self.params['Thick'], self.params['Thick unit'])


class SimProfile(Component):
    """
    Store Profile information.

    This class is intended to store the information inherent in to a depth
    profile fit. This should include the fit profile, it's properties, and
    its range.  Curently also includes the fitted data and the resulting error.
    I'm not sure that it is important for this fitted information to be in a
    single classs.
    """

    def __init__(self, obj):
        obj = obj

        self.depth = obj.data['Depth'].to_numpy()
        self.sims = obj.data['Na'].to_numpy()
        self.pred = np.ones_like(self.sims)

        self.conditions = pd.Series({'thick': obj.thick_cm,
                                     'time': obj.params['Stress Time'],
                                     'temp': obj.params['Temp'],
                                     'e_field': obj.params['Volt']/obj.thick_cm,
                                     'volt': obj.params['Volt']})
        self.info = pd.Series({'ident': self.ident,
                               'sample': obj.params['Sample'],
                               'type': obj.params['Type'],
                               'class': 'SimProfile',
                               'measurement': obj.params['Measurement']})
        self.bkg_index = self.x1_index
        self.min_range = 2
        self.min_index = 0
        self.diff = 1e-17
        self.surf_conc = 1e17
        self.error = 1
        self.p_value = 0
        self.error_log = pd.DataFrame({'diff': self.diff,
                                       'surf_conc': self.surf_conc,
                                       'x0': self.x0_index,
                                       'x1': self.x0_index,
                                       'error': self.error,
                                       'p-value': self.p_value})


    @property
    def data(self):
        """
        Calculate.

        generic discription
        """
        self._data['depth'] = self.depth
        self._data['SIMS'] = self.sims
        self._data['log(SIMS)'] = np.log10(self.sims)
        self._data['pred'] = self.pred
        self._data['log(SIMS)'] = np.log10(self.pred)
        self._data['residuals'] = self.sims - self.pred
        self._data['residuals of log'] = np.log10(self.sims) - np.log10(self.pred)
        self._data['log of residuals'] = np.log10(self.sims - self.pred)
        return self._data

    @property
    def ident(self):
        """
        Calculate.

        generic discription
        """
        return id(self)

    @property
    def x0_index(self):
        """
        Calculate.

        generic discription only change
        """
        if not hasattr(self,'_x0_index'):
            self._x0_index = 0
        return self._x0_index

    @x0_index.setter
    def x0_index(self, value):
        if self._x1_index <= value or value < 0:
            print('Obj', self.ident, 'atempted to set x0 to', value)
        else:
            self._x0_index = int(value)

    @property
    def x1_index(self):
        """
        Calculate.

        generic discription
        """
        if not hasattr(self,'_x1_index'):
            self._x1_index = max(self.data.index)
        return self._x1_index

    @x1_index.setter
    def x1_index(self, value):
        if self._x0_index >= value or value > max(self.data.index):
            print('Obj', self.ident, 'atempted to set x1 to', value)
        else:
            self._x1_index = int(value)

    @property
    def x0_loc(self):
        """
        Calculate.

        generic discription
        """
        return gf.fromcm(self.depth[self.x0_index], 'um')

    @property
    def x1_loc(self):
        """
        Calculate.

        generic discription
        """
        return gf.fromcm(self.depth[self.x1_index], 'um')

    @property
    def index_range(self):
        """
        Calculate.

        generic discription
        """
        return self.x1_index-self.x0_index+1

    @property
    def depth_range(self):
        """
        Calculate.

        generic discription
        """
        return self.x1_loc-self.x0_loc


    @property
    def ks_test(self):
        """
        Calculate.

        generic discription
        """
        self.ks_stat, self.ks_p = stats.ks_2samp(self.pred[self.x0_index:self.x1_index],
                              self.sims[self.x0_index:self.x1_index])
        return self.ks_p

    @property
    def shap_test(self):
        """
        Calculate.

        generic discription
        """
        try:
            self.shap_stat, self.shap_p = stats.shapiro(self.residual)
        except (ValueError, TypeError) as error:
            self.shap_p = 0
            self.shap_stat = 1
        return self.shap_p

    def set_error(self, save_res=True, **kwargs):
        """
        Calculate.

        generic discription.
        err(self, instr='None', use_sample_w=False, w_array=None, to_log=False, **kwargs):
        """
        if not hasattr(self, '_err_func_to_use'):
            self._err_func_to_use = getattr(
                ProfileOps(self.data, self.pred, x0=self.x0_index, x1=self.x1_index), 'err')
        err_temp = self._err_func_to_use(**kwargs)
        if save_res:
            self.error = err_temp

        return err_temp

    def set_best_error(self, use_index=True, x_in=-1, reset=True, reverse=False, **kwargs):
        """
        Calculate.

        generic discription
        set_error(self, save_res=True, **kwargs):
        err(self, instr='None', use_sample_w=False, w_array=None, to_log=False, **kwargs):
        """
        if reset:
            err_last = 1
        else:
            err_last = self.error

        if not reverse:
            if use_index:
                x_in = self.x1_index
            elif x_in <= 0:
                x_in = max(self.data.index)

            self.err_array = np.array([self.set_error(False, x0=start, x1=x_in,  **kwargs)
                                       for start in range(x_in-self.min_range+1)])

            if np.min(self.err_array) < err_last:
                self.min_index = np.argmin(self.err_array)
                err_last = self.error
                self.error = np.min(self.err_array)
        else:
            if use_index:
                start = self.x0_index
            elif x_in <= 0:
                start = 0
            self.err_array = np.array([self.set_error(False, x0=start, x1=stop,  **kwargs)
                                       for stop in range(start+self.min_range, self.x1_index+1)])

            if np.min(self.err_array) < err_last:
                self.min_index = np.argmin(self.err_array)
                err_last = self.error
                self.error = np.min(self.err_array)


class PredProfile(SimProfile):
    """
    Generate profile from diff, surf_conc, and simulation parameters.

    Creates a simulated profile by fitting real data.
    """

    def __init__(self, true_prof, diff=None, surf_conc=None, **kwargs):
        """
        Calculate.

        generic discription
        """
        # constant once set
        self.std_info = pd.Series({'xmin': -14})
        _diff_max = -14
        _conc_min = 15
        _conc_max = 21
        _xmin = 0
        super().__init__(true_prof)

        if diff is not None:
            self.diff = diff
        if surf_conc is not None:
            self.surf_conc = surf_conc

        self.unpack_kwargs(kwargs)

        self.pred = c_np(depth=self.depth, diff=self.diff,
                         c_0=self.surf_conc, thick=self.conditions['thick'],
                         temp=gf.CtoK(self.conditions['temp']),
                         e_app=self.conditions['e_field'],
                         time=self.conditions['time'])

    def unpack_kwargs(self, kwargs):
        """
        Calculate.

        generic discription
        """
        self.__dict__.update(kwargs)

    @property
    def pred(self):
        """
        Calculate.

        call and use the private variable
        """
        return np.where(self._pred <= 1e-30, 1e-30, self._pred)

    @pred.setter
    def pred(self, pred_in):
        self._pred = pred_in


class FitProfile(SimProfile):
    """
    Calculate.

    generic discription
    """

    _curve_fit_keys = list(curve_fit.__code__.co_varnames) + ['x_scale', 'xtol', 'jac']

    def __init__(self, true_prof, x0_index=None, x1_index=None, **kwargs):
        """
        Calculate.

        generic discription
        """
        super().__init__(true_prof)
        if x0_index is not None:
            if isinstance(x0_index, (float, np.float)):
                self.x0_index = gf.find_nearest(self.depth, x0_index)
            elif isinstance(x0_index, (int, np.int)):
                self.x0_index = x0_index
        if x1_index is not None:
            if isinstance(x1_index, (float, np.float)):
                self.x1_index = gf.find_nearest(self.depth, x1_index)
            elif isinstance(x1_index, (int, np.int)):
                self.x1_index = x1_index

        self.c_np_new = partial(c_np, thick=self.conditions['thick'],
                                temp=gf.CtoK(self.conditions['temp']),
                                e_app=self.conditions['e_field'],
                                time=self.conditions['time'])

        self.curve_fit_kwargs = {'x_scale': 'jac', 'xtol': 1e-12, 'jac': '3-point'}
        self.unpack_kwargs(kwargs)
        # print(self.x0_index,'-',self.x1_index)
        self.fitter([-20, -14, -10], [15, 19, 21], **kwargs)

    def unpack_kwargs(self, kwargs):
        """
        Calculate.

        generic discription
        """
        self.curve_fit_kwargs.update({key: kwargs[key] for key in kwargs
                                      if key in self._curve_fit_keys})
        [kwargs.pop(x) for x in self._curve_fit_keys if x in kwargs.keys()]
        self.__dict__.update(kwargs)

    def fitter(self, D_pred, C0_pred, **kwargs):
        """
        Calculate.

        generic discription
        """
        self.unpack_kwargs(kwargs)
        try:
            fittemp = curve_fit(self.c_np_new,
                                self.depth[self.x0_index:self.x1_index+1],
                                self.sims[self.x0_index:self.x1_index+1],
                                p0=(D_pred[1], C0_pred[1]),
                                bounds=((D_pred[0], C0_pred[0]), (D_pred[2], C0_pred[2])),
                                **self.curve_fit_kwargs)
        except RuntimeError:
            self.fit_res = [self.diff, self.diff, self.surf_conc, self.surf_conc]
            print(self.x0_index, '-', self.x1_index)

        else:
            self.fit_res = [10**fittemp[0][0],
                            (10**(fittemp[0][0] + np.sqrt(np.diag(fittemp[1]))[0]) -
                             10**(fittemp[0][0]-np.sqrt(np.diag(fittemp[1]))[0]))/2,
                            10**fittemp[0][1],
                            (10**(fittemp[0][1]+np.sqrt(np.diag(fittemp[1]))[1]) -
                             10**(fittemp[0][1]-np.sqrt(np.diag(fittemp[1]))[1]))/2]
        self.diff = self.fit_res[0]
        self.surf_conc = self.fit_res[2]

    @property
    def pred(self):
        """
        Calculate.

        call and use the private variable
        """
        self._pred = np.array(self.c_np_new(self.depth, self.diff, self.surf_conc))
        return np.where(self._pred <= 1e-30, 1e-30, self._pred)

    @property
    def D_cov(self):
        """
        Calculate.

        call and use the private variable
        """
        return self.fit_res[1]

    @property
    def C0_cov(self):
        """
        Calculate.

        call and use the private variable
        """
        return self.fit_res[3]


class ProfileOps:
    """
    Calculate.

    requires Obj input, can create subclas with new init to do profiles directly if needed
    """

    _mape_keys = list(MAPE.__code__.co_varnames)

    def __init__(self, data, pred, x0=0, x1=None, **kwargs):
        """
        Calculate.

        requires Obj input, can create subclas with new init to do profiles directly if needed
        """
        self.data = data.copy()
        self.pred = pred
        self.x0 = x0
        if x1 is None:
            self.x1 = len(self.pred)-1
        else:
            self.x1 = x1
        self.unpack_kwargs(kwargs)

    def unpack_kwargs(self, kwargs):
        """
        Calculate.

        generic discription
        """
        self.mape_kwargs = {key: kwargs[key] for key in kwargs if key in self._mape_keys}
        [kwargs.pop(x) for x in self._mape_keys if x in kwargs.keys()]
        self.__dict__.update(kwargs)

    @property
    def w_constant(self):
        """
        Calculate.

        generic discription
        """
        return self._w_constant

    @w_constant.setter
    def w_constant(self, instr='logic'):
        self.w_range = len(self.sims)

        if instr.lower() == 'logic':
            vals_incl = len(self.sims[self.x0:self.x1+1]
                            [(self.sims > self.pred)[self.x0:self.x1+1]])
        elif instr.lower() == 'base':
            vals_incl = len(self.sims[self.x0:self.x1+1])
        else:
            vals_incl = self.w_range/100

        if vals_incl <= 0:
            vals_incl = self.w_range/100

        self._w_constant = self.w_range/(100 * vals_incl)

    def sample_weights(self, w_array, **kwargs):
        """
        Calculate sample weights.

        can pass a precalculated array or nothing.  sample_w is generated regardless
        """
        if w_array is None:
            if self.data.loc[0, 'Na'] < 100:
                self.mape_kwargs['sample_weight'] = np.array([
                    (10**self.data.loc[x, 'Na'])/(10**self.pred[x])
                    if self.pred[x] > self.data.loc[x, 'Na']
                    else 1 for x in range(self.x0, self.x1+1)])
            else:
                self.mape_kwargs['sample_weight'] = np.array([
                    (self.data.loc[x, 'Na'])/(self.pred[x])
                    if self.pred[x] > self.data.loc[x, 'Na']
                    else 1 for x in range(self.x0, self.x1+1)])
        elif type(w_array) is not np.ndarray:
            self.mape_kwargs['sample_weight'] = np.array(w_array)

    def err(self, instr='None', use_sample_w=False, w_array=None, to_log=False, **kwargs):
        """
        Calculate error.

        error for the input information, information generated at call,
        requires type input for constant, can pass the information to sample
        weights if desired. can rewrite to always pass sample_weights via
        kwargs.
        """
        if to_log and self.data.loc[0, 'Na'] > 100:
            self.data['Na'] = np.log10(self.sims)
            self.pred = np.log10(self.pred)

        self.unpack_kwargs(kwargs)

        self.w_constant = str(instr)
        if use_sample_w:
            self.sample_weights(w_array)

        self.error = (MAPE(self.sims[self.x0:self.x1+1],
                           self.pred[self.x0:self.x1+1],
                           **self.mape_kwargs) * self.w_constant)

        return self.error


class Matrix_Ops:
    """
    Treat initial data.

    Imports
    """

    def __init__(self, true_prof, cls_type, size=100, min_range=2, **kwargs):
        """
        Calculate.

        generic discription  cols = [15,21], rows = [-17,-11]
        """
        self.true_prof = true_prof
        self.data = self.true_prof.data.copy()
        self.cls_type = cls_type
        self.size = size
        self.min_range = min_range

        self.col = col
        self.row = row

        if self.cls_type == 'PredProfile':
            if len(col) == 0:
                col = [15, 21]
                row = [-17, -11]
            self.cols = np.logspace(col[0], col[1], self.size)
            self.rows = np.logspace(row[0], row[1], self.size)
            self.cols = np.array([gf.Sig_figs(x, 3) for x in self.cols])
            self.rows = np.array([gf.Sig_figs(x, 3) for x in self.rows])

        elif self.cls_type == 'FitProfile':
            if len(col) == 0: # values to go from 0 to max --> linear array made to the size requested
        # if in is sent, I'm sending to get by index values if size < x1 then there are spaces
                col = [0, self.depth[-1]]
                row = [0, self.depth[-1]]
            elif isinstance(col[1], (int, np.int)) and self.size >= col[1]: # fits
                self.cols = self.depth[col[0]:col[1]]
                self.rows = self.depth[col[0]:col[1]]

                col = [self.depth[col[0]], self.data.loc[col[1], 'Depth']]
                row = [self.data.loc[col[0], 'Depth'], self.data.loc[row[1], 'Depth']]

            self.cols = np.linspace(col[0], col[1], self.size)
            self.rows = np.linspace(row[0], row[1], self.size)


            self.cols = np.array([gf.Sig_figs(x, 3) for x in self.cols])
            self.rows = np.array([gf.Sig_figs(x, 3) for x in self.rows])
            # self.cols = range(cols[0], cols[1])
            # self.rows = range(rows[0], rows[1])
        else:
            print('There is an error')

    def blank_matrix(self, start=None, stop=None, size=None, array_gen='log'):

        if self.cls_type == 'PredProfile':

            return pd.DataFramenp.logspace(start, stop, self.size)

        if self.cls_type == 'PredProfile':
            if columns=None:
                col = [15, 21]
                row = [-17, -11]
            self.cols = np.logspace(col[0], col[1], self.size)
            self.rows = np.logspace(row[0], row[1], self.size)
            self.cols = np.array([gf.Sig_figs(x, 3) for x in self.cols])
            self.rows = np.array([gf.Sig_figs(x, 3) for x in self.rows])

    @property
    def ident(self):
        return id(self)

    @property
    def error_matrix(self):
        """
        Calculate.

        generic discription
        """
        if hasattr(self, 'obj_matrix'):
            return self.obj_matrix.applymap(lambda x: x.error
                                            if not isinstance(x, (int, np.int)) else None)
        else:
            return np.ones((self.size, self.size))

    @property
    def fit_curves(self):
        """
        Calculate.

        generic discription
        """
        col = ['Location', 'error', 'diff', 'surf_conc', 'x0 index', 'x1 index',
               'range (points)', 'x0', 'x1', 'range (um)']

        if hasattr(self, 'min_array'):
            return pd.DataFrame([[tuple(x),
                                self.obj_matrix.iloc[tuple(x)].error,
                                self.obj_matrix.iloc[tuple(x)].diff,
                                self.obj_matrix.iloc[tuple(x)].surf_conc,
                                self.obj_matrix.iloc[tuple(x)].x0_index,
                                self.obj_matrix.iloc[tuple(x)].x1_index,
                                self.obj_matrix.iloc[tuple(x)].index_range,
                                self.obj_matrix.iloc[tuple(x)].x0_loc,
                                self.obj_matrix.iloc[tuple(x)].x1_loc,
                                self.obj_matrix.iloc[tuple(x)].depth_range]
                                for x in self.min_array], columns=col)
        else:
            return pd.DataFrame(np.empty((0, len(col)), int), columns=col)


    def gen_matrix(self, **kwargs):
        """
        Calculate.

        generic discription
        """
        if self.cls_type == 'PredProfile':
            self.obj_matrix = pd.DataFrame([[
                PredProfile(self.true_prof, diff=y, surf_conc=x, **kwargs) for x in self.cols]
                for y in self.rows], columns=self.cols, index=self.rows)
        elif self.cls_type == 'FitProfile':
            self.obj_matrix = pd.DataFrame([[
                FitProfile(self.true_prof, start=x, stop=y, **kwargs) if (x < y)
                else 1 for x in self.cols] for y in self.rows],
                columns=self.cols, index=self.rows)
        else:
            print('There is an error')

        self.obj_operator = Composite()
        self.obj_matrix.applymap(lambda x: self.obj_operator.add(x)
                                      if not isinstance(x, (int, np.int)) else None)
        if self.obj_operator._family[0].min_range != self.min_range:
            self.obj_operator.set_attr(attr='min_range', num=self.min_range, limit=False)

    def error_calc(self, get_best=True, **kwargs): #reset=False
        """
        Calculate.

        generic discription
        """
        if not hasattr(self, 'obj_matrix'):
            self.gen_matrix(**kwargs)
        # if reset:
        #     self.obj_operator.set_attr(attr='error', num=1, limit=False)
        #     self.obj_operator.del_attr(attr='_err_func_to_use', limit=False)

        if get_best:
            self.obj_operator.set_best_error(**kwargs)
        else:
            self.obj_operator.set_error(**kwargs)

    def set_bkg(self, **kwargs):
        """
        Calculate.

        generic discription
        """
        if not hasattr(self, 'obj_matrix'):
            self.gen_matrix()

        self.error_calc(use_index=False, **kwargs)

        self.minima_bkg = np.unravel_index(
            self.error_matrix.to_numpy(na_value=np.inf).argmin(), self.error_matrix.shape)

        # finds the start of bkgrnd --> returns int
        self.bkg_index = self.obj_matrix.iloc[self.minima_bkg].min_index

        if self.bkg_index == 0:
            self.bkg_index = max(self.data.index)

        self.obj_operator.set_attr(attr='x0_index', num=0, limit=False)
        # self.obj_operator.set_attr(attr='x1_index', num=self.bkg_index, limit=False)
        self.obj_operator.set_attr(attr='bkg_index', num=self.bkg_index, limit=False)

        # forces range on low surf_conc end and recalculates
        self.error_calc(get_best=False)

    def set_surf(self, **kwargs):
        """
        Calculate.

        generic discription
        """
        if not hasattr(self, 'obj_matrix'):
            self.gen_matrix()

        self.error_calc(use_index=False, x_in=0, reverse=True, **kwargs)

        self.minima_surf = np.unravel_index(
            self.error_matrix.to_numpy(na_value=np.inf).argmin(), self.error_matrix.shape)

        # finds the end of the 1st profile
        self.surf_index = self.obj_matrix.iloc[self.minima_surf].min_index

        if self.surf_index == 0:
            self.surf_index = self.min_range

        self.obj_operator.set_attr(attr='x0_index', num=self.surf_index, limit=False)

        # forces range on low surf_conc end and recalculates
        self.error_calc(get_best=False)

    def find_ranges(self, method='fst_fwd', run_bkg=False, **kwargs):
        """
        Calculate.

        generic discription
        """
        if not hasattr(self, 'obj_matrix'):
            self.gen_matrix()

        if not hasattr(self, 'bkg_index') and run_bkg:
            self.set_bkg(**kwargs)

        self.min_array = np.empty((0, 2), int)

        if method == 'fst_fwd':
            self.fwd(**kwargs)
        elif method == 'slw_fwd':
            self.fwd(fast=False, **kwargs)
        elif method == 'fst_rev':
            self.rev(**kwargs)
        elif method == 'slw_rev':
            self.rev(fast=False, **kwargs)



    def fwd(self, fast=True, full_range=False, **kwargs):
        """
        Calculate.

        generic discription
        """
        if not full_range:
            if not hasattr(self, 'bkg_index'):
                self.set_bkg(**kwargs)
            start = self.bkg_index
            stop = self.sims.argmax()
            min_loc = self.minima_bkg  # tuple
        else:
            start = max(self.data.index)
            stop = self.sims.argmax()
            min_loc = (self.rows.argmax(), self.cols.argmin())

        self.min_array = np.append(self.min_array, [np.array(self.minima_bkg)], axis=0)
        slow_range = np.array(range(start-1, stop-1, -1))
        iterations = 0
        index_temp = start
        self.obj_operator.set_attr(attr='x0_index', num=index_temp, limit=False)
        while index_temp > stop and iterations < start:

            # run the fitting program for the new range of surf_conc --> no output
            # self.obj_matrix.applymap(lambda x: self.obj_operator.drop(x)
            #                               if x.diff > self.rows[min_loc[0]]
            #                               and x.surf_conc < self.cols[min_loc[1]] else None)
            held_index_temp = index_temp
            # self.obj_operator.set_attr(attr='x1_index', num=index_temp, limit=True)
            self.error_calc(use_index=False, x_in=index_temp, limit=True, reset=True, **kwargs)

            # generate a temporary error array above prev best--> df matrix
            err_temp = self.obj_matrix.applymap(
                lambda x: x.error if x.diff < self.rows[min_loc[0]]
                and x.surf_conc > self.cols[min_loc[1]] else 1)

            # find indexes (diff&surf_conc) of minimum value in range just tested --> tuple
            min_loc = np.unravel_index(np.array(err_temp).argmin(), np.array(err_temp).shape)
            # self.obj_matrix.iloc[min_loc].x0_index = self.obj_matrix.iloc[min_loc].min_index

            self.min_array = np.append(self.min_array, [np.array(min_loc)], axis=0)
            iterations += 1

            self.obj_matrix.applymap(lambda x: self.obj_operator.drop(x)
                              if x.diff > self.rows[min_loc[0]]
                              and x.surf_conc < self.cols[min_loc[1]] else None)
            if fast:
                # get the x0 index of the minima location, only for data location
                self.obj_operator.set_attr(attr='x1_index', num=index_temp, limit=True)
                index_temp = self.obj_matrix.iloc[min_loc].min_index
                self.obj_operator.set_attr(attr='x0_index', num=index_temp, limit=True)
                # self.obj_matrix.applymap(lambda x: setattr(x, 'x0_index', x.min_index)
                #                               if x.min_index < x.x1_index else None)

                # self.obj_matrix.iloc[min_loc].x0_index = self.obj_matrix.iloc[min_loc].min_index
            else:
                index_temp = slow_range[iterations]
        # self.obj_matrix.applymap(lambda x: setattr(x, 'x0_index', x.min_index)
        #                               if x.min_index < x.x1_index else None)


    def rev(self, fast=True, full_range=True, **kwargs):
        """
        Calculate.

        generic discription
        """
        if not full_range:
            if not hasattr(self, 'bkg_index'):
                self.set_bkg(**kwargs)
            if not hasattr(self, 'minima_surf'):
                self.set_surf(**kwargs)
            start = self.sims.argmax()
            stop = self.bkg_index
            min_loc = self.minima_surf  # tuple
        else:
            start = self.sims.argmax()
            stop = max(self.data.index)
            min_loc = (self.rows.argmin(), self.cols.argmax())

        slow_range = np.array(range(start-1, stop-1))
        iterations = 0
        index_temp = start
        while index_temp < stop and iterations < stop:
            # run the fitting program for the new range of surf_conc --> no output
            self.obj_matrix.applymap(lambda x: self.obj_operator.drop(x)
                                          if x.diff < self.rows[min_loc[0]]
                                          and x.surf_conc > self.cols[min_loc[1]] else None)
            self.error_calc(use_index=False, x_in=index_temp, reverse=True, limit=True, **kwargs)

            # generate a temporary error array above prev best--> df matrix
            err_temp = self.obj_matrix.applymap(
                lambda x: x.error if x.diff > self.rows[min_loc[0]]
                and x.surf_conc < self.cols[min_loc[1]] else 1)

            # find indexes (diff&surf_conc) of minimum value in range just tested --> tuple
            min_loc = np.unravel_index(np.array(err_temp).argmin(), np.array(err_temp).shape)
            self.fit_curves = min_loc

            iterations += 1
            if fast:
                # get the x0 index of the minima location, only for data location
                index_temp = self.obj_matrix.iloc[min_loc].min_index
                self.obj_operator.set_attr(attr='x0_index', num=index_temp, limit=True)
            else:
                index_temp = slow_range[iterations]
            # set x1 to new value in upcoming round prior to analysis


class Analysis:
    """
    Treat initial data.

    Imports
    """

    def __init__(self, obj_matrix, **kwargs):
        """
        Calculate.

        generic discription  cols = [15,21], rows = [-17,-11]
        """
        if str(type(obj_matrix)) == "<class '__main__.Matrix_Ops'>":
            self.obj = obj_matrix
            obj_matrix =  self.obj.obj_matrix

        self.obj_matrix = obj_matrix

    def map_plot(self, info='error', name=None, matrix=None, nan=1, zlog=True, **kwargs):
        """
        Calculate.

        generic discription
        """
        if matrix is None:
            to_plot = self.get_matrix(info, nan)
        else:
            to_plot = matrix

        if self.obj_matrix.iloc[1,0].__class__.__name__ == 'FitProfile':
            if name is None:
                name = 'Fit Profile'

            plt_kwargs = {'name': name, 'xname': "Start Point (um)",
                          'yname': "End Point (um)", 'zname': "Error"}
            plt_kwargs.update({key: kwargs[key] for key in kwargs})
            if zlog:
                gf.log_map(to_plot.columns*1e4, to_plot.index*1e4,
                           to_plot.to_numpy(), **plt_kwargs)
            else:
                gf.lin_map(to_plot.columns*1e4, to_plot.index*1e4,
                           to_plot.to_numpy(), **plt_kwargs)

            plt.show()

        if self.obj_matrix.iloc[1,0].__class__.__name__ == 'PredProfile':
            if name is None:
                name = 'Pred Profile'

            plt_kwargs = {'name': name, 'xname': 'Surface Concentration (cm-3)',
                          'yname': 'Diffusivity (cm2/s)', 'zname': "Error"}
            plt_kwargs.update({key: kwargs[key] for key in kwargs})
            if zlog:
                gf.log_map(to_plot.columns, to_plot.index, to_plot.to_numpy(),
                           logs='both', **plt_kwargs)
            else:
                gf.lin_map(to_plot.columns, to_plot.index, to_plot.to_numpy(),
                           logs='both', **plt_kwargs)

            plt.show()



    def get_single(self, loc, attr='pred'):
        """
        Calculate.

        generic discription
        """
        if not isinstance(loc[0], (int, np.int)):
            x = gf.find_nearest(self.obj_matrix.columns, loc[0])
        else:
            x = loc[0]

        if not isinstance(loc[1], (int, np.int)):
            y = gf.find_nearest(self.obj_matrix.rows, loc[1])
        else:
            y = loc[1]

        try:
            res = getattr(self.obj_matrix.iloc[x, y], attr)
        except AttributeError:
            print('Not an attribute')
            res = None

        return res

    def get_matrix(self, attr='x0_index', nan = 1):
        """
        Calculate.

        generic discription
        """

        # func = self.obj_matrix.applymap(lambda x: getattr(x, attr)
        #                                 if not isinstance(x, (int, np.int)) else None)
        if hasattr(self, 'obj_matrix'):
            return self.obj_matrix.applymap(lambda x: getattr(x, attr)
                                            if not isinstance(x, (int, np.int)) else nan)
        else:
            return np.ones((self.obj_matrix.size, self.obj_matrix.size))

    def check_error(self, **kwargs):
        """
        Calculate.

        generic discription
        """
        self.error_matrix = self.obj_matrix.applymap(lambda x: ProfileOps(x.data, x.pred,x.x0_index, x.x1_index) if not isinstance(x, (int, np.int)) else 1)
        # self.error_operator = Composite()
        # self.error_matrix.applymap(lambda x: self.error_operator.add(x)
        #                               if isinstance(x, (int, np.int)) else None)

        return self.error_matrix.applymap(lambda x: x.err(**kwargs)  if not isinstance(x, (int, np.int)) else 1)


    def residuals(self,loc, log_form=False, plot=True):

        self.pred = self.get_single(loc,'pred')
        self.data = self.get_single(loc,'data')

        if log_form:
            self.resids = np.log10(self.sims) - np.log10(self.pred)
            if plot:
                sns.residplot(x=np.log10(self.pred), y=np.log10(self.sims), lowess=True, color="g")

        else:
            self.resids = self.sims - self.pred
            if plot:
                sns.residplot(x=self.pred, y=self.sims, lowess=True, color="g")


        return self.resids

    def stitcher(self, fits):
        self.fits=fits

        if isinstance(self.fits, list):
            self.indexed = list()
            for row in reversed(range(len(self.fits))):

                x0 = self.fits[row][0]
                x1 = self.fits[row][1]
                profile = FitProfile(sample_data,start=x0, stop=x1).pred
                if x0 <= 1:
                    x0 = 0
                if x1 == 199:
                    x1 = 200
                [self.indexed.append(x) for x in profile[x0:x1]]
                if not hasattr(self, 'additive'):
                    self.additive = np.zeros_like(profile)
                self.additive += profile

        else:
            self.indexed = list()
            for row in reversed(range(len(self.fits))):
                profile = self.get_single(self.fits.loc[row,'Location'],'pred')
                x0 = self.fits.loc[row,'x0 index']
                x1 = self.fits.loc[row,'x1 index']
                if x0 <= 1:
                    x0 = 0
                if x1 == 199:
                    x1 = 200
                [self.indexed.append(x) for x in profile[x0:x1]]
                if not hasattr(self, 'additive'):
                    self.additive = np.zeros_like(profile)
                self.additive += profile



# %% Import data
mypath = "C:\\Users\\j2cle\\Work Docs\\Data\\Analysis\\SIMS\\"
figpath = 'C:\\Users\\j2cle\\Work Docs\\Data\\Analysis\\SIMS\\Fig_fits'
df_log = pd.read_excel(f'{mypath}/Sample Log for fitting.xlsx',
                       index_col=0, skiprows=1).dropna(axis=0, how='all')

if 0:
    df_log.drop(['R-60 SF ROI'], inplace=True)

sample_all = {x: DataProfile(df_log.loc[x, :]) for x in df_log.index}

sample_data = sample_all[df_log.index[18]]


# %% Initialize "Fit" matrix
by_x = Matrix_Ops(sample_data, 'FitProfile', 20, col=[0, 60], row=[0, 60], min_range=1)

# ProfileOps(instr='logic' or 'base' or 'none', use_sample_w=True or false)
by_x.error_calc(reset=True, get_best=False, to_log=False, instr='none', use_sample_w=False)
#%%
evaluation = Analysis(by_x)
error_updated = evaluation.check_error()

evaluation.map_plot(info='error',nan=1.01,z_limit=[1e-4, 1]) #
evaluation.map_plot(matrix=error_updated,nan=1.01,z_limit=[1e-4, 1])
evaluation.map_plot(info='ks_test',nan=1.1,zlog=False,z_limit=[1e-5, 1],zname='P-value')
evaluation.map_plot(info='shap_test',nan=1.1,zlog=False,z_limit=[0, 1],zname='P-value')
evaluation.map_plot(info='diff',nan=1,z_limit=[1e-16, 1e-12],zname='diff',levels=60)
evaluation.map_plot(info='surf_conc',nan=1,z_limit=[1e16, 1e19],zname='surf_conc',levels=60)



# %% Initialize "Pred" matrix
by_D = Matrix_Ops(sample_data, 'PredProfile', 50)
# %% run pred matrix analysis
by_D.set_bkg(to_log=False, instr='none', use_sample_w=False)
# ProfileOps(instr='logic' or 'base' or 'none', use_sample_w=True or false)
# by_D.error_calc(run_bkg=True, reset=False, get_best=True, to_log=True,
#                 instr='logic', use_sample_w=True)

by_D.find_ranges(method='fst_fwd', get_best=True, to_log=False,instr='logic', use_sample_w=True)
# name=name, xname="Start Point (um)", yname="End Point (um)",
# zname="Error", logs="", levels=50, x_limit=[0, 0], y_limit=[0, 0], z_limit=[0, 0]

by_D_error = by_D.error_matrix
by_D_obj = by_D.obj_matrix
by_D_fits = by_D.fit_curves
# by_D.plot(name='Ranged Error: raw', z_limit=[1e-4, 1])

# %%
test = Analysis(by_D_obj)

test.stitcher(by_D_fits)
comb_fit = np.array(test.indexed)
raw_data = test.get_single((0,0), attr='data')['Na'].to_numpy()
raw_depth = test.get_single((0,0), attr='data')['Depth'].to_numpy()*1e4
resids = np.log10(raw_data) - np.log10(comb_fit)
sns.scatterplot(x=raw_depth, y=resids)
stats.ks_2samp(raw_data, comb_fit)


#%%
test = Analysis(by_D_obj)
man = [(72,199),(34,72),(1,34)]
test.stitcher(man)
comb_fit = np.array(test.indexed)
raw_data = test.get_single((0,0), attr='data')['Na'].to_numpy()
raw_depth = test.get_single((0,0), attr='data')['Depth'].to_numpy()*1e4
resids = np.log10(raw_data) - np.log10(comb_fit)
sns.scatterplot(x=raw_depth, y=resids)
stats.ks_2samp(raw_data, comb_fit)
# sns.scatterplot(x= np.log10(comb_fit), y=resids)

# resid_data = test.residuals((4,5))
# resid_data1 = test.residuals((7,12), log_form = False)
# resid_data2 = test.residuals((7,12), log_form = True)

# plot=pd.DataFrame(resid_data2)
# test_data = test.data['Depth'].to_numpy()
# sns.scatterplot(x=test_data*1e4,y=resid_data2)

# depth_to_resid = pd.DataFrame({'Depth':test.data['Depth'].to_numpy()*1e4,'Real':np.log10(test.data['Na'].to_numpy()),'Pred':np.log10(test.pred),'Residuals':test.residuals((7,12), log_form = True, plot=False)})
# sns.lmplot(x="Pred", y="Residuals", data=depth_to_resid.iloc[1:50,:])
# sns.residplot(x=(depth_to_resid.iloc[1:37,2]),y=depth_to_resid.iloc[1:37,3], lowess=True, color="g")
# sns.scatterplot(x=(depth_to_resid.iloc[:,0]),y=(depth_to_resid.iloc[:,3]))

# from matplotlib import pyplot
# import seaborn

# import probscale
# probscale.probplot(depth_to_resid['Real'], plottype='pp', datascale='log',problabel='Percentile', datalabel='Total Bill (USD)', scatter_kws=dict(marker='.', linestyle='none', label='Bill Amount'))
