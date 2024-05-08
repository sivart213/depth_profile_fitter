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

from scipy.special import erfc
# from scipy import stats
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


def c_np_array(depth, diff, c_0, thick, temp, e_app, time):
    """
    Calculate.

    generic discription
    """
    mob = diff/(gf.KB_EV*temp)
    term_B = erfc(-mob*e_app*time/(2*np.sqrt(diff*time)))
    return (c_0/(2*term_B)) * (erfc((depth - mob*e_app * time)/(2*np.sqrt(diff * time))) +
                               erfc(-(depth-2*thick + mob*e_app*time)/(2*np.sqrt(diff*time))))


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

def get_single(obj,loc,attr='pred'):
    matrix= obj.obj_matrix

    if not isinstance(loc[0],int):
        x = gf.find_nearest(matrix.columns, loc[0])
    else:
        x = loc[0]

    if not isinstance(loc[1],int):
        y = gf.find_nearest(matrix.columns, loc[1])
    else:
        y = loc[1]

    try:
        res = getattr(matrix.iloc[x,y],attr)
    except ValueError:
        print('Not an attribute')
        res = None

    return res

def get_matrix(obj,attr='x0_index'):
    """
    Calculate.

    generic discription
    """
    if hasattr(obj, 'obj_matrix'):
        return obj.obj_matrix.applymap(lambda x: getattr(x, attr)
                                            if not isinstance(x, int) else None)
    else:
        return np.ones((obj.size, obj.size))

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


class Profile_Data:
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


class Sim_Profile(Component):
    """
    Calculate.

    generic discription
    """

    def __init__(self, obj):
        self.true_prof = obj
        self.data = self.true_prof.data.copy()
        self.x0_index = 0
        self.x1_index = len(self.data['Na'])-1
        self.bkg_index = self.x1_index
        self.min_index = 0
        self.min_range = 2
        self.error = 1
        self.D = 1e-17
        self.C0 = 1e17
        self.min_index = 0
        self.matrix_loc = None

    @property
    def x1_index(self):
        """
        Calculate.

        generic discription
        """
        return self._x1_index

    @x1_index.setter
    def x1_index(self, value):
        if self.x0_index >= value:
            self.x0_index = 0
        if value > max((self.data['Na'].index)):
            value = max((self.data['Na'].index))
        self._x1_index = value

    @property
    def x0_loc(self):
        """
        Calculate.

        generic discription
        """
        return gf.fromcm(self.data.loc[self.x0_index, 'Depth'], 'um')

    @property
    def x1_loc(self):
        """
        Calculate.

        generic discription
        """
        return gf.fromcm(self.data.loc[self.x1_index, 'Depth'], 'um')

    @property
    def limited_depth(self):
        """
        Calculate.

        generic discription
        """
        return self.data['Depth'].to_numpy()[self.x0_index:self.x1_index+1]

    @property
    def limited_conc(self):
        """
        Calculate.

        generic discription
        """
        return self.pred[self.x0_index:self.x1_index+1]

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

    def set_error(self, save_res=True, **kwargs):
        """
        Calculate.

        generic discription.
        err(self, instr='None', use_sample_w=False, w_array=None, to_log=False, **kwargs):
        """
        if not hasattr(self, '_err_func_to_use'):
            self._err_func_to_use = getattr(
                Error(self.data, self.pred, x0=self.x0_index, x1=self.x1_index), 'err')
        err_temp = self._err_func_to_use(**kwargs)
        if save_res:
            self.error = err_temp

        return err_temp

    def set_best_error(self, use_index=True, x_in=-1, reset=False, reverse=False, **kwargs):
        """
        Calculate.

        generic discription
        set_error(self, save_res=True, **kwargs):
        err(self, instr='None', use_sample_w=False, w_array=None, to_log=False, **kwargs):
        """
        if reset:
            self.error = 1

        err_last = self.error

        if not reverse:
            if use_index:
                x_in = self.x1_index
            elif x_in <= 0:
                x_in = len(self.data['Na'])-1

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


class Pred_Profile(Sim_Profile):
    """
    Calculate.

    generic discription
    """

    def __init__(self, true_prof, **kwargs):
        """
        Calculate.

        generic discription
        """
        # constant once set
        super().__init__(true_prof)
        self.unpack_kwargs(kwargs)

        self.pred = c_np(depth=self.data['Depth'].to_numpy(), diff=self.D,
                         c_0=self.C0, thick=self.true_prof.thick_cm,
                         temp=gf.CtoK(self.true_prof.params['Temp']),
                         e_app=self.true_prof.params['Volt']/self.true_prof.thick_cm,
                         time=self.true_prof.params['Stress Time'])

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


class Fit_Profile(Sim_Profile):
    """
    Calculate.

    generic discription
    """

    _curve_fit_keys = list(curve_fit.__code__.co_varnames) + ['x_scale', 'xtol', 'jac']

    def __init__(self, true_prof, start=0, stop=0, **kwargs):
        """
        Calculate.

        generic discription
        """
        super().__init__(true_prof)
        if type(start) is np.float64 or type(start) is float:
            start = gf.find_nearest(self.data['Depth'], start)
        if type(stop) is np.float64 or type(stop) is float:
            stop = gf.find_nearest(self.data['Depth'], stop)
        self.x0_index = start
        if stop != 0:
            self.x1_index = stop

        self.c_np_new = partial(c_np, thick=self.true_prof.thick_cm,
                                temp=gf.CtoK(self.true_prof.params['Temp']),
                                e_app=self.true_prof.params['Volt'] /
                                self.true_prof.thick_cm,
                                time=self.true_prof.params['Stress Time'])

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
                                self.data['Depth'].to_numpy()[self.x0_index:self.x1_index+1],
                                self.data['Na'].to_numpy()[self.x0_index:self.x1_index+1],
                                p0=(D_pred[1], C0_pred[1]),
                                bounds=((D_pred[0], C0_pred[0]), (D_pred[2], C0_pred[2])),
                                **self.curve_fit_kwargs)
        except RuntimeError:
            self.fit_res = [self.D, self.D, self.C0, self.C0]
            print(self.x0_index, '-', self.x1_index)

        else:
            self.fit_res = [10**fittemp[0][0],
                            (10**(fittemp[0][0] + np.sqrt(np.diag(fittemp[1]))[0]) -
                             10**(fittemp[0][0]-np.sqrt(np.diag(fittemp[1]))[0]))/2,
                            10**fittemp[0][1],
                            (10**(fittemp[0][1]+np.sqrt(np.diag(fittemp[1]))[1]) -
                             10**(fittemp[0][1]-np.sqrt(np.diag(fittemp[1]))[1]))/2]
        self.D = self.fit_res[0]
        self.C0 = self.fit_res[2]

    @property
    def pred(self):
        """
        Calculate.

        call and use the private variable
        """
        self._pred = np.array(self.c_np_new(self.data['Depth'].to_numpy(), self.D, self.C0))
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


class Error:
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
        self.w_range = len(self.data['Na'])

        if instr.lower() == 'logic':
            vals_incl = len(self.data['Na'].to_numpy()[self.x0:self.x1+1]
                            [(self.data['Na'].to_numpy() > self.pred)[self.x0:self.x1+1]])
        elif instr.lower() == 'base':
            vals_incl = len(self.data['Na'].to_numpy()[self.x0:self.x1+1])
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
            self.data['Na'] = np.log10(self.data['Na'].to_numpy())
            self.pred = np.log10(self.pred)

        self.unpack_kwargs(kwargs)

        self.w_constant = str(instr)
        if use_sample_w:
            self.sample_weights(w_array)

        self.error = (MAPE(self.data['Na'].to_numpy()[self.x0:self.x1+1],
                           self.pred[self.x0:self.x1+1],
                           **self.mape_kwargs) * self.w_constant)

        return self.error


class Matrix_Ops:
    """
    Treat initial data.

    Imports
    """

    def __init__(self, true_prof, cls_type, size=100, col=[], row=[], min_range=2, **kwargs):
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

        if self.cls_type == 'Pred_Profile':
            if len(col) == 0:
                col = [15, 21]
                row = [-17, -11]
            self.cols = np.logspace(col[0], col[1], self.size)
            self.rows = np.logspace(row[0], row[1], self.size)
        elif self.cls_type == 'Fit_Profile':
            if len(col) == 0:
                col = [0, self.data['Depth'].to_numpy()[-1]]
                row = [0, self.data['Depth'].to_numpy()[-1]]
            elif type(col[1]) is int:
                col = [self.data.loc[col[0], 'Depth'], self.data.loc[col[1], 'Depth']]
                row = [self.data.loc[col[0], 'Depth'], self.data.loc[row[1], 'Depth']]
            self.cols = np.linspace(col[0], col[1], self.size)
            self.rows = np.linspace(row[0], row[1], self.size)
            # self.cols = range(cols[0], cols[1])
            # self.rows = range(rows[0], rows[1])
        else:
            print('There is an error')

    @property
    def error_matrix(self):
        """
        Calculate.

        generic discription
        """
        if hasattr(self, 'obj_matrix'):
            return self.obj_matrix.applymap(lambda x: x.error
                                                 if type(x) is not int else 1)
        else:
            return np.ones((self.size, self.size))

    @property
    def x0_matrix(self):
        """
        Calculate.

        generic discription
        """
        if hasattr(self, 'obj_matrix'):
            return self.obj_matrix.applymap(lambda x: x.x0_index
                                                 if type(x) is not int else None)
        else:
            return np.ones((self.size, self.size))

    @property
    def x1_matrix(self):
        """
        Calculate.

        generic discription
        """
        if hasattr(self, 'obj_matrix'):
            return self.obj_matrix.applymap(lambda x: x.x1_index
                                                 if type(x) is not int else None)
        else:
            return np.ones((self.size, self.size))

    @property
    def min_matrix(self):
        """
        Calculate.

        generic discription
        """
        if hasattr(self, 'obj_matrix'):
            return self.obj_matrix.applymap(lambda x: x.min_index
                                                 if type(x) is not int else None)
        else:
            return np.ones((self.size, self.size))

    @property
    def curve_matrix(self):
        """
        Calculate.

        generic discription
        """
        if hasattr(self, 'obj_matrix'):
            return self.obj_matrix.applymap(lambda x: x.pred if type(x) is not int else None)
        else:
            return np.ones((self.size, self.size))

    @property
    def fit_curves(self):
        """
        Calculate.

        generic discription
        """
        col = ['error', 'D', 'C0', 'x0 index', 'x1 index',
               'range (points)', 'x0', 'x1', 'range (um)', 'Depth (range)', 'Na (range)',
               'Na (all)']
        if not hasattr(self, '_fit_curves_array'):
            self._fit_curves_array = np.empty((0, len(col)), int)
        return pd.DataFrame(self._fit_curves_array, columns=col)

    @fit_curves.setter
    def fit_curves(self, min_loc):
        if len(min_loc) != 2 or type(min_loc) is not tuple:
            min_loc = (0, 0)
        _fit_curves_list = [
            self.obj_matrix.iloc[min_loc].error,
            self.obj_matrix.iloc[min_loc].D,
            self.obj_matrix.iloc[min_loc].C0,
            self.obj_matrix.iloc[min_loc].x0_index,
            self.obj_matrix.iloc[min_loc].x1_index,
            self.obj_matrix.iloc[min_loc].index_range,
            self.obj_matrix.iloc[min_loc].x0_loc,
            self.obj_matrix.iloc[min_loc].x1_loc,
            self.obj_matrix.iloc[min_loc].depth_range,
            self.obj_matrix.iloc[min_loc].limited_depth,
            self.obj_matrix.iloc[min_loc].limited_conc,
            self.obj_matrix.iloc[min_loc].data['Na'].to_numpy()]

        if not hasattr(self, '_fit_curves_array'):
            self._fit_curves_array = np.empty((0, len(_fit_curves_list)), int)
        self._fit_curves_array = np.append(
            self._fit_curves_array, np.atleast_2d(np.array(_fit_curves_list)), axis=0)

    def gen_matrix(self, **kwargs):
        """
        Calculate.

        generic discription
        """
        if self.cls_type == 'Pred_Profile':
            self.obj_matrix = pd.DataFrame([[
                Pred_Profile(self.true_prof, D=y, C0=x, **kwargs) for x in self.cols]
                for y in self.rows], columns=self.cols, index=self.rows)
        elif self.cls_type == 'Fit_Profile':
            self.obj_matrix = pd.DataFrame([[
                Fit_Profile(self.true_prof, start=x, stop=y, **kwargs) if (x < y)
                else 1 for x in self.cols] for y in self.rows],
                columns=self.cols, index=self.rows)
        else:
            print('There is an error')

        self.obj_operator = Composite()
        self.obj_matrix.applymap(lambda x: self.obj_operator.add(x)
                                      if type(x) is not int else None)
        if self.obj_operator._family[0].min_range != self.min_range:
            self.obj_operator.set_attr(attr='min_range', num=self.min_range, limit=False)

    def error_calc(self, reset=False, get_best=True, **kwargs):
        """
        Calculate.

        generic discription
        """
        if not hasattr(self, 'obj_matrix'):
            self.gen_matrix(**kwargs)
        if reset:
            self.obj_operator.set_attr(attr='error', num=1, limit=False)
            self.obj_operator.del_attr(attr='_err_func_to_use', limit=False)

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
            self.obj_matrix.to_numpy(na_value=np.inf).argmin(), self.obj_matrix.shape)

        # finds the start of bkgrnd --> returns int
        self.bkg_index = self.obj_matrix.iloc[self.minima_bkg].min_index

        if self.bkg_index == 0:
            self.bkg_index = len(self.data['Na'])-1

        self.obj_operator.set_attr(attr='x0_index', num=0, limit=False)
        self.obj_operator.set_attr(attr='x1_index', num=self.bkg_index, limit=False)
        self.obj_operator.set_attr(attr='bkg_index', num=self.bkg_index, limit=False)

        # forces range on low C0 end and recalculates
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
            self.obj_matrix.to_numpy(na_value=np.inf).argmin(), self.obj_matrix.shape)

        # finds the end of the 1st profile
        self.surf_index = self.obj_matrix.iloc[self.minima_surf].min_index

        if self.surf_index == 0:
            self.surf_index = self.min_range

        self.obj_operator.set_attr(attr='x0_index', num=self.surf_index, limit=False)

        # forces range on low C0 end and recalculates
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

        if method == 'fst_fwd':
            self.fwd(**kwargs)
        elif method == 'slw_fwd':
            self.fwd(fast=False, **kwargs)
        elif method == 'fst_rev':
            self.rev(**kwargs)
        elif method == 'slw_rev':
            self.rev(fast=False, **kwargs)

        self.min_array = np.empty((0, 2), int)

    def fwd(self, fast=True, full_range=False, **kwargs):
        """
        Calculate.

        generic discription
        """
        if not full_range:
            if not hasattr(self, 'bkg_index'):
                self.set_bkg(**kwargs)
            start = self.bkg_index
            stop = np.array(self.data['Na']).argmax()
            min_loc = self.minima_bkg  # tuple
        else:
            start = max((self.data['Na'].index))
            stop = np.array(self.data['Na']).argmax()
            min_loc = (self.rows.argmax(), self.cols.argmin())

        self.min_array = np.append(self.min_array, [np.array(self.set_bkg)], axis=0)
        slow_range = np.array(range(start-1, stop-1, -1))
        iterations = 0
        index_temp = start

        while index_temp > stop and iterations < start:
            # run the fitting program for the new range of C0 --> no output
            self.obj_matrix.applymap(lambda x: self.obj_operator.drop(x)
                                          if x.D > self.rows[min_loc[0]]
                                          and x.C0 < self.cols[min_loc[1]] else None)
            self.error_calc(use_index=False, x_in=index_temp, limit=True, **kwargs)

            # generate a temporary error array above prev best--> df matrix
            err_temp = self.obj_matrix.applymap(
                lambda x: x.error if x.D < self.rows[min_loc[0]]
                and x.C0 > self.cols[min_loc[1]] else 1)

            # find indexes (D&C0) of minimum value in range just tested --> tuple
            min_loc = np.unravel_index(np.array(err_temp).argmin(), np.array(err_temp).shape)
            self.obj_matrix.iloc[min_loc].x0_index = self.obj_matrix.iloc[min_loc].min_index


            self.min_array = np.append(self.min_array, [np.array(min_loc)], axis=0)
            iterations += 1
            if fast:
                # get the x0 index of the minima location, only for data location
                index_temp = self.obj_matrix.iloc[min_loc].min_index
                self.obj_operator.set_attr(attr='x1_index', num=index_temp, limit=True)
                # self.obj_matrix.iloc[min_loc].x0_index = self.obj_matrix.iloc[min_loc].min_index
            else:
                index_temp = slow_range[iterations]
        self.obj_matrix.applymap(lambda x: setattr(x, 'x0_index', x.min_index)
                                      if x.min_index < x.x1_index else None)
        self.fit_curves = [x for x in self.min_array]

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
            start = np.array(self.data['Na']).argmax()
            stop = self.bkg_index
            min_loc = self.minima_surf  # tuple
        else:
            start = np.array(self.data['Na']).argmax()
            stop = max((self.data['Na'].index))
            min_loc = (self.rows.argmin(), self.cols.argmax())

        slow_range = np.array(range(start-1, stop-1))
        iterations = 0
        index_temp = start
        while index_temp < stop and iterations < stop:
            # run the fitting program for the new range of C0 --> no output
            self.obj_matrix.applymap(lambda x: self.obj_operator.drop(x)
                                          if x.D < self.rows[min_loc[0]]
                                          and x.C0 > self.cols[min_loc[1]] else None)
            self.error_calc(use_index=False, x_in=index_temp, reverse=True, limit=True, **kwargs)

            # generate a temporary error array above prev best--> df matrix
            err_temp = self.obj_matrix.applymap(
                lambda x: x.error if x.D > self.rows[min_loc[0]]
                and x.C0 < self.cols[min_loc[1]] else 1)

            # find indexes (D&C0) of minimum value in range just tested --> tuple
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

    def plot(self, info='error_matrix', name=None, **kwargs):
        """
        Calculate.

        generic discription
        """
        # fig, ax = plt.subplots()
        to_plot = getattr(self, info)

        if self.cls_type == 'Fit_Profile':
            if name is None:
                name = 'Fit_Profile'

            plt_kwargs = {'name': name, 'xname': "Start Point (um)",
                          'yname': "End Point (um)", 'zname': "Error"}
            plt_kwargs.update({key: kwargs[key] for key in kwargs})

            gf.Log_Map(to_plot.columns*1e4, to_plot.index*1e4, to_plot.to_numpy(), **plt_kwargs)


        if self.cls_type == 'Pred_Profile':
            if name is None:
                name = 'Pred_Profile'

            plt_kwargs = {'name': name, 'xname': 'C0 (cm-3)',
                          'yname': 'D (cm2/s)', 'zname': "Error"}
            plt_kwargs.update({key: kwargs[key] for key in kwargs})
            gf.Log_Map(to_plot.columns, to_plot.index, to_plot.to_numpy(),
                       logs='both', **plt_kwargs)


# %% Import data
mypath = "C:\\Users\\j2cle\\Work Docs\\Data\\Analysis\\SIMS\\"
figpath = 'C:\\Users\\j2cle\\Work Docs\\Data\\Analysis\\SIMS\\Fig_fits'
df_log = pd.read_excel(f'{mypath}/Sample Log for fitting.xlsx',
                       index_col=0, skiprows=1).dropna(axis=0, how='all')

if 0:
    df_log.drop(['R-60 SF ROI'], inplace=True)

sample_all = {x: Profile_Data(df_log.loc[x, :]) for x in df_log.index}

sample_data = sample_all[df_log.index[18]]


# # %% Initialize "Fit" matrix
# by_x = Matrix_Ops(sample_data, 'Fit_Profile', 100, col=[0, 60], row=[0, 60], min_range=1)
# #%%
# # error(instr='logic' or 'base' or 'none', use_sample_w=True or false)
# by_x.error_calc(reset=True, get_best=False, to_log=False, instr='logic', use_sample_w=False)
# by_x_error = by_x.error_matrix

# by_x.plot(name='Error: LT ratio', z_limit=[1e-4, 0.1])

# %% Initialize "Pred" matrix
by_D = Matrix_Ops(sample_data, 'Pred_Profile', 100)
# %%
by_D.set_bkg(to_log=False, instr='logic', use_sample_w=True)
# error(instr='logic' or 'base' or 'none', use_sample_w=True or false)
# by_D.error_calc(run_bkg=True, reset=False, get_best=True, to_log=True,
#                 instr='logic', use_sample_w=True)

by_D.find_ranges(method='fst_fwd', reset=False, get_best=True, to_log=True,
                 instr='logic', use_sample_w=True)
# name=name, xname="Start Point (um)", yname="End Point (um)",
# zname="Error", logs="", levels=50, x_limit=[0, 0], y_limit=[0, 0], z_limit=[0, 0]

by_D_error = by_D.error_matrix
by_D.plot(name='Ranged Error: raw', z_limit=[1e-4, 1])
