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
# import operator as op
# from matplotlib.colors import LogNorm
# import matplotlib.ticker as ticker
# from matplotlib.ticker import MaxNLocator
import seaborn as sns
import abc
# import statsmodels.api as sm
from scipy.special import erfc
from scipy import stats
from sklearn import metrics
from functools import partial

from scipy.optimize import curve_fit
import warnings


warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings("ignore")

sns.set_style('dark')


# %% Functions
def c_np(depth, diff, conc, thick, temp, e_app, time, log_form=False):
    """Return sum of squared errors (pred vs actual)."""
    if diff < 0:
        diff = 10**float(diff)
        conc = 10**float(conc)

    mob = diff/(gf.KB_EV*temp)
    term_B = erfc(-mob*e_app*time/(2*np.sqrt(diff*time)))
    if log_form:
        return np.log10((conc/(2*term_B)) * (erfc((depth - mob*e_app * time)/(2*np.sqrt(diff*time))) +
                                             erfc(-(depth-2*thick+mob*e_app*time)/(2*np.sqrt(diff*time)))))
    else:
        return (conc/(2*term_B)) * (erfc((depth - mob*e_app * time)/(2*np.sqrt(diff*time))) +
                                    erfc(-(depth-2*thick+mob*e_app*time)/(2*np.sqrt(diff*time))))


def linear(x, coeffs):
    """Return sum of squared errors (pred vs actual)."""
    return coeffs[1] + coeffs[0] * x


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


# %% Classes
class Component(metaclass=abc.ABCMeta):
    """Return sum of squared errors (pred vs actual)."""

    @abc.abstractmethod
    def set_error(self, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        pass

    @abc.abstractmethod
    def set_best_error(self, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        pass


class Composite(Component):
    """Return sum of squared errors (pred vs actual)."""

    _type = 'composite'

    def __init__(self, limit=False):
        self._children = list()
        self._family = list()
        self.limit = limit

    @property
    def chores(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.limit:
            self._worker = self._children
        else:
            self._worker = self._family

        return self._worker

    def set_error(self, limit=False, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        self.limit = limit
        for work in (self.chores):
            work.set_error(**kwargs)

    def set_best_error(self, limit=False, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        self.limit = limit
        for work in (self.chores):
            work.set_best_error(**kwargs)

    def set_attr(self, attr='start_index', val=0, limit=False, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        self.limit = limit
        for work in (self.chores):
            setattr(work.prof, attr, val)

    def get_attr(self, attr='error', limit=False, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        work_attr = list()
        self.limit = limit
        for work in (self.chores):
            work_attr.append(getattr(work.prof, attr))
        return work_attr

    def del_attr(self, attr='error', limit=False, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        self.limit = limit
        for work in (self.chores):
            if hasattr(work.prof, attr):
                delattr(work.prof, attr)

    def add(self, component):
        """Return sum of squared errors (pred vs actual)."""
        self._children.append(component)
        if component not in self._family:
            self._family.append(component)

    def drop(self, component):
        """Return sum of squared errors (pred vs actual)."""
        if component in self._children:
            self._children.remove(component)

    def get_prof(self, limit=False):
        """Return sum of squared errors (pred vs actual)."""
        work_prof = list()
        self.limit = limit
        for work in self.chores:
            work_prof.append(work.prof)
        return work_prof

    def gen_df(self, var=None, limit=False):
        """Return sum of squared errors (pred vs actual)."""
        self.limit = limit

        if var is None:
            var = ['start_index',
                   'stop_index',
                   'index_range',
                   'start_loc',
                   'stop_loc',
                   'diff',
                   'conc',
                   'error',
                   'stats',
                   ]
        if 'stats.' in var:
            var_loc = int(np.where(np.array(var) == 'stats.')[0][0])
            var[var_loc], attr = var[var_loc].split('.')
            self.set_attr('stats_attr', attr)

        listed = [[getattr(work.prof, x) for x in var] for work in self.chores]
        listed = [[x.to_dict() if isinstance(x, (pd.DataFrame, pd.Series))
                   else x for x in y] for y in listed]
        return pd.DataFrame(listed, columns=var)


class DataProfile:
    """Return sum of squared errors (pred vs actual)."""

    def __init__(self, slog, **kwargs):
        self.params = slog

        self.data_treatment()

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

        self.data_bgd = pd.Series()

        self.limit_test()
        # if 'tof' in self.params['Type'].lower():
        self.regress_test(**kwargs)

    def data_treatment(self):
        """Return sum of squared errors (pred vs actual)."""
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

    def limit_test(self, thresh=0.025):
        """Return sum of squared errors (pred vs actual)."""
        lin_loc, lin_lim = lin_test(
            self.data['Depth'].to_numpy(), self.data['Na'].to_numpy(), thresh)

        if lin_lim > self.fit_depth_cm*1.1 or lin_lim < self.fit_depth_cm*0.9:
            self.data_bgd['bgd_lim'] = gf.find_nearest(self.data['Depth'].to_numpy(),
                                                       self.fit_depth_cm)
        else:
            self.data_bgd['bgd_lim'] = lin_loc

    def regress_test(self, alpha=0.05, ind_range=10, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        stop = len(self.data['Depth'])
        cng = int(len(self.data['Depth'])*0.02)
        perc = 0
        while perc < 0.2 and stop > len(self.data['Depth'])*0.5:
            self.p = np.ones(stop-10)
            for x in range(stop-10):
                coeff = stats.linregress(self.data['Depth'].to_numpy()[x:stop],
                                         self.data['Na'].to_numpy()[x:stop], **kwargs)[:2]
                resid = (self.data['Na'].to_numpy()[x:stop] -
                         linear(self.data['Depth'].to_numpy()[x:stop], coeff))
                self.p[x] = stats.normaltest(resid)[1]
            stop -= cng
            perc = len(self.p[self.p > alpha])/len(self.p)

        itr = 0
        while self.p[itr] < alpha and itr < int((len(self.data['Na'])-10)*.75):
            itr += 1
        self.data_bgd['bgd_max'] = itr

        ind = 0
        while ind < ind_range and itr < int((len(self.data['Na'])-10)*.9):
            ind += 1
            itr += 1
            if self.p[itr] < alpha:
                ind = 0
        self.data_bgd['bgd_min'] = itr-ind

        self.data_bgd['bgd_ave'] = int((self.data_bgd['bgd_max']+self.data_bgd['bgd_min'])/2)
        coeff = stats.linregress(self.data['Depth'].to_numpy()[self.data_bgd['bgd_ave']:],
                                 self.data['Na'].to_numpy()[self.data_bgd['bgd_ave']:],
                                 **kwargs)[:2]
        self.data_bgd['P-value'] = self.p[self.data_bgd['bgd_ave']]
        self.data_bgd['slope'] = coeff[0]
        self.data_bgd['intercept'] = coeff[1]

    @property
    def thick_cm(self):
        """Return sum of squared errors (pred vs actual)."""
        return gf.tocm(self.params['Thick'], self.params['Thick unit'])


class BaseProfile:
    """
    Store Profile information.

    This class is intended to store the information inherent in to a depth
    profile fit. This should include the fit profile, it's properties, and
    its range.  Curently also includes the fitted data and the resulting error.
    I'm not sure that it is important for this fitted information to be in a
    single classs.
    """

    _type = 'base'

    def __init__(self, obj):
        self.min_range = 2
        self.min_index = 0
        self.diff = 1e-14
        self.conc = 1e18
        self.error = 1
        self.p_value = 0

        self.data_bgd = obj.data_bgd.copy()
        self.depth = obj.data['Depth'].to_numpy()
        self.sims = obj.data['Na'].to_numpy()
        self.pred = np.ones_like(self.sims)
        self.max_index = len(self.depth)-1
        self.bgd_index = self.data_bgd['bgd_ave']

        self.stats_obj = Stats(self.data[self.start_index:self.stop_index+1],)
        self.stats_attr = 'mean_abs_perc_err'

        self.std_values = pd.DataFrame({'conc': (15, 18, 21),
                                        'diff': (-17, -14, -11),
                                        'depth': (0, int(self.max_index/2), self.max_index)},
                                       index=('low', 'mid', 'high'))
        self.conditions = pd.Series({'thick': obj.thick_cm,
                                     'time': obj.params['Stress Time'],
                                     'temp': obj.params['Temp'],
                                     'e_field': obj.params['Volt']/obj.thick_cm,
                                     'volt': obj.params['Volt']})
        self.info = pd.Series({'ident': self.ident,
                               'sample': obj.params['Sample'],
                               'type': obj.params['Type'],
                               'class': 'BaseProfile',
                               'measurement': obj.params['Measurement']})

    @property
    def data(self):
        """Return sum of squared errors (pred vs actual)."""
        _data = pd.DataFrame(columns=('depth', 'SIMS', 'log(SIMS)', 'pred',
                                      'log(pred)', 'weight', 'residuals',
                                      'residuals from stats'))

        _data['depth'] = self.depth
        _data['SIMS'] = self.sims
        _data['log(SIMS)'] = np.log10(self.sims)
        _data['pred'] = self.pred
        _data['log(pred)'] = np.log10(self.pred)
        _data['weight'] = [self.pred[x]/self.sims[x] if self.pred[x] > self.sims[x]
                           else 1 for x in range(self.max_index+1)]
        _data_stats = Stats(_data)
        if hasattr(self, 'stats_obj'):
            _data_stats.log_form = self.stats_obj.log_form
            _data_stats.resid_type = self.stats_obj.resid_type

        _data['residuals'] = self.sims - self.pred
        _data['residuals from stats'] = _data_stats.residuals
        return _data

    @property
    def pred(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, '_pred'):
            self._pred = np.ones_like(self.sims)
        return np.where(self._pred <= 1e-30, 1e-30, self._pred)

    @pred.setter
    def pred(self, pred_in):
        self._pred = pred_in

    @property
    def ident(self):
        """Return sum of squared errors (pred vs actual)."""
        return id(self)

    @property
    def error_log(self):
        """Return sum of squared errors (pred vs actual)."""
        return self._error_log

    @error_log.setter
    def error_log(self, value):
        if not hasattr(self, '_error_log'):
            self._error_log = pd.DataFrame(columns=('diff', 'conc', 'start',
                                                    'stop', 'error', 'p-value'))
        if value in self._error_log.index:
            value = max(self._error_log.index) + 1
        self._error_log.loc[value, :] = (self.diff, self.conc, self.start_index,
                                         self.stop_index, self.error, self.p_value)

    @property
    def start_index(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, '_start_index'):
            self._start_index = 0
        return self._start_index

    @start_index.setter
    def start_index(self, value):
        if self._stop_index <= value or value < 0:
            print('Obj', self.ident, 'atempted to set start index to', value)
        else:
            self._start_index = int(value)

    @property
    def stop_index(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, '_stop_index'):
            self._stop_index = self.max_index
        return self._stop_index

    @stop_index.setter
    def stop_index(self, value):
        if self._start_index >= value or value > self.max_index:
            print('Obj', self.ident, 'atempted to set stop index to', value)
        else:
            self._stop_index = int(value)

    @property
    def diff(self):
        """Return sum of squared errors (pred vs actual)."""
        return gf.Sig_figs(self._diff, 4)

    @diff.setter
    def diff(self, value):
        self._diff = value

    @property
    def conc(self):
        """Return sum of squared errors (pred vs actual)."""
        return gf.Sig_figs(self._conc, 4)

    @conc.setter
    def conc(self, value):
        self._conc = value

    @property
    def start_loc(self):
        """Return sum of squared errors (pred vs actual)."""
        return gf.Sig_figs(gf.fromcm(self.depth[self.start_index], 'um'), 5)

    @property
    def stop_loc(self):
        """Return sum of squared errors (pred vs actual)."""
        return gf.Sig_figs(gf.fromcm(self.depth[self.stop_index], 'um'), 5)

    @property
    def index_range(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.stop_index-self.start_index+1

    @property
    def depth_range(self):
        """Return sum of squared errors (pred vs actual)."""
        return gf.Sig_figs((self.stop_loc-self.start_loc), 5)

    @property
    def stats(self):
        """Return sum of squared errors (pred vs actual)."""
        return getattr(self.stats_obj, self.stats_attr)

    @property
    def stats_settings(self):
        """Return sum of squared errors (pred vs actual)."""
        return self._stats_settings

    @stats_settings.setter
    def stats_settings(self, args):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, '_stats_settings'):
            self._stats_settings = vars(self.stats_obj)
        if (isinstance(args, (list, np.ndarray, tuple, dict)) and
                len(args) == 2 and args[0] in self._stats_settings):
            self._stats_settings[args[0]] = args[1]


class PredProfile(BaseProfile):
    """
    Generate profile from diff, conc, and simulation parameters.

    Creates a simulated profile by fitting real data.
    """

    _type = 'pred'

    def __init__(self, sims_obj, diff=None, conc=None, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        # constant once set
        super().__init__(sims_obj)

        if diff is not None:
            self.diff = diff
        if conc is not None:
            self.conc = conc

        self.unpack_kwargs(kwargs)

        self.pred = c_np(depth=self.depth, diff=self.diff,
                         conc=self.conc, thick=self.conditions['thick'],
                         temp=gf.CtoK(self.conditions['temp']),
                         e_app=self.conditions['e_field'],
                         time=self.conditions['time'])

        self.info['class'] = 'PredProfile'

        self.stats_obj = Stats(self.data[self.start_index:self.stop_index+1],)
        self.stats_attr = 'mean_abs_perc_err'

        self.error_log = 0

    def unpack_kwargs(self, kwargs):
        """Return sum of squared errors (pred vs actual)."""
        self.__dict__.update(kwargs)


class FitProfile(BaseProfile):
    """Return sum of squared errors (pred vs actual)."""

    _type = 'fit'
    _curve_fit_keys = list(curve_fit.__code__.co_varnames) + ['x_scale', 'xtol', 'jac']

    def __init__(self, sims_obj, start_index=None, stop_index=None, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        super().__init__(sims_obj)
        if start_index is not None:
            if isinstance(start_index, (float, np.float)):
                self.start_index = gf.find_nearest(self.depth, start_index)
            elif isinstance(start_index, (int, np.integer)):
                self.start_index = start_index
        if stop_index is not None:
            if isinstance(stop_index, (float, np.float)):
                self.stop_index = gf.find_nearest(self.depth, stop_index)
            elif isinstance(stop_index, (int, np.integer)):
                self.stop_index = stop_index

        self.curve_fit_kwargs = {'x_scale': 'jac', 'xtol': 1e-12, 'jac': '3-point'}
        self.unpack_kwargs(kwargs)

        self.fitter(**kwargs)

        self.info['class'] = 'FitProfile'

        self.stats_obj = Stats(self.data[self.start_index:self.stop_index+1],)
        self.stats_attr = 'mean_abs_perc_err'

        self.error_log = 0

    def unpack_kwargs(self, kwargs):
        """Return sum of squared errors (pred vs actual)."""
        self.curve_fit_kwargs.update({key: kwargs[key] for key in kwargs
                                      if key in self._curve_fit_keys})
        [kwargs.pop(x) for x in self._curve_fit_keys if x in kwargs.keys()]
        self.__dict__.update(kwargs)

    def fitter(self, diff_pred=None, conc_pred=None, log_form=False, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        if diff_pred is None:
            diff_pred = self.std_values['diff']
        if conc_pred is None:
            conc_pred = self.std_values['conc']

        self.unpack_kwargs(kwargs)

        self.c_np_new = partial(c_np, thick=self.conditions['thick'],
                                temp=gf.CtoK(self.conditions['temp']),
                                e_app=self.conditions['e_field'],
                                time=self.conditions['time'],
                                log_form=log_form)

        sims = self.sims
        if log_form:
            sims = self.data['log(SIMS)'].to_numpy()

        try:
            fittemp = curve_fit(self.c_np_new,
                                self.depth[self.start_index:self.stop_index+1],
                                sims[self.start_index:self.stop_index+1],
                                p0=(diff_pred['mid'], conc_pred['mid']),
                                bounds=((diff_pred['low'], conc_pred['low']),
                                        (diff_pred['high'], conc_pred['high'])),
                                **self.curve_fit_kwargs)
        except RuntimeError:
            self.fit_res = [self.diff, self.diff, self.conc, self.conc]
            print(self.start_index, '-', self.stop_index)

        else:
            self.fit_res = [10**fittemp[0][0],
                            (10**(fittemp[0][0] + np.sqrt(np.diag(fittemp[1]))[0]) -
                             10**(fittemp[0][0]-np.sqrt(np.diag(fittemp[1]))[0]))/2,
                            10**fittemp[0][1],
                            (10**(fittemp[0][1]+np.sqrt(np.diag(fittemp[1]))[1]) -
                             10**(fittemp[0][1]-np.sqrt(np.diag(fittemp[1]))[1]))/2]

        self.diff = self.fit_res[0]
        self.conc = self.fit_res[2]

        self.pred = np.array(self.c_np_new(self.depth, self.diff, self.conc, log_form=False))

    @property
    def diff_cov(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.fit_res[1]

    @property
    def surf_conc_cov(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.fit_res[3]


class ProfileOps(Component):
    """
    Calculate.

    requires Obj input, can create subclas with new init to do profiles directly if needed
    """

    _type = 'profile_operator'
    _error_keys = list(metrics.mean_absolute_percentage_error.__code__.co_varnames)

    def __init__(self, prof_obj, log_form=True, **kwargs):
        """
        Calculate.

        requires Obj input, can create subclas with new init to do profiles directly if needed
        """
        self.prof = prof_obj
        self.data = self.prof.data.copy()
        self.log_form = log_form

        self.unpack_kwargs(kwargs)

    def unpack_kwargs(self, kwargs):
        """Return sum of squared errors (pred vs actual)."""
        self.error_kwargs = {key: kwargs[key] for key in kwargs if key in self._error_keys}
        [kwargs.pop(x) for x in self._error_keys if x in kwargs.keys()]
        self.__dict__.update(kwargs)

    @property
    def ident(self):
        """Return sum of squared errors (pred vs actual)."""
        return id(self)

    @property
    def start(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, '_start'):
            self._start = 0
        return self._start

    @start.setter
    def start(self, value):
        if value is None:
            self._start = self.prof.start_index
        elif self._stop <= value or value < 0:
            print('Obj', self.ident, 'atempted to set start to', value)
        else:
            self._start = int(value)

    @property
    def stop(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, '_stop'):
            self._stop = self.max_index
        return self._stop

    @stop.setter
    def stop(self, value):
        if value is None:
            self._stop = self.prof.stop_index
        elif self._start >= value or value > self.prof.max_index:
            print('Obj', self.ident, 'atempted to set stop to', value)
        else:
            self._stop = int(value)

    @property
    def w_constant(self):
        """Calculate constant, may shift to use the depth range instead."""
        return self._w_constant

    @w_constant.setter
    def w_constant(self, instr='logic'):
        self.w_range = len(self.prof.sims)

        if instr.lower() == 'logic':
            vals_incl = len(self.prof.sims[self.start:self.stop+1]
                            [(self.prof.sims > self.prof.pred)[self.start:self.stop+1]])
        elif instr.lower() == 'base':
            vals_incl = len(self.prof.sims[self.start:self.stop+1])
        else:
            vals_incl = self.w_range

        if vals_incl <= self.prof.min_range:
            vals_incl = self.prof.min_range

        self._w_constant = self.w_range/(vals_incl)

    def set_error(self, start=None, stop=None, save_res=True, instr='None',
                  use_sample_w=False, w_array=None, log_form=False, **kwargs):
        """
        Calculate error.

        error for the input information, information generated at call,
        requires type input for constant, can pass the information to sample
        weights if desired. can rewrite to always pass sample_weights via
        kwargs.
        """
        # if log_form:
        #     sims = self.data['log(SIMS)'].to_numpy()
        #     pred = self.data['log(pred)'].to_numpy()
        # else:
        #     sims = self.data['SIMS'].to_numpy()
        #     pred = self.data['pred'].to_numpy()

        self.start = start
        self.stop = stop

        self.unpack_kwargs(kwargs)

        if use_sample_w and w_array is None:
            self.error_kwargs['sample_weight'] = (self.data['weight'].to_numpy()
                                                  [self.start:self.stop+1])

        self.w_constant = str(instr)

        self.ops_stats = Stats(self.data[self.start:self.stop+1],
                               log_form, **self.error_kwargs)

        self.error = self.ops_stats.mean_abs_perc_err * self.w_constant

        # self.error = (MAPE(sims[self.start:self.stop+1],
        #                    pred[self.start:self.stop+1],
        #                    **self.error_kwargs) * self.w_constant)

        if save_res:
            self.prof.error = self.error

        self.prof.error_log = 0

        return self.error

    def set_best_error(self, use_index=True, x_in=-1, reset=True, reverse=False,
                       save_res=True, **kwargs):
        """
        Calculate.

        generic discription
        set_error(self, save_res=True, **kwargs):
        err(self, instr='None', use_sample_w=False, w_array=None, log_form=False, **kwargs):
        """
        if reset:
            err_last = 1
        else:
            err_last = self.prof.error

        if use_index:
            x_in = self.prof.stop_index
        elif x_in <= 0:
            x_in = self.prof.max_index

        err_array = np.array([self.set_error(start=x, stop=x_in, save_res=False, **kwargs)
                              for x in range(x_in-self.prof.min_range+1)])

        if np.min(err_array) < err_last:
            self.prof.min_index = np.argmin(err_array)
            err_last = self.error
            self.error = np.min(err_array)

        if save_res:
            self.prof.error = self.error

        self.prof.error_log = 0

        return self.error

    # def ks_test(self, start=None, stop=None):
    #     """
    #     Calculate.

    #     generic discription
    #     """
    #     self.start = start
    #     self.stop = stop

    #     self.ks_stat, self.ks_p = stats.ks_2samp(self.prof.pred[self.start:self.stop],
    #                                              self.prof.sims[self.start:self.stop])

    # def shap_test(self, resid_type='residuals', start=None, stop=None):
    #     """
    #     Calculate.

    #     generic discription
    #     """
    #     self.start = start
    #     self.stop = stop

    #     try:
    #         self.shap_stat, self.shap_p = stats.shapiro(self.data[resid_type].to_numpy()
    #                                                     [self.start:self.stop])
    #     except (ValueError, TypeError):
    #         self.shap_p = 0
    #         self.shap_stat = 1


class Stats:
    """Return sum of squared errors (pred vs actual)."""

    def __init__(self, depth_df=None, log_form=False, resid_type='base',
                 depth=None, meas=None, pred=None, **kwargs):

        self.log_form = log_form
        self.resid_type = resid_type

        if depth_df is not None:
            depth = depth_df['depth'].to_numpy(copy=True)
            meas = depth_df['SIMS'].to_numpy(copy=True)
            pred = depth_df['pred'].to_numpy(copy=True)

        self.depth = depth
        self.meas = meas
        self.pred = pred

        self.kwargs = kwargs

    def lin_reg(self, x=None, y=None, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        if x is None:
            x = self.depth
        if y is None:
            y = self.meas
        self.reg_res = stats.linregress(x, y, **kwargs)
        return (self.reg_res[1], self.reg_res[0])

    @property
    def meas(self):
        """Return SIMS data in log or normal form."""
        if self.log_form and self._meas.min() > 25:
            return np.log10(self._meas)
        else:
            return self._meas

    @meas.setter
    def meas(self, value):
        """Set SIMS data."""
        self._meas = value

    @property
    def pred(self):
        """Return predicted data in log or normal form."""
        if self.log_form and self._pred.min() > 0 and self._pred.max() > 25:
            return np.log10(self._pred)
        else:
            return self._pred

    @pred.setter
    def pred(self, value):
        """Set predicted data."""
        self._pred = value

    @property
    def dft(self):
        """Return degrees of freedom population dep. variable variance."""
        return self.depth.shape[0] - 1

    @property
    def dfe(self):
        """Return degrees of freedom population error variance."""
        return self.depth.shape[0]

    @property
    def sse(self):
        """Return sum of squared errors (pred vs actual)."""
        return np.sum((self.meas - self.pred) ** 2)

    @property
    def sst(self):
        """Return total sum of squared errors (actual vs avg(actual))."""
        return np.sum((self.meas - np.mean(self.meas)) ** 2)

    @property
    def r_squared(self):
        """Return calculated value of r^2."""
        return 1 - self.sse/self.sst

    @property
    def adj_r_squared(self):
        """Return calculated value of adjusted r^2."""
        return 1 - (self.sse/self.dfe) / (self.sst/self.dft)

    @property
    def residuals(self):
        """Return calculated external standardized residual.."""
        if 'int' in self.resid_type.lower():
            return self.int_std_res
        elif 'ext' in self.resid_type.lower():
            return self.ext_std_res
        else:
            return self.meas - self.pred

    @property
    def int_std_res(self):
        """Return calculated internal standardized residual."""
        n = len(self.depth)
        diff_mean_sqr = np.dot((self.depth - np.mean(self.depth)),
                               (self.depth - np.mean(self.depth)))
        h_ii = (self.depth - np.mean(self.depth)) ** 2 / diff_mean_sqr + (1 / n)
        Var_e = np.sqrt(sum((self.meas - self.pred) ** 2)/(n-2))
        return (self.meas - self.pred)/(Var_e*((1-h_ii) ** 0.5))

    @property
    def ext_std_res(self):
        """Return calculated external standardized residual.."""
        r = self.int_std_res
        n = len(r)
        return [r_i*np.sqrt((n-2-1)/(n-2-r_i**2)) for r_i in r]

    @property
    def normal_test(self):
        """Return calculated value from ks."""
        try:
            self.stat, self.p = stats.normaltest(self.residuals)
        except (ValueError, TypeError):
            self.p = np.nan
            self.stat = np.nan
        return self.p

    @property
    def shap_test(self):
        """Return calculated value of the shap test."""
        try:
            self.shap_stat, self.shap_p = stats.shapiro(self.residuals)
        except (ValueError, TypeError):
            self.shap_p = np.nan
            self.shap_stat = np.nan
        return self.shap_p

    @property
    def ks_test(self):
        """Return calculated value from ks."""
        self.ks_stat, self.ks_p = stats.ks_2samp(self.pred, self.meas)
        return self.ks_p

    @property
    def chi_sq(self):
        """Return calculated value from ks."""
        try:
            self.chi_stat, self.chi_p = stats.chisquare(self.pred, self.meas)
        except (ValueError, TypeError):
            self.chi_p = np.nan
            self.chi_stat = np.nan
        return self.chi_p

    @property
    def explained_var_score(self):
        """Return calculated explained_variance_score."""
        return metrics.explained_variance_score(self.meas, self.pred, **self.kwargs)

    @property
    def max_err(self):
        """Return calculated max_error."""
        return metrics.max_error(self.meas, self.pred)

    @property
    def mean_abs_err(self):
        """Return calculated mean_absolute_error."""
        return metrics.mean_absolute_error(self.meas, self.pred, **self.kwargs)

    @property
    def mean_squ_error(self):
        """Return calculated mean_squared_error."""
        return metrics.mean_squared_error(self.meas, self.pred, **self.kwargs)

    @property
    def root_mean_abs_err(self):
        """Return calculated root mean_squared_error."""
        return np.sqrt(metrics.mean_absolute_error(self.meas, self.pred, **self.kwargs))

    @property
    def mean_abs_perc_err(self):
        """Return calculated mean_absolute_percentage_error."""
        return metrics.mean_absolute_percentage_error(self.meas, self.pred, **self.kwargs)

    @property
    def mean_sq_log_err(self):
        """Return calculated mean_squared_log_error."""
        return metrics.mean_squared_log_error(self.meas, self.pred, **self.kwargs)

    @property
    def median_abs_err(self):
        """Return calculated median_absolute_error."""
        return metrics.median_absolute_error(self.meas, self.pred, **self.kwargs)

    @property
    def r_sq_score(self):
        """Return calculated r2_score."""
        return metrics.r2_score(self.meas, self.pred, **self.kwargs)

    @property
    def mean_poisson_dev(self):
        """Return calculated mean_poisson_deviance."""
        return metrics.mean_poisson_deviance(self.meas, self.pred, **self.kwargs)

    @property
    def mean_gamma_dev(self):
        """Return calculated mean_gamma_deviance."""
        return metrics.mean_gamma_deviance(self.meas, self.pred, **self.kwargs)


class MatrixOps:
    """Return sum of squared errors (pred vs actual)."""

    _type = 'matrix_operator'

    def __init__(self, sims_obj, cls_type, xrange=[None, None, None, None],
                 yrange=[None, None, None, None], size=50, min_range=2, **kwargs):
        self.cls_type = cls_type
        self.size = size
        self.std_values = BaseProfile(sims_obj).std_values
        if 'fit' in cls_type.lower() and xrange[0] is None:
            xrange = ['depth', None, None, 'index']
        if 'fit' in cls_type.lower() and yrange[0] is None:
            yrange = ['depth', None, None, 'index']
        if 'pred' in cls_type.lower() and xrange[0] is None:
            xrange = ['conc', None, None, 'log']
        if 'pred' in cls_type.lower() and yrange[0] is None:
            yrange = ['diff', None, None, 'log']

        self.xrange = xrange
        self.yrange = yrange

        self.obj_operator = Composite()
        if 'fit' in cls_type.lower():
            [self.obj_operator.add(ProfileOps(
                FitProfile(sims_obj, start_index=x, stop_index=y, **kwargs), **kwargs))
                for x in self.xrange for y in self.yrange if (x < y)]
        if 'pred' in cls_type.lower():
            [self.obj_operator.add(ProfileOps(
                PredProfile(sims_obj, diff=y, conc=x, **kwargs), **kwargs))
                for x in self.xrange for y in self.yrange]

        if self.obj_operator._family[0].prof.min_range != min_range:
            self.obj_operator.set_attr(attr='min_range', num=min_range, limit=False)

    @property
    def ident(self):
        """Return sum of squared errors (pred vs actual)."""
        return id(self)

    @property
    def xrange(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, '_xrange'):
            self.xrange = ['depth', None, None, 'index']
        return self._xrange

    @xrange.setter
    def xrange(self, value):
        if value[1] is None:
            start_point = self.std_values.loc['low', value[0]]
        else:
            start_point = value[1]
        if value[2] is None:
            end_point = self.std_values.loc['high', value[0]]
        else:
            end_point = value[2]

        if 'ind' in value[3].lower():
            self._xrange = np.linspace(start_point, end_point, self.size, dtype=int)
        elif 'lin' in value[3].lower():
            self._xrange = np.linspace(start_point, end_point, self.size)
        elif 'log' in value[3].lower():
            self._xrange = np.logspace(start_point, end_point, self.size)
        else:
            self._xrange = np.array(range(self.size))

    @property
    def yrange(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, '_yrange'):
            self.yrange = ['depth', None, None, 'index']
        return self._yrange

    @yrange.setter
    def yrange(self, value):
        if value[1] is None:
            start_point = self.std_values.loc['low', value[0]]
        else:
            start_point = value[1]
        if value[2] is None:
            end_point = self.std_values.loc['high', value[0]]
        else:
            end_point = value[2]

        if 'ind' in value[3].lower():
            self._yrange = np.linspace(start_point, end_point, self.size, dtype=int)
        elif 'lin' in value[3].lower():
            self._yrange = np.linspace(start_point, end_point, self.size)
        elif 'log' in value[3].lower():
            self._yrange = np.logspace(start_point, end_point, self.size)
        else:
            self._yrange = np.array(range(self.size))

    def error_calc(self, get_best=True, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        if get_best:
            self.obj_operator.set_best_error(**kwargs)
        else:
            self.obj_operator.set_error(**kwargs)


class Analysis:
    """Return sum of squared errors (pred vs actual)."""

    def __init__(self, obj=None, info=[None, None, None], **kwargs):
        if obj is not None:
            if obj._type == 'matrix_operator':
                self.parent_obj = obj
                self.composite = obj.obj_operator
            elif obj._type == 'composite':
                self.parent_obj = None
                self.composite = obj
            elif obj._type == 'profile_operator':
                self.parent_obj = obj
                self.profile = obj.prof
            elif obj._type == 'fit' or obj._type == 'pred' or obj._type == 'base':
                self.parent_obj = None
                self.profile = obj
        self.info = info

    @property
    def family_df(self):
        """Return sum of squared errors (pred vs actual)."""
        if hasattr(self, 'composite'):
            return self.composite.gen_df()
        else:
            return None

    @property
    def depth(self):
        """Return sum of squared errors (pred vs actual)."""
        if hasattr(self, 'composite'):
            return self.composite._family[0].data['depth'].to_numpy()*1e4
        else:
            return None

    @property
    def data(self):
        """Return sum of squared errors (pred vs actual)."""
        if hasattr(self, 'profile'):
            self._data = self.profile.data.copy()
            self._data['depth'] = self._data['depth']*1e4
            return self._data
        else:
            return None

    @property
    def info(self):
        """Return sum of squared errors (pred vs actual)."""
        return self._info

    @info.setter
    def info(self, val):
        self._info = ['start_index', 'stop_index', 'error']
        if isinstance(val, (list, tuple)) and len(val) == 3:
            self._base_info = ['start_index', 'stop_index', 'error']
            self._info = [x if isinstance(x, str) else self._base_info[i]
                          for i, x in enumerate(val)]
            if 'stats.' in self._info[2].lower():
                self.info[2], attr = self._info[2].split('.')
                self.composite.set_attr('stats_attr', attr)

    @property
    def matrix(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.family_df.pivot_table(values=self.info[2],
                                          columns=self.info[0],
                                          index=self.info[1])

    def focus(self, pairs=None, pair_names=['start_index', 'index_range'], var=[]):
        """Return sum of squared errors (pred vs actual)."""
        if var == []:
            var = ['start_index', 'stop_index', 'index_range', 'start_loc',
                   'stop_loc', 'diff', 'conc', 'error', 'stats']
        df = self.composite.gen_df(var=list(np.unique(var+pair_names)))

        self.focus_df = df.pivot_table(index=pair_names)
        for check in var:
            if check not in self.focus_df.columns and check not in pair_names:
                self.focus_df[check] = df[check].to_numpy()
        if pairs is not None:
            self.focus_df = self.focus_df.loc[pairs]
        return self.focus_df

    def peaks_solver(self, num_of_peaks=2, stop_max=2.5, start_max=2, start_min=2):
        """Return sum of squared errors (pred vs actual)."""
        limited_df = self.family_df[(self.family_df['stop_loc'] < stop_max) &
                                    (self.family_df['start_loc'] < start_max) &
                                    (self.family_df['start_index'] > start_min)].copy().dropna()

        limited_df[['min','max']] = 0

        limited_df = limited_df.sort_values(by=['index_range', 'start_index'])
        limited_df = limited_df.reset_index(drop=True)

        # limited_df['min'][(limited_df['error'].shift(1) > limited_df['error']) &
        #                   (limited_df['error'].shift(-1) > limited_df['error'])] = 1
        # limited_df['max'][(limited_df['error'].shift(1) < limited_df['error']) &
        #                   (limited_df['error'].shift(-1) < limited_df['error'])] = -1

        # limited_df['min'][(limited_df['stop_index'] == limited_df['stop_index'].max())] = 0
        # limited_df['max'][(limited_df['start_index'] == limited_df['start_index'].min())] = 0

        limited_df['min'] = limited_df['error'][
            (limited_df['error'].shift(1) > limited_df['error']) &
            (limited_df['error'].shift(-1) > limited_df['error'])]
        limited_df['max'] = limited_df['error'][
            (limited_df['error'].shift(1) < limited_df['error']) &
            (limited_df['error'].shift(-1) < limited_df['error'])]

        start_min_df = limited_df.pivot_table(index='index_range', columns='start_index', values='min').fillna(0)
        start_max_df = limited_df.pivot_table(index='index_range', columns='start_index', values='max').fillna(0)

        test=start_min_df+start_max_df
        # test=test.fillna(0)

        # self.start_topo = pd.DataFrame()
        # self.start_topo['min_sum'] = start_min_df.sum()
        # self.start_topo['max_sum'] = start_max_df.sum()
        # self.start_topo['comb_sum'] = self.start_topo['min_sum'] + self.start_topo['max_sum']
        # self.start_topo['comb_res'] = self.start_topo['comb_sum']


        # for x in range(len(self.start_topo)-1):
        #     if self.start_topo.iloc[x, 3] > 0:
        #         outer_int = x
        #         inner_int = x+1
        #         while self.start_topo.iloc[inner_int, 3] > 0 and inner_int < len(self.start_topo):
        #             self.start_topo.iloc[outer_int, 3] += self.start_topo.iloc[inner_int, 3]
        #             self.start_topo.iloc[inner_int, 3] = 0
        #             inner_int += 1

        # min_peaks = self.start_topo['comb_res'][(self.start_topo['comb_res'] > 3)]

        # self.peak_df = pd.DataFrame(columns=['start_index', 'count', 'min_bound', 'max_bound'])
        # self.peak_df['start_index'] = min_peaks.index.to_numpy()
        # self.peak_df['count'] = min_peaks.to_numpy()
        # self.peak_df['min_bound'] = self.peak_df['start_index'].shift(1)
        # self.peak_df['max_bound'] = self.peak_df['start_index'].shift(-1)

        # min_peaks = self.start_topo['comb_res'][(self.start_topo['comb_res'] > 3)]

        # self.peak_df = pd.DataFrame(columns=['start_index', 'count', 'min_bound', 'max_bound'])
        # self.peak_df['start_index'] = min_peaks.index.to_numpy()
        # self.peak_df['count'] = min_peaks.to_numpy()
        # self.peak_df['min_bound'] = self.peak_df['start_index'].shift(1)
        # self.peak_df['max_bound'] = self.peak_df['start_index'].shift(-1)

        # self.peak_df.iloc[0,2] = limited_df['start_index'].min()
        # self.peak_df.iloc[-1,3] = limited_df['stop_index'].max()


        # self.peak_df['min_bound'] = self.peak_df['min_bound'].apply(
        #     lambda x: self.peak_df['start_index'].min()
        #     if x != self.peak_df['start_index'].min() else 0)
        # self.peak_df['max_bound'] = self.peak_df['max_bound'].apply(
        #     lambda x: self.peak_df['start_index'].max()
        #     if x != self.peak_df['start_index'].max() else limited_df['start_index'].max())

        # stop_min_df = limited_df.pivot_table(index='index_range', columns='stop_index', values='min')
        # stop_max_df = limited_df.pivot_table(index='index_range', columns='stop_index', values='max')

        # stop_topo = pd.DataFrame()
        # stop_topo['min_sum'] = stop_min_df.sum()
        # stop_topo['max_sum'] = stop_max_df.sum()
        # stop_topo['comb_sum'] = stop_topo['min_sum'] + stop_topo['max_sum']

        # comb_matrix = min_matrix + max_matrix

        # min_stats = min_matrix.describe()
        # max_stats = max_matrix.describe()
        # comb_stats = comb_matrix.describe()


        min_df = limited_df[['start_index', 'index_range', 'error', 'min']].dropna()
        min_df = min_df.sort_values(by=['start_index', 'index_range'])

        min_df['cum_min'] = min_df.groupby('start_index')['min'].cummax()
        min_df['min_new'] = min_df['error'][min_df['cum_min'] <= min_df['error']]

        min_df = min_df.dropna()

        self.min_series = min_df.value_counts('start_index')
        self.min_series = self.min_series.sort_index()

        max_df = limited_df[['start_index', 'index_range', 'error', 'max']]

        max_df = max_df.sort_values(by=['index_range', 'start_index'])
        max_df['max'] = max_df['max'].apply(lambda x: 0 if np.isnan(x) else 1)

        min_peaks = self.min_series.nlargest(num_of_peaks)

        self.peak_df = pd.DataFrame(columns=['start_index', 'count', 'min_bound', 'max_bound'])
        self.peak_df['start_index'] = self.min_series.nlargest(num_of_peaks).index.to_numpy()
        self.peak_df['count'] = self.min_series.nlargest(num_of_peaks).to_numpy()
        self.peak_df['min_bound'] = self.min_series.nlargest(num_of_peaks).index.to_numpy()
        self.peak_df['max_bound'] = self.min_series.nlargest(num_of_peaks).index.to_numpy()

        self.peak_df['min_bound'] = self.peak_df['min_bound'].apply(
            lambda x: self.peak_df['start_index'].min()
            if x != self.peak_df['start_index'].min() else 0)
        self.peak_df['max_bound'] = self.peak_df['max_bound'].apply(
            lambda x: self.peak_df['start_index'].max()
            if x != self.peak_df['start_index'].max() else limited_df['start_index'].max())

        range_df = pd.DataFrame(columns=['index_range'])
        range_df['index_range'] = max_df['index_range'].unique()

        self.peak_dict = {}
        for n in range(num_of_peaks):
            self.peak_dict[n] = list(limited_df[['start_index', 'index_range']]
                                     [limited_df['start_index'] == min_peaks.index[n]]
                                     .itertuples(index=False, name=None))
            max_df['start_shift_%d' % n] = max_df['start_index']-min_peaks.index[n]

            for i, x in enumerate(range_df['index_range']):
                range_df.loc[i, 'start_min_%d' % n] = (max_df['start_index']
                                                       [(max_df['start_shift_%d' % n] < 0) &
                                                        (max_df['max']) &
                                                        (max_df['index_range'] == x)].max())
                if np.isnan(range_df.loc[i, 'start_min_%d' % n]).all():
                    range_df['start_min_%d' % n] = range_df['start_min_%d' % n
                                                            ].fillna(min_peaks.index[n])
                range_df.loc[i, 'start_max_%d' % n] = (max_df['start_index']
                                                       [(max_df['start_shift_%d' % n] > 0) &
                                                        (max_df['max']) &
                                                        (max_df['index_range'] == x)].min())

        range_df = range_df.dropna()

        range_df = range_df.astype(int)

        if (self.min_series.nlargest(num_of_peaks).index[1] <
                self.min_series.nlargest(num_of_peaks).index[0]):
            range_df = range_df.reindex(columns=[range_df.columns[0], range_df.columns[3],
                                                  range_df.columns[4], range_df.columns[1],
                                                  range_df.columns[2]])

        for n in range(1, len(range_df.columns)):
            if min_peaks.index[-(-n//2)-1] > 5:
                range_df.iloc[:, n] = range_df.iloc[:, n].replace((0, 1))

        index_range = range_df['index_range'].to_numpy(copy=True)
        range_df['index_range'] = range_df.iloc[:, 1]-1
        range_df = range_df.where(range_df.shift(1, axis=1) < range_df,
                                  other=range_df.shift(1, axis=1), axis='index')
        range_df['index_range'] = index_range

        self.range_dict = {i: (list(range_df.loc[:, ('start_min_%d' % i, 'index_range')]
                               .itertuples(index=False, name=None)) +
                               list(range_df.loc[:, ('start_max_%d' % i, 'index_range')]
                               .itertuples(index=False, name=None)))
                           for i, n in enumerate(min_peaks.index)}

        self.region_dict = {}
        for n in range(num_of_peaks):
            self.region_dict[n] = self.peak_dict[n] + self.range_dict[n]

        return self.range_dict, self.peak_dict, self.start_topo

    def peaks(self, peak=0, min_range=3, pair_names=['start_index', 'index_range'], **kwargs):
        """Evaluate for removal."""
        if not hasattr(self, 'range_dict'):
            self.peaks_solver()

        pairs = self.region_dict[peak]
        pairs_new = [x for x in pairs if x[0] >= self.peak_df.iloc[peak, 2] and
                     x[0]+x[1] <= self.peak_df.iloc[peak, 3]]

        focii = self.focus(pairs=pairs_new, pair_names=pair_names)
        focii = (focii[(focii['diff'] != 1e-11) &
                       (focii.index.get_level_values(1) > min_range) &
                       (focii['stop_index'] > self.peak_df.iloc[peak, 0])])

        focus_stats = focii.describe()
        return focus_stats, focii

    def check_error(self, **kwargs):
        """Evaluate for removal."""
        self.error_matrix = self.obj_matrix.applymap(
            lambda x: ProfileOps(x.data, x.pred, x.start_index, x.stop_index)
            if not isinstance(x, (int, np.integer)) else 1)

        return self.error_matrix.applymap(lambda x: x.err(**kwargs)
                                          if not isinstance(x, (int, np.integer)) else 1)

    def stitcher(self, sims_obj, *args):
        """Return sum of squared errors (pred vs actual)."""
        if len(args) == 1:
            args = args[0]
        self.fits = tuple(args)
        if isinstance(self.fits, tuple):
            self.indexed = list()
            self.profile_list = list()
            self.profile_num = list()
            num = 1
            for row in reversed(range(len(self.fits))):

                start = self.fits[row][0]
                stop = self.fits[row][1]
                self.profile_list.append(FitProfile(sims_obj, start_index=start, stop_index=stop))
                profile = self.profile_list[-1].pred
                if start <= 1:
                    start = 0
                if stop == 199:
                    stop = 200
                [(self.indexed.append(x), self.profile_num.append(num))
                 for x in profile[start:stop]]

                num += 1

            self.stitched_res = BaseProfile(sims_obj)
            self.stitched_res.pred = np.array(self.indexed)
            self.stitched_res._data['Range number'] = np.array(self.profile_num)

            return self.stitched_res


class Plotter(Analysis):
    """Return sum of squared errors (pred vs actual)."""

    def __init__(self, obj=None, info=[None, None, None], **kwargs):
        super().__init__(obj, info)

    def map_plot(self, name=None, info=[None, None, None], matrix=None, conv=[1, 1],
                 zlog=True, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        conv_info = None
        if 'depth_range' in info:
            for i, x in enumerate(info):
                if x == 'depth_range':
                    info[i] = 'index_range'
                    conv_info = i

        if matrix is None and not hasattr(self, 'composite'):
            print('Insert matrix!')
            return
        elif matrix is None:
            self.info = info
            to_plot = self.matrix
        else:
            to_plot = matrix

        if name is None:
            name = 'Fit Profile'

        if conv_info == 0 and self.depth is not None:
            to_plot.columns = [self.depth[x] for x in to_plot.columns]
        if conv_info == 1 and self.depth is not None:
            to_plot.index = [self.depth[x] for x in to_plot.index]

        # standard plot information
        plt_kwargs = {'name': name, 'xname': "Start Point (um)",
                      'yname': "End Point (um)", 'zname': "Error",
                      'cmap': 'kindlmann'}
        plt_kwargs.update({key: kwargs[key] for key in kwargs})

        gf.map_plt(to_plot.columns*conv[0], to_plot.index*conv[1],
                   np.ma.masked_invalid(to_plot.to_numpy()), **plt_kwargs)

        plt.show()

    def prof_plot(self, name=None, data_in=None, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        multi_plot = False
        if data_in is None and not hasattr(self, 'data'):
            print('Insert data!')
            return
        elif data_in is None:
            to_plot = self.data
        else:
            if data_in.index.names[0] is not None:
                data_in = data_in.reset_index()
                if 'pred' in data_in.columns:
                    multi_plot = True
            to_plot = data_in

        if name is None:
            name = 'Residual Plot'

        plt_kwargs = {'name': name, 'xname': 'Depth (um)',
                      'yname': 'Residuals', 'palette': 'kindlmann'}

        plt_kwargs.update({key: kwargs[key] for key in kwargs})
        if not multi_plot:
            gf.scatter(data=to_plot, **plt_kwargs)
        else:
            for pair in to_plot.index:
                plot_dict = to_plot.loc[pair, :].to_dict()
                plot_df = pd.DataFrame(plot_dict)
                gf.scatter(data=plot_df, **plt_kwargs)

        plt.show()
