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
from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar
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
# from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error,  mean_absolute_percentage_error, median_absolute_error, r2_score, mean_poisson_deviance, mean_gamma_deviance, mean_squared_log_error

warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings("ignore")

sns.set_style('dark')


# %% Functions
def c_np(depth, diff, conc, thick, temp, e_app, time):
    """Return sum of squared errors (pred vs actual)."""
    if diff < 0:
        diff = 10**float(diff)
        conc = 10**float(conc)
    mob = diff/(gf.KB_EV*temp)
    term_B = erfc(-mob*e_app*time/(2*np.sqrt(diff*time)))
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

    @property
    def family_names(self):
        """Return sum of squared errors (pred vs actual)."""
        self._family_names = pd.Series()
        for obj in self._family:
            x = obj.prof
            if x._type.lower() == 'fit':
                self._family_names[str(x.start_index)+'-'+str(x.stop_index)] = x
            elif x._type.lower() == 'pred':
                self._family_names[str(x.diff)+'-'+str(x.conc)] = x
            else:
                self._family_names[x.ident] = x
        return self._family_names

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

    def set_attr(self, attr='start_index', num=0, limit=False, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        self.limit = limit
        for work in (self.chores):
            setattr(work.prof, attr, num)

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

    def matrix(self, info=[None, None, None], limit=False, get_obj=False, **kwargs):
        """
        Generate Matrix.

        Create Matrix where info is [cols name, ind name, and value name]
        """
        if info[0] is None:
            info[0] = 'start_index'
        if info[1] is None:
            info[1] = 'stop_index'
        if info[2] is None:
            info[2] = 'error'

        self.limit = limit

        if 'stats' in info[2].lower():
            info[2], attr = info[2].split('.')
            self.set_attr('stats', attr)

        temp_df = pd.DataFrame()
        if get_obj:
            for work in (self.chores):
                cols = getattr(work.prof, info[0])
                rows = getattr(work.prof, info[1])
                temp_df.loc[rows, cols] = work.prof
        else:
            for work in (self.chores):
                cols = getattr(work.prof, info[0])
                rows = getattr(work.prof, info[1])
                temp_df.loc[rows, cols] = getattr(work.prof, info[2])

        return temp_df


class DataProfile:
    """Return sum of squared errors (pred vs actual)."""

    def __init__(self, slog, **kwargs):
        self.params = slog
        if not np.isnan(self.params['Fit depth/limit']):
            self.fit_depth_cm = gf.tocm(self.params['Fit depth/limit'], self.params['Fit Dep unit'])
        else:
            self.params['Fit depth/limit'] = lin_test(self.data['Depth'].to_numpy(),
                                                      self.data['Na'].to_numpy(), 0.05)[1]
            self.params['Fit Dep unit'] = 'cm'

        self.data_treatment()

        self.data_bgd = pd.Series()

        self.limit_test()
        # if 'tof' in self.params['Type'].lower():
        self.regress_test(**kwargs)

    def data_treatment(self):
        """Return sum of squared errors (pred vs actual)."""
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
            if 'x' in col.lower() or 'depth' in col.lower():
                self.data['Depth'] = gf.tocm(
                    data_raw[col].to_numpy(copy=True), self.params['X unit'])
            if 'na+' in col.lower() and 'conc' in col.lower():
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


@dataclass
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

    min_range = 2
    min_index = 0
    diff = 1e-14
    conc = 1e18
    error = 1
    p_value = 0

    stats = 'mean_abs_perc_err'
    stats_base = 'mean_abs_perc_err'

    @classmethod
    def factory(cls,obj):
        return cls(
            data_bgd = obj.data_bgd.copy(),
            depth = obj.data['Depth'].to_numpy(),
            sims = obj.data['Na'].to_numpy(),
            pred = np.ones_like(obj.data['Na'].to_numpy()),
            max_index = len(self.depth)-1,
            bgd_index = self.data_bgd['bgd_ave'],



            std_values = pd.DataFrame({'conc': (15, 18, 21),
                                            'diff': (-17, -14, -11),
                                            'depth': (0, int(self.max_index/2), self.max_index)},
                                           index=('low', 'mid', 'high')),
            conditions = pd.Series({'thick': obj.thick_cm,
                                         'time': obj.params['Stress Time'],
                                         'temp': obj.params['Temp'],
                                         'e_field': obj.params['Volt']/obj.thick_cm,
                                         'volt': obj.params['Volt']}),
            info = pd.Series({'ident': self.ident,
                                   'sample': obj.params['Sample'],
                                   'type': obj.params['Type'],
                                   'class': 'BaseProfile',
                                   'measurement': obj.params['Measurement']}),
                    )
        self.data_bgd = obj.data_bgd.copy()
        self.depth = obj.data['Depth'].to_numpy()
        self.sims = obj.data['Na'].to_numpy()
        self.pred = np.ones_like(self.sims)
        self.max_index = len(self.depth)-1
        self.bgd_index = self.data_bgd['bgd_ave']



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
        if not hasattr(self, '_data'):
            self._data = pd.DataFrame(columns=('depth', 'SIMS', 'log(SIMS)', 'pred',
                                               'log(pred)', 'weight', 'residuals',
                                               'residuals of log', 'ISR', 'ESR'))
        self._data['depth'] = self.depth
        self._data['SIMS'] = self.sims
        self._data['log(SIMS)'] = np.log10(self.sims)
        self._data['pred'] = self.pred
        self._data['log(pred)'] = np.log10(self.pred)
        self._data['weight'] = [self.pred[x]/self.sims[x] if self.pred[x] > self.sims[x]
                                else 1 for x in range(self.max_index+1)]
        self._data['residuals'] = self.sims - self.pred
        self._data['residuals of log'] = np.log10(self.sims) - np.log10(self.pred)
        self._data['ISR'] = Stats(self._data).int_std_res
        self._data['ESR'] = Stats(self._data).ext_std_res
        return self._data

    @property
    def pred(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, '_pred'):
            self._pred = np.ones_like(self.sims)
        return np.where(self._pred <= 1e-30, 1e-30, self._pred)

    @pred.setter
    def pred(self, pred_in):
        if not hasattr(self, '_data'):
            self._data = pd.DataFrame(columns=('depth', 'SIMS', 'log(SIMS)', 'pred',
                                               'log(pred)', 'weight', 'residuals',
                                               'residuals of log', 'log of residuals'))
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
        return gf.Sig_figs((self.stop_loc-self.start_loc),5)

    @property
    def stats(self):
        """Return sum of squared errors (pred vs actual)."""
        return self._stat_attr

    @stats.setter
    def stats(self, attr):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, '_stats'):
            self._stats = Stats(self.data[self.start_index:self.stop_index+1])
        self._stat_attr = getattr(self._stats, attr)

    @property
    def stats_base(self):
        """Return sum of squared errors (pred vs actual)."""
        return self._stat_base_attr

    @stats_base.setter
    def stats_base(self, attr):
        """Return sum of squared errors (pred vs actual)."""
        self._stats_base = Stats(self.data[self.start_index:self.stop_index+1])
        self._stat_base_attr = getattr(self._stats_base, attr)


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

        self.stats = 'mean_abs_perc_err'
        self.stats_base = 'mean_abs_perc_err'

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

        self.c_np_new = partial(c_np, thick=self.conditions['thick'],
                                temp=gf.CtoK(self.conditions['temp']),
                                e_app=self.conditions['e_field'],
                                time=self.conditions['time'])

        self.curve_fit_kwargs = {'x_scale': 'jac', 'xtol': 1e-12, 'jac': '3-point'}
        self.unpack_kwargs(kwargs)

        self.fitter(**kwargs)

        self.info['class'] = 'FitProfile'

        self.stats = 'mean_abs_perc_err'
        self.stats_base = 'mean_abs_perc_err'

        self.error_log = 0

    def unpack_kwargs(self, kwargs):
        """Return sum of squared errors (pred vs actual)."""
        self.curve_fit_kwargs.update({key: kwargs[key] for key in kwargs
                                      if key in self._curve_fit_keys})
        [kwargs.pop(x) for x in self._curve_fit_keys if x in kwargs.keys()]
        self.__dict__.update(kwargs)

    def fitter(self, diff_pred=None, conc_pred=None, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        if diff_pred is None:
            diff_pred = self.std_values['diff']
        if conc_pred is None:
            conc_pred = self.std_values['conc']
        self.unpack_kwargs(kwargs)
        try:
            fittemp = curve_fit(self.c_np_new,
                                self.depth[self.start_index:self.stop_index+1],
                                self.sims[self.start_index:self.stop_index+1],
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

        self.pred = np.array(self.c_np_new(self.depth, self.diff, self.conc))

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

        self.prof._stats = Stats(self.data[self.start:self.stop+1],
                                 log_form,  **self.error_kwargs)

        self.prof.stats = 'mean_abs_perc_err'

        self.error = self.prof.stats * self.w_constant

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

    def __init__(self, depth_df, log_form=False, **kwargs):

        self.depth = depth_df['depth'].to_numpy(copy=True)
        self.meas = depth_df['SIMS'].to_numpy(copy=True)

        if len(depth_df.columns) > 2 and log_form:
            self.meas = depth_df['log(SIMS)'].to_numpy(copy=True)
            self.pred = depth_df['log(pred)'].to_numpy(copy=True)
        elif len(depth_df.columns) > 2:
            self.pred = depth_df['pred'].to_numpy(copy=True)

        # degrees of freedom population dep. variable variance
        self._dft = self.depth.shape[0] - 1
        # degrees of freedom population error variance
        self._dfe = self.depth.shape[0]

        self.avg_x = np.mean(self.depth)
        self.avg_y = np.mean(self.meas)
        self.avg_m = np.mean(self.pred)
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
    def sse(self):
        """Return sum of squared errors (pred vs actual)."""
        squared_errors = (self.meas - self.pred) ** 2
        return np.sum(squared_errors)

    @property
    def sst(self):
        """Return total sum of squared errors (actual vs avg(actual))."""
        squared_errors = (self.meas - self.avg_y) ** 2
        return np.sum(squared_errors)

    @property
    def r_squared(self):
        """Return calculated value of r^2."""
        return 1 - self.sse/self.sst

    @property
    def adj_r_squared(self):
        """Return calculated value of adjusted r^2."""
        return 1 - (self.sse/self._dfe) / (self.sst/self._dft)

    @property
    def residuals(self):
        """Return calculated external standardized residual.."""
        return self.meas - self.pred

    @property
    def int_std_res(self):
        """Return calculated internal standardized residual."""
        n = len(self.depth)
        diff_mean_sqr = np.dot((self.depth - self.avg_x), (self.depth - self.avg_x))
        # h_ii_alt = np.diff(y_hat)/np.diff(Y)
        # h_ii_alt = np.append(h_ii_alt,h_ii_alt[-1])
        h_ii = (self.depth - self.avg_x) ** 2 / diff_mean_sqr + (1 / n)
        Var_e = np.sqrt(sum((self.meas - self.pred) ** 2)/(n-2))
        return self.residuals/(Var_e*((1-h_ii) ** 0.5))

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
    def chi_sq(self):
        """Return calculated value from ks."""
        try:
            self.chi_stat, self.chi_p = stats.chisquare(self.pred, self.meas)
        except (ValueError, TypeError):
            self.chi_p = np.nan
            self.chi_stat = np.nan
        return self.chi_p

    @property
    def ks_test(self):
        """Return calculated value from ks."""
        self.ks_stat, self.ks_p = stats.ks_2samp(self.pred, self.meas)
        return self.ks_p

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

    @property
    def error_matrix(self):
        """Return sum of squared errors (pred vs actual)."""
        if hasattr(self, 'obj_matrix'):
            return self.obj_matrix.applymap(lambda x: x.error
                                            if not isinstance(x, (int, np.integer)) else None)
        else:
            return np.ones((self.size, self.size))

    @property
    def fit_curves(self):
        """Return sum of squared errors (pred vs actual)."""
        col = ['Location', 'error', 'diff', 'conc', 'start index', 'stop index',
               'range (points)', 'start', 'stop', 'range (um)']

        if hasattr(self, 'min_array'):
            return pd.DataFrame([[tuple(x),
                                self.obj_matrix.iloc[tuple(x)].error,
                                self.obj_matrix.iloc[tuple(x)].diff,
                                self.obj_matrix.iloc[tuple(x)].conc,
                                self.obj_matrix.iloc[tuple(x)].start_index,
                                self.obj_matrix.iloc[tuple(x)].stop_index,
                                self.obj_matrix.iloc[tuple(x)].index_range,
                                self.obj_matrix.iloc[tuple(x)].start_loc,
                                self.obj_matrix.iloc[tuple(x)].stop_loc,
                                self.obj_matrix.iloc[tuple(x)].depth_range]
                                for x in self.min_array], columns=col)
        else:
            return pd.DataFrame(np.empty((0, len(col)), int), columns=col)

    def error_calc(self, get_best=True, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        if get_best:
            self.obj_operator.set_best_error(**kwargs)
        else:
            self.obj_operator.set_error(**kwargs)

    def set_bgd(self, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        # if not hasattr(self, 'obj_matrix'):
        #     self.gen_matrix()

        # self.error_calc(use_index=False, **kwargs)
        self.obj_operator.set_best_error(use_index=False, **kwargs)

        self.minima_bgd = np.unravel_index(
            self.error_matrix.to_numpy(na_value=np.inf).argmin(), self.error_matrix.shape)

        # finds the start of bgdrnd --> returns int
        self.bgd_index = self.obj_matrix.iloc[self.minima_bgd].min_index

        if self.bgd_index == 0:
            self.bgd_index = self.max_index

        self.obj_operator.set_attr(attr='start_index', num=0, limit=False)
        self.obj_operator.set_attr(attr='bgd_index', num=self.bgd_index, limit=False)

        # forces range on low conc end and recalculates
        self.error_calc(get_best=False)

    def find_ranges(self, method='fst_fwd', run_bgd=False, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, 'obj_matrix'):
            self.gen_matrix()

        if not hasattr(self, 'bgd_index') and run_bgd:
            self.set_bgd(**kwargs)

        self.min_array = np.empty((0, 2), int)

        if method == 'fst_fwd':
            self.fwd(**kwargs)
        elif method == 'slw_fwd':
            self.fwd(fast=False, **kwargs)

    def fwd(self, fast=True, full_range=False, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        if not full_range:
            if not hasattr(self, 'bgd_index'):
                self.set_bgd(**kwargs)
            start = self.bgd_index
            stop = self.sims.argmax()
            min_loc = self.minima_bgd  # tuple
        else:
            start = self.max_index
            stop = self.sims.argmax()
            min_loc = (self.rows.argmax(), self.cols.argmin())

        self.min_array = np.append(self.min_array, [np.array(self.minima_bgd)], axis=0)
        slow_range = np.array(range(start-1, stop-1, -1))
        iterations = 0
        index_temp = start
        self.obj_operator.set_attr(attr='start_index', num=index_temp, limit=False)
        while index_temp > stop and iterations < start:
            self.error_calc(use_index=False, x_in=index_temp, limit=True, reset=True, **kwargs)

            # generate a temporary error array above prev best--> df matrix
            err_temp = self.obj_matrix.applymap(
                lambda x: x.error if x.diff < self.rows[min_loc[0]]
                and x.conc > self.cols[min_loc[1]] else 1)

            # find indexes (diff&conc) of minimum value in range just tested --> tuple
            min_loc = np.unravel_index(np.array(err_temp).argmin(), np.array(err_temp).shape)

            self.min_array = np.append(self.min_array, [np.array(min_loc)], axis=0)
            iterations += 1

            self.obj_matrix.applymap(lambda x: self.obj_operator.drop(x)
                                     if x.diff > self.rows[min_loc[0]]
                                     and x.conc < self.cols[min_loc[1]] else None)
            if fast:
                # get the start index of the minima location, only for data location
                self.obj_operator.set_attr(attr='stop_index', num=index_temp, limit=True)
                index_temp = self.obj_matrix.iloc[min_loc].min_index
                self.obj_operator.set_attr(attr='start_index', num=index_temp, limit=True)

            else:
                index_temp = slow_range[iterations]


class Analysis:
    """Return sum of squared errors (pred vs actual)."""

    def __init__(self, obj=None, **kwargs):
        if obj is not None:
            if obj._type == 'matrix_operator':
                self.parent_obj = obj
                self.composite = obj.obj_operator
                self.key_list = self.composite.family_names.index
            elif obj._type == 'composite':
                self.parent_obj = None
                self.composite = obj
                self.key_list = self.composite.family_names.index
            elif obj._type == 'profile_operator':
                self.parent_obj = obj
                self.profile = obj.prof
                self.data = obj.prof.data.copy()
            elif obj._type == 'fit' or obj._type == 'pred' or obj._type == 'base':
                self.parent_obj = None
                self.profile = obj
                self.data = obj.data.copy()

    def find_prof(self, loc, conv_key=True, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        try:
            self.profile = self.composite.family_names[str(loc[0])+'-'+str(loc[1])]
        except KeyError:
            print('Key not found, check key_list')

    # def key_converter(self,key_names,key_values,convert_info, **kwargs):
    #     self.key_list_split = [x.split('-') for x in self.key_list]
    #     if key_names == 'depth'

        # elif x._type.lower() == 'pred':
        #     self._family_names[str(x.diff)+'-'+str(x.conc)] = x
        # else:
        #     self._family_names[x.ident] = x
        # if not isinstance(loc[0], (int, np.integer)):
        #     x = gf.find_nearest(self.obj_matrix.columns, loc[0])
        # else:
        #     x = loc[0]

        # if not isinstance(loc[1], (int, np.integer)):
        #     y = gf.find_nearest(self.obj_matrix.rows, loc[1])
        # else:
        #     y = loc[1]

        # try:
        #     res = getattr(self.obj_matrix.iloc[x, y], attr)
        # except AttributeError:
        #     print('Not an attribute')
        #     res = None

        # return res

    def check_error(self, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        self.error_matrix = self.obj_matrix.applymap(
            lambda x: ProfileOps(x.data, x.pred, x.start_index, x.stop_index)
            if not isinstance(x, (int, np.integer)) else 1)
        # self.error_operator = Composite()
        # self.error_matrix.applymap(lambda x: self.error_operator.add(x)
        #                               if isinstance(x, (int, np.integer)) else None)

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
                # [self.profile_num.append(num) for x in profile[start:stop]]
                num += 1
                # if not hasattr(self, 'additive'):
                #     self.additive = np.zeros_like(profile)
                # self.additive += profile
            self.stitched_res = BaseProfile(sims_obj)
            self.stitched_res.pred = np.array(self.indexed)
            self.stitched_res._data['Range number'] = np.array(self.profile_num)

            return self.stitched_res

        # else:
        #     self.indexed = list()
        #     for row in reversed(range(len(self.fits))):
        #         profile = self.get_single(self.fits.loc[row,'Location'],'pred')
        #         start = self.fits.loc[row,'start index']
        #         stop = self.fits.loc[row,'stop index']
        #         if start <= 1:
        #             start = 0
        #         if stop == 199:
        #             stop = 200
        #         [self.indexed.append(x) for x in profile[start:stop]]
        #         if not hasattr(self, 'additive'):
        #             self.additive = np.zeros_like(profile)
        #         self.additive += profile


class Plotter:
    """Return sum of squared errors (pred vs actual)."""

    def __init__(self, obj=None, **kwargs):
        if obj is not None:
            if obj._type == 'matrix_operator':
                self.parent_obj = obj
                self.composite = obj.obj_operator
            elif obj._type == 'composite':
                self.parent_obj = None
                self.composite = obj
            elif obj._type == 'profile_operator':
                self.parent_obj = obj
                self.data = obj.prof.data.copy()
                self.data['depth'] = self.data['depth']*1e4
            elif obj._type == 'fit' or obj._type == 'pred' or obj._type == 'base':
                self.parent_obj = None
                self.data = obj.data.copy()
                self.data['depth'] = self.data['depth']*1e4

    def map_plot(self, name=None, info=[None, None, None], matrix=None, conv=[1, 1],
                 zlog=True, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        if matrix is None and not hasattr(self, 'composite'):
            print('Insert matrix!')
            return
        elif matrix is None:
            to_plot = self.composite.matrix(info)
        else:
            to_plot = matrix

        if name is None:
            name = 'Fit Profile'

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
        if data_in is None and not hasattr(self, 'data'):
            print('Insert data!')
            return
        elif data_in is None:
            to_plot = self.data
        else:
            to_plot = data_in

        if name is None:
            name = 'Residual Plot'

        plt_kwargs = {'name': name, 'xname': 'Depth (um)',
                      'yname': 'Residuals','palette': 'kindlmann'}
        plt_kwargs.update({key: kwargs[key] for key in kwargs})

        gf.scatter(data=to_plot, **plt_kwargs)

        plt.show()
