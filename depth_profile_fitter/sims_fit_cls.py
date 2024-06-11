# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 12:36:55 2021.

@author: j2cle
"""

# %% import section
import abc
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import erfc
# from scipy import stats
from sklearn import metrics
from functools import partial
from scipy.optimize import curve_fit


import research_tools as rt
import research_tools.functions.unit_conversion as rtu

warnings.simplefilter("ignore", np.RankWarning)
warnings.filterwarnings("ignore")

sns.set_style("dark")


# %% Functions
def c_np(depth, diff, conc, thick, temp, e_app, time, log_form=False):
    """Return the nernst planck analytic sol"""
    # TODO: See if the logarithmic form can be removed

    # if diffusivity is < 0 then its the log(D) and needs to be converted
    if diff < 0:
        diff = 10 ** float(diff)
        conc = 10 ** float(conc)
    if log_form:
        return np.log10(rt.nernst_planck_analytic_sol(conc, depth, thick, e_app, diff, time, 1, temp))
    else:
        return rt.nernst_planck_analytic_sol(conc, depth, thick, e_app, diff, time, 1, temp)



def depth_conv(data_in, unit, layer_act, layer_meas):
    """Return data in the correct depth"""
    data_out = data_in
    if unit != "s":
        if not pd.isnull(layer_meas):
            data_out[data_in < layer_meas] = (
                data_in[data_in < layer_meas] * layer_act / layer_meas
            )
            data_out[data_in >= layer_meas] = (
                (data_in[data_in >= layer_meas] - layer_meas)
                * ((max(data_in) - layer_act) / (max(data_in) - layer_meas))
            ) + layer_act

        if unit != "cm":
            data_out = rtu.Length(data_in, unit).cm #TODO: fix unit conversion
    return data_out


def lin_test(x, y, lim=0.025):
    """Perform a linear test"""
    line_info = np.array([np.polyfit(x[-n:], y[-n:], 1) for n in range(1, len(x))])

    delta = np.diff(line_info[int(0.1 * len(x)) :, 0])
    delta = delta / max(abs(delta))
    bounds = np.where(delta < -lim)[0]
    if bounds[0] + len(bounds) - 1 == bounds[-1]:
        bound = len(x) - bounds[0]
    else:
        bound = (
            len(x)
            - [
                bounds[n]
                for n in range(1, len(bounds))
                if bounds[n - 1] + 1 != bounds[n]
            ][-1]
        )
    return bound, x[bound]


def peak_cycles(
    obj,
    focus_df,
    min_start=None,
    max_start=None,
    min_range=2,
    peak_range=None,
    max_range=None,
    pair_set=None,
    old_range=False,
):

    for peak in focus_df.index:
        if pair_set != "all":
            pair_set = peak
        focus_stats, focii = obj.pks_analyze(
            peak=peak,
            min_start=min_start,
            max_start=max_start,
            min_range=min_range,
            peak_range=peak_range,
            max_range=max_range,
            pair_set=pair_set,
            old_range=old_range,
        )

        focus_df.loc[peak, "count"] = focus_stats.loc["count", "diff"]
        focus_df.loc[peak, "start"] = focii.index.get_level_values(0).min()
        focus_df.loc[peak, "stop"] = focii["stop_index"].max()
        focus_df.loc[peak, "min start"] = focus_stats.loc["min", "start_loc"]
        focus_df.loc[peak, "ave start"] = focus_stats.loc["mean", "start_loc"]
        focus_df.loc[peak, "max start"] = focus_stats.loc["max", "start_loc"]
        focus_df.loc[peak, "min stop"] = focus_stats.loc["min", "stop_loc"]
        focus_df.loc[peak, "ave stop"] = focus_stats.loc["mean", "stop_loc"]
        focus_df.loc[peak, "max stop"] = focus_stats.loc["max", "stop_loc"]
        focus_df.loc[peak, "error"] = focus_stats.loc["mean", "error"]
        focus_df.loc[peak, "error std"] = focus_stats.loc["std", "error"]
        focus_df.loc[peak, "diff"] = focus_stats.loc["mean", "diff"]
        focus_df.loc[peak, "diff std"] = focus_stats.loc["std", "diff"]
        focus_df.loc[peak, "conc"] = float(focus_stats.loc["mean", "conc"])
        focus_df.loc[peak, "conc std"] = focus_stats.loc["std", "conc"]
    return focus_df


def pivot_cleaner(table_in):
    if table_in.shape[0] == table_in.shape[1]:
        return table_in
    table1 = table_in.melt(ignore_index=False, value_name="error").reset_index()
    table2 = table1.pivot_table(
        index=[table1.columns[0], table1.columns[1]]
    ).to_xarray()
    table2 = table2.interpolate_na(dim=table1.columns[0])

    table_out = table2.to_dataframe().pivot_table(
        index=table1.columns[0], columns=table1.columns[1], values=table1.columns[2]
    )

    return table_out


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

    _type = "composite"

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
        for work in self.chores:
            work.set_error(**kwargs)

    def set_best_error(self, limit=False, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        self.limit = limit
        for work in self.chores:
            work.set_best_error(**kwargs)

    def set_attr(self, attr="start_index", val=0, limit=False, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        self.limit = limit
        for work in self.chores:
            setattr(work.prof, attr, val)

    def get_attr(self, attr="error", limit=False, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        work_attr = list()
        self.limit = limit
        for work in self.chores:
            work_attr.append(getattr(work.prof, attr))
        return work_attr

    def del_attr(self, attr="error", limit=False, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        self.limit = limit
        for work in self.chores:
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
            var = [
                "start_index",
                "stop_index",
                "index_range",
                "start_loc",
                "stop_loc",
                "depth_range",
                "diff",
                "conc",
                "error",
                "stats",
            ]
        if "stats." in var:
            var_loc = int(np.where(np.array(var) == "stats.")[0][0])
            var[var_loc], attr = var[var_loc].split(".")
            self.set_attr("stats_attr", attr)
        listed = [[getattr(work.prof, x) for x in var] for work in self.chores]
        listed = [
            [x.to_dict() if isinstance(x, (pd.DataFrame, pd.Series)) else x for x in y]
            for y in listed
        ]
        return pd.DataFrame(listed, columns=var)


class DataProfile:
    """Return sum of squared errors (pred vs actual)."""

    def __init__(self, slog, limit=False, loc=None, even=False, **kwargs):
        self.params = slog

        self.data_treatment()

        if not np.isnan(self.params["Layer (actual)"]):
            self.a_layer_cm = rtu.Length(
                self.params["Layer (actual)"], self.params["A-Layer unit"]
            ) #TODO: fix unit conversion
        else:
            self.a_layer_cm = 0
        if not np.isnan(self.params["Fit depth/limit"]):
            self.fit_depth_cm = rtu.Length(
                self.params["Fit depth/limit"], self.params["Fit Dep unit"]
            ).cm #TODO: fix unit conversion
        else:
            self.params["Fit depth/limit"] = lin_test(
                self.data["Depth"].to_numpy(), self.data["Na"].to_numpy(), 0.05
            )[1]
            self.params["Fit Dep unit"] = "cm"
        if not np.isnan(self.params["Layer (profile)"]):
            self.p_layer_cm = rtu.Length(
                self.params["Layer (profile)"], self.params["P-Layer unit"]
            ).cm #TODO: fix unit conversion
        self.data_bgd = pd.Series()

        self.limit_test()
        # if 'tof' in self.params['Type'].lower():
        self.regress_test(**kwargs)

        if limit:
            if loc is None:
                self.data = self.data.iloc[self.data_bgd["bgd_ave"], :]
            elif isinstance(loc, (int, np.integer)):
                self.data = self.data.iloc[loc, :]
            else:
                self.data = self.data[self.data["Depth"] < loc]
        if even:
            self.data["Depth"] = np.linspace(
                self.data["Depth"].min(),
                self.data["Depth"].max(),
                len(self.data["Depth"]),
            )

    def data_treatment(self):
        """Return sum of squared errors (pred vs actual)."""
        if self.params["Type"] == "NREL MIMS":
            data_raw = pd.read_excel(
                self.params["Data File Location"],
                sheet_name=self.params["Tab"],
                usecols=self.params["Columns"],
            ).dropna()
        elif "matrix" in self.params["Type"].lower():
            data_raw = pd.read_excel(
                self.params["Data File Location"], header=[0, 1], index_col=0
            )
            data_raw.columns = data_raw.columns.map("{0[0]} {0[1]}".format)
            data_raw = data_raw.reset_index()
        elif "TOF" in self.params["Type"]:
            header_in = pd.read_csv(
                self.params["Data File Location"],
                delimiter="\t",
                header=None,
                skiprows=2,
                nrows=3,
            ).dropna(axis=1, how="all")
            header_in = header_in.fillna(method="ffill", axis=1)
            headers = [
                header_in.iloc[0, x] + " " + header_in.iloc[2, x]
                for x in range(header_in.shape[1])
            ]
            data_raw = pd.read_csv(
                self.params["Data File Location"],
                delimiter="\t",
                header=None,
                names=headers,
                index_col=False,
                skiprows=5,
            )
        elif "DSIMS" in self.params["Type"]:
            header_in = pd.read_csv(
                self.params["Data File Location"],
                delimiter="\t",
                header=None,
                skiprows=14,
                nrows=2,
            ).dropna(axis=1, how="all")
            header_temp = (
                header_in.iloc[0, :].dropna().to_list()
                + header_in.iloc[0, :].dropna().to_list()
            )
            header_in.iloc[0, : len(header_temp)] = sorted(
                header_temp, key=lambda y: header_temp.index(y)
            )
            header_in = header_in.dropna(axis=1)
            headers = [
                header_in.iloc[0, x] + " " + header_in.iloc[1, x]
                for x in range(header_in.shape[1])
            ]
            data_raw = (
                pd.read_csv(
                    self.params["Data File Location"],
                    delimiter="\t",
                    header=None,
                    names=headers,
                    index_col=False,
                    skiprows=16,
                )
                .dropna()
                .astype(float)
            )
        self.data = pd.DataFrame(np.ones((len(data_raw), 2)), columns=["Depth", "Na"])
        data_cols = list(data_raw.columns)

        if "atoms" in self.params["Y unit"].lower():
            col_type = "conc"
        elif "dsims" in self.params["Type"].lower():
            col_type = "i [c/s]"
        else:
            col_type = "inten"
        for col in data_cols:
            if self.params["Type"] == "NREL MIMS":
                if "x" in col.lower() or "depth" in col.lower():
                    self.data["Depth"] = depth_conv(
                        data_raw[col].to_numpy(copy=True),
                        self.params["X unit"],
                        self.params["Layer (actual)"],
                        self.params["Layer (profile)"],
                    )
                if "na" in col.lower():
                    na_col = col
                if self.params["Matrix"] in col:
                    data_matrix = data_raw[col].to_numpy()
            elif "matrix" in self.params["Type"].lower():
                if col.lower() == "z":
                    self.data["Depth"] = rtu.Length(
                        data_raw[col].to_numpy(copy=True), self.params["X unit"]
                    ).cm #TODO: fix unit conversion
                if col == self.params["Measurement"] + " " + str(
                    int(self.params["Sample"][-1]) - 1
                ):
                    na_col = col
            elif "tof" in self.params["Type"].lower():
                if "x" in col.lower() or "depth" in col.lower():
                    self.data["Depth"] = rtu.Length(
                        data_raw[col].to_numpy(copy=True), self.params["X unit"]
                    ).cm #TODO: fix unit conversion
                if (
                    self.params["Ion"].lower() in col.lower()
                    and col_type in col.lower()
                ):
                    na_col = col
                if self.params["Matrix"] in col and "inten" in col.lower():
                    data_matrix = data_raw[col].to_numpy()
            elif "dsims" in self.params["Type"].lower():
                if "na time" in col.lower():
                    self.data["Depth"] = rtu.Length(
                        data_raw[col].to_numpy(copy=True)
                        * self.params["Max X"]
                        / data_raw[col].max(),
                        self.params["X unit"],
                    ).cm #TODO: fix unit conversion
                if (
                    self.params["Ion"].lower() in col.lower()
                    and col_type in col.lower()
                ):
                    na_col = col
                if " ".join([self.params["Matrix"].lower(), col_type]) in col.lower():
                    data_matrix = data_raw[col].to_numpy()
        if "counts" in self.params["Y unit"] and not np.isnan(self.params["RSF"]):
            self.data["Na"] = (
                data_raw[na_col].to_numpy() / np.mean(data_matrix) * self.params["RSF"]
            )
        elif "counts" in self.params["Y unit"] and not np.isnan(self.params["SF"]):
            self.data["Na"] = data_raw[na_col].to_numpy() * self.params["SF"]
        else:
            self.data["Na"] = data_raw[na_col].to_numpy()

    def limit_test(self, thresh=0.025):
        """Return sum of squared errors (pred vs actual)."""
        lin_loc, lin_lim = lin_test(
            self.data["Depth"].to_numpy(), self.data["Na"].to_numpy(), thresh
        )

        if lin_lim > self.fit_depth_cm * 1.1 or lin_lim < self.fit_depth_cm * 0.9:
            self.data_bgd["bgd_lim"] = rt.find_nearest(
                self.data["Depth"].to_numpy(), self.fit_depth_cm
            )
        else:
            self.data_bgd["bgd_lim"] = lin_loc

    def regress_test(self, alpha=0.05, ind_range=10, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        stop = len(self.data["Depth"])
        cng = int(len(self.data["Depth"]) * 0.02)
        perc = 0
        while perc < 0.2 and stop > len(self.data["Depth"]) * 0.5:
            self.p = np.ones(stop - 10)
            for x in range(stop - 10):
                coeff = stats.linregress(
                    self.data["Depth"].to_numpy()[x:stop],
                    self.data["Na"].to_numpy()[x:stop],
                    **kwargs
                )[:2]
                resid = self.data["Na"].to_numpy()[x:stop] - rt.line(
                    self.data["Depth"].to_numpy()[x:stop], coeff[0], coeff[1]
                )
                self.p[x] = stats.normaltest(resid)[1]
            stop -= cng
            perc = len(self.p[self.p > alpha]) / len(self.p)
        itr = 0
        while self.p[itr] < alpha and itr < int((len(self.data["Na"]) - 10) * 0.75):
            itr += 1
        self.data_bgd["bgd_max"] = itr

        ind = 0
        while ind < ind_range and itr < int((len(self.data["Na"]) - 10) * 0.9):
            ind += 1
            itr += 1
            if self.p[itr] < alpha:
                ind = 0
        self.data_bgd["bgd_min"] = itr - ind

        self.data_bgd["bgd_ave"] = int(
            (self.data_bgd["bgd_max"] + self.data_bgd["bgd_min"]) / 2
        )
        coeff = stats.linregress(
            self.data["Depth"].to_numpy()[self.data_bgd["bgd_ave"] :],
            self.data["Na"].to_numpy()[self.data_bgd["bgd_ave"] :],
            **kwargs
        )[:2]
        self.data_bgd["P-value"] = self.p[self.data_bgd["bgd_ave"]]
        self.data_bgd["slope"] = coeff[0]
        self.data_bgd["intercept"] = coeff[1]

    @property
    def thick_cm(self):
        """Return sum of squared errors (pred vs actual)."""
        return rtu.Length(self.params["Thick"], self.params["Thick unit"]).cm #TODO: fix unit conversion


class BaseProfile:
    """
    Store Profile information.

    This class is intended to store the information inherent in to a depth
    profile fit. This should include the fit profile, it's properties, and
    its range.  Curently also includes the fitted data and the resulting error.
    I'm not sure that it is important for this fitted information to be in a
    single class.
    """

    _type = "base"

    def __init__(self, obj):
        self.min_range = 2
        self.min_index = 0
        self.diff = 1e-15
        self.conc = 1e18
        self.error = 1
        self.p_value = 0

        self.data_bgd = obj.data_bgd.copy()
        self.depth = obj.data["Depth"].to_numpy()
        self.sims = obj.data["Na"].to_numpy()
        self.pred = np.ones_like(self.sims)
        self.max_index = len(self.depth) - 1
        # self.bgd_index = min(self.data_bgd['bgd_ave'],len(self.depth)-2)

        self.stats_obj = rt.Statistics(
            self.data[self.start_index : self.stop_index + 1],
        )
        self.stats_attr = "mean_abs_perc_err"

        self.std_values = pd.DataFrame(
            {
                "conc": (15, 18, 21),
                "diff": (-18, -15, -12),
                "depth": (0, int(self.max_index / 2), self.max_index),
            },
            index=("low", "mid", "high"),
        )
        self.conditions = pd.Series(
            {
                "thick": obj.thick_cm,
                "time": obj.params["Stress Time"],
                "temp": obj.params["Temp"],
                "e_field": obj.params["Volt"] / obj.thick_cm,
                "volt": obj.params["Volt"],
            }
        )
        self.info = pd.Series(
            {
                "ident": self.ident,
                "sample": obj.params["Sample"],
                # 'type': obj.params['Type'],
                "class": "BaseProfile",
            }
        )
        # 'measurement': obj.params['Measurement']})

    @property
    def data(self):
        """Return sum of squared errors (pred vs actual)."""
        _data = pd.DataFrame(
            columns=(
                "depth",
                "SIMS",
                "log(SIMS)",
                "pred",
                "log(pred)",
                "weight",
                "residuals",
                "residuals from stats",
            )
        )

        _data["depth"] = self.depth
        _data["SIMS"] = self.sims
        _data["log(SIMS)"] = np.log10(self.sims)
        _data["pred"] = self.pred
        _data["log(pred)"] = np.log10(self.pred)
        _data["weight"] = [
            self.pred[x] / self.sims[x] if self.pred[x] > self.sims[x] else 1
            for x in range(self.max_index + 1)
        ]
        _data_stats = rt.Statistics(_data)
        if hasattr(self, "stats_obj"):
            _data_stats.log_form = self.stats_obj.log_form
            _data_stats.resid_type = self.stats_obj.resid_type
        _data["residuals"] = self.sims - self.pred
        _data["residuals from stats"] = _data_stats.residuals
        return _data

    @property
    def pred(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_pred"):
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
        if not hasattr(self, "_error_log"):
            self._error_log = pd.DataFrame(
                columns=("diff", "conc", "start", "stop", "error", "p-value")
            )
        if value in self._error_log.index:
            value = max(self._error_log.index) + 1
        self._error_log.loc[value, :] = (
            self.diff,
            self.conc,
            self.start_index,
            self.stop_index,
            self.error,
            self.p_value,
        )

    @property
    def start_index(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_start_index"):
            self._start_index = 0
        return self._start_index

    @start_index.setter
    def start_index(self, value):
        if self._stop_index <= value or value < 0:
            print("Obj", self.ident, "atempted to set start index to", value)
        else:
            self._start_index = int(value)

    @property
    def stop_index(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_stop_index"):
            self._stop_index = self.max_index
        return self._stop_index

    @stop_index.setter
    def stop_index(self, value):
        if self._start_index >= value or value > self.max_index:
            print("Obj", self.ident, "atempted to set stop index to", value)
        else:
            self._stop_index = int(value)

    @property
    def diff(self):
        """Return sum of squared errors (pred vs actual)."""
        return rt.sig_figs_round(self._diff, 4)

    @diff.setter
    def diff(self, value):
        self._diff = value

    @property
    def conc(self):
        """Return sum of squared errors (pred vs actual)."""
        return rt.sig_figs_round(self._conc, 4)

    @conc.setter
    def conc(self, value):
        self._conc = value

    @property
    def start_loc(self):
        """Return sum of squared errors (pred vs actual)."""
        return rt.sig_figs_round(rtu.Length(self.depth[self.start_index], "cm").um, 5) #TODO: fix unit conversion

    @property
    def stop_loc(self):
        """Return sum of squared errors (pred vs actual)."""
        return rt.sig_figs_round(rtu.Length(self.depth[self.stop_index], "cm").um, 5) #TODO: fix unit conversion

    @property
    def index_range(self):
        """Return sum of squared errors (pred vs actual)."""
        return self.stop_index - self.start_index + 1

    @property
    def depth_range(self):
        """Return sum of squared errors (pred vs actual)."""
        return rt.sig_figs_round((self.stop_loc - self.start_loc), 5)

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
        if not hasattr(self, "_stats_settings"):
            self._stats_settings = vars(self.stats_obj)
        if (
            isinstance(args, (list, np.ndarray, tuple, dict))
            and len(args) == 2
            and args[0] in self._stats_settings
        ):
            self._stats_settings[args[0]] = args[1]


class PredProfile(BaseProfile):
    """
    Generate profile from diff, conc, and simulation parameters.

    Creates a simulated profile by fitting real data.
    """

    _type = "pred"

    def __init__(self, sims_obj, diff=None, conc=None, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        # constant once set
        super().__init__(sims_obj)

        if diff is not None:
            self.diff = diff
        if conc is not None:
            self.conc = conc
        self.unpack_kwargs(kwargs)

        self.pred = c_np(
            depth=self.depth,
            diff=self.diff,
            conc=self.conc,
            thick=self.conditions["thick"],
            temp=rt.convert_temp(self.conditions["temp"], "C", "K"),
            e_app=self.conditions["e_field"],
            time=self.conditions["time"],
        )

        self.info["class"] = "PredProfile"

        self.stats_obj = rt.Statistics(
            self.data[self.start_index : self.stop_index + 1],
        )
        self.stats_attr = "mean_abs_perc_err"

        self.error_log = 0

    def unpack_kwargs(self, kwargs):
        """Return sum of squared errors (pred vs actual)."""
        self.__dict__.update(kwargs)


class FitProfile(BaseProfile):
    """
    Generate profile from ?.

    Creates a simulated profile by ?.
    """

    _type = "fit"
    _curve_fit_keys = list(curve_fit.__code__.co_varnames) + ["x_scale", "xtol", "jac"]

    def __init__(self, sims_obj, start_index=None, stop_index=None, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        super().__init__(sims_obj)
        if start_index is not None:
            if isinstance(start_index, (float, np.float)):
                self.start_index = rt.find_nearest(self.depth, start_index) #TODO: fix unit conversion
            elif isinstance(start_index, (int, np.integer)):
                self.start_index = start_index
        if stop_index is not None:
            if isinstance(stop_index, (float, np.float)):
                index = rt.find_nearest(self.depth, stop_index) #TODO: fix unit conversion
                if index == self.start_index:
                    index += 1
                self.stop_index = index
            elif isinstance(stop_index, (int, np.integer)):
                self.stop_index = stop_index
        self.curve_fit_kwargs = {"x_scale": "jac", "xtol": 1e-12, "jac": "3-point"}
        self.unpack_kwargs(kwargs)

        self.fitter(**kwargs)

        self.info["class"] = "FitProfile"

        self.stats_obj = rt.Statistics(
            self.data[self.start_index : self.stop_index + 1],
        )
        self.stats_attr = "mean_abs_perc_err"

        self.error_log = 0

    def unpack_kwargs(self, kwargs):
        """Return sum of squared errors (pred vs actual)."""
        self.curve_fit_kwargs.update(
            {key: kwargs[key] for key in kwargs if key in self._curve_fit_keys}
        )
        [kwargs.pop(x) for x in self._curve_fit_keys if x in kwargs.keys()]
        self.__dict__.update(kwargs)

    def fitter(self, diff_pred=None, conc_pred=None, log_form=False, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        if diff_pred is None:
            diff_pred = self.std_values["diff"]
        if conc_pred is None:
            conc_pred = self.std_values["conc"]
        self.unpack_kwargs(kwargs)

        self.c_np_new = partial(
            c_np,
            thick=self.conditions["thick"],
            temp=rt.convert_temp(self.conditions["temp"], "C", "K"),
            e_app=self.conditions["e_field"],
            time=self.conditions["time"],
            log_form=log_form,
        )

        sims = self.sims
        if log_form:
            sims = self.data["log(SIMS)"].to_numpy()
        try:
            fittemp = curve_fit(
                self.c_np_new,
                self.depth[self.start_index : self.stop_index + 1],
                sims[self.start_index : self.stop_index + 1],
                p0=(diff_pred["mid"], conc_pred["mid"]),
                bounds=(
                    (diff_pred["low"], conc_pred["low"]),
                    (diff_pred["high"], conc_pred["high"]),
                ),
                **self.curve_fit_kwargs
            )
        except RuntimeError:
            self.fit_res = [self.diff, self.diff, self.conc, self.conc]
            print(self.start_index, "-", self.stop_index)
        else:
            self.fit_res = [
                10 ** fittemp[0][0],
                (
                    10 ** (fittemp[0][0] + np.sqrt(np.diag(fittemp[1]))[0])
                    - 10 ** (fittemp[0][0] - np.sqrt(np.diag(fittemp[1]))[0])
                )
                / 2,
                10 ** fittemp[0][1],
                (
                    10 ** (fittemp[0][1] + np.sqrt(np.diag(fittemp[1]))[1])
                    - 10 ** (fittemp[0][1] - np.sqrt(np.diag(fittemp[1]))[1])
                )
                / 2,
            ]
        self.diff = self.fit_res[0]
        self.conc = self.fit_res[2]

        self.pred = np.array(
            self.c_np_new(self.depth, self.diff, self.conc, log_form=False)
        )

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

    _type = "profile_operator"
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
        self.error_kwargs = {
            key: kwargs[key] for key in kwargs if key in self._error_keys
        }
        [kwargs.pop(x) for x in self._error_keys if x in kwargs.keys()]
        self.__dict__.update(kwargs)

    @property
    def ident(self):
        """Return sum of squared errors (pred vs actual)."""
        return id(self)

    @property
    def start(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_start"):
            self._start = 0
        return self._start

    @start.setter
    def start(self, value):
        if value is None:
            self._start = self.prof.start_index
        elif self._stop <= value or value < 0:
            print("Obj", self.ident, "atempted to set start to", value)
        else:
            self._start = int(value)

    @property
    def stop(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_stop"):
            self._stop = self.max_index
        return self._stop

    @stop.setter
    def stop(self, value):
        if value is None:
            self._stop = self.prof.stop_index
        elif self._start >= value or value > self.prof.max_index:
            print("Obj", self.ident, "atempted to set stop to", value)
        else:
            self._stop = int(value)

    @property
    def w_constant(self):
        """Calculate constant, may shift to use the depth range instead."""
        return self._w_constant

    @w_constant.setter
    def w_constant(self, instr="logic"):
        self.w_range = len(self.prof.sims)

        if instr.lower() == "logic":
            vals_incl = len(
                self.prof.sims[self.start : self.stop + 1][
                    (self.prof.sims > self.prof.pred)[self.start : self.stop + 1]
                ]
            )
        elif instr.lower() == "base":
            vals_incl = len(self.prof.sims[self.start : self.stop + 1])
        else:
            vals_incl = self.w_range
        if vals_incl <= self.prof.min_range:
            vals_incl = self.prof.min_range
        self._w_constant = self.w_range / (vals_incl)

    def set_error(
        self,
        start=None,
        stop=None,
        save_res=True,
        instr="None",
        use_sample_w=False,
        w_array=None,
        log_form=False,
        **kwargs
    ):
        """
        Calculate error.

        error for the input information, information generated at call,
        requires type input for constant, can pass the information to sample
        weights if desired. can rewrite to always pass sample_weights via
        kwargs.
        """

        if (
            self.prof.std_values["diff"].isin([np.log10(self.prof.diff)]).any()
            or self.prof.std_values["conc"].isin([np.log10(self.prof.conc)]).any()
        ):
            self.error = 1
        else:
            self.start = start
            self.stop = stop

            self.unpack_kwargs(kwargs)

            if use_sample_w and w_array is None:
                self.error_kwargs["sample_weight"] = self.data["weight"].to_numpy()[
                    self.start : self.stop + 1
                ]
            self.w_constant = str(instr)

            self.ops_stats = rt.Statistics(
                self.data[self.start : self.stop + 1], log_form, **self.error_kwargs
            )

            self.error = self.ops_stats.mean_abs_perc_err * self.w_constant
        if save_res:
            self.prof.error = self.error
        self.prof.error_log = 0

        return self.error

    def set_best_error(
        self,
        use_index=True,
        x_in=-1,
        reset=True,
        reverse=False,
        save_res=True,
        **kwargs
    ):
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
        err_array = np.array(
            [
                self.set_error(start=x, stop=x_in, save_res=False, **kwargs)
                for x in range(x_in - self.prof.min_range + 1)
            ]
        )

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



class MatrixOps:
    """Higher level operator"""

    _type = "matrix_operator"

    def __init__(
        self,
        sims_obj,
        cls_type,
        xrange=[None, None, None, None],
        yrange=[None, None, None, None],
        min_range=2,
        size=50,
        **kwargs
    ):
        self.cls_type = cls_type
        self.size = size
        self.std_values = BaseProfile(sims_obj).std_values
        self.max_ind = sims_obj.data.index.max()
        self.max_depth = sims_obj.data["Depth"].max()

        if "fit" in cls_type.lower() and xrange[0] is None:
            xrange = ["depth", None, None, "index"]
        if "fit" in cls_type.lower() and yrange[0] is None:
            yrange = ["depth", None, None, "index"]
        if "pred" in cls_type.lower() and xrange[0] is None:
            xrange = ["conc", None, None, "log"]
        if "pred" in cls_type.lower() and yrange[0] is None:
            yrange = ["diff", None, None, "log"]
        self.xrange = xrange
        self.yrange = yrange

        self.obj_operator = Composite()
        if "fit" in cls_type.lower():
            range_lim = self.xrange[min_range] - self.xrange[0]
            [
                self.obj_operator.add(
                    ProfileOps(
                        FitProfile(sims_obj, start_index=x, stop_index=y, **kwargs),
                        **kwargs
                    )
                )
                for x in self.xrange
                for y in self.yrange
                if (x + range_lim <= y)
            ]
        if "pred" in cls_type.lower():
            [
                self.obj_operator.add(
                    ProfileOps(
                        PredProfile(sims_obj, diff=y, conc=x, **kwargs), **kwargs
                    )
                )
                for x in self.xrange
                for y in self.yrange
            ]
        if self.obj_operator._family[0].prof.min_range != min_range:
            self.obj_operator.set_attr(attr="min_range", num=min_range, limit=False)

    @property
    def ident(self):
        """Return sum of squared errors (pred vs actual)."""
        return id(self)

    @property
    def xrange(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_xrange"):
            self.xrange = ["depth", None, None, "index"]
        return self._xrange

    @xrange.setter
    def xrange(self, value):
        if value[1] is None:
            value[1] = self.std_values.loc["low", value[0]]
        if value[2] is None:
            value[2] = self.std_values.loc["high", value[0]]
        if "ind" in value[3].lower():
            self._xrange = np.linspace(
                value[1],
                min(value[2], self.max_ind),
                min(self.size, self.max_ind + 1),
                dtype=int,
            )
        elif "lin" in value[3].lower():
            self._xrange = np.linspace(
                value[1],
                min(value[2], self.max_depth),
                min(self.size, self.max_ind + 1),
            )
        elif "log" in value[3].lower():
            self._xrange = np.logspace(
                value[1], value[2], min(self.size, self.max_ind + 1)
            )
        else:
            self._xrange = np.array(range(min(self.size, self.max_ind + 1)))

    @property
    def yrange(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, "_yrange"):
            self.yrange = ["depth", None, None, "index"]
        return self._yrange

    @yrange.setter
    def yrange(self, value):
        if value[1] is None:
            value[1] = self.std_values.loc["low", value[0]]
        if value[2] is None:
            value[2] = self.std_values.loc["high", value[0]]
        if "ind" in value[3].lower():
            self._yrange = np.linspace(
                value[1],
                min(value[2], self.max_ind),
                min(self.size, self.max_ind + 1),
                dtype=int,
            )
        elif "lin" in value[3].lower():
            self._yrange = np.linspace(
                value[1],
                min(value[2], self.max_depth),
                min(self.size, self.max_ind + 1),
            )
        elif "log" in value[3].lower():
            self._yrange = np.logspace(
                value[1], value[2], min(self.size, self.max_ind + 1)
            )
        else:
            self._yrange = np.array(range(min(self.size, self.max_ind + 1)))

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
            if obj._type == "matrix_operator":
                self.parent_obj = obj
                self.composite = obj.obj_operator
            elif obj._type == "composite":
                self.parent_obj = None
                self.composite = obj
            elif obj._type == "profile_operator":
                self.parent_obj = obj
                self.profile = obj.prof
            elif obj._type == "fit" or obj._type == "pred" or obj._type == "base":
                self.parent_obj = None
                self.profile = obj
        self.info = info

    @property
    def family_df(self):
        """Return sum of squared errors (pred vs actual)."""
        if hasattr(self, "composite"):
            return self.composite.gen_df()
        else:
            return None

    @property
    def depth(self):
        """Return sum of squared errors (pred vs actual)."""
        if hasattr(self, "composite"):
            return self.composite._family[0].data["depth"].to_numpy() * 1e4
        else:
            return None

    @property
    def data(self):
        """Return sum of squared errors (pred vs actual)."""
        if hasattr(self, "profile"):
            self._data = self.profile.data.copy()
            self._data["depth"] = self._data["depth"] * 1e4
            return self._data
        else:
            return None

    @property
    def info(self):
        """Return sum of squared errors (pred vs actual)."""
        return self._info

    @info.setter
    def info(self, val):
        self._info = ["start_index", "stop_index", "error"]
        if isinstance(val, (list, tuple)) and len(val) == 3:
            self._base_info = ["start_index", "stop_index", "error"]
            self._info = [
                x if isinstance(x, str) else self._base_info[i]
                for i, x in enumerate(val)
            ]
            if "stats." in self._info[2].lower():
                self.info[2], attr = self._info[2].split(".")
                self.composite.set_attr("stats_attr", attr)

    @property
    def matrix(self):
        """Return matrix via pivot table."""
        return self.family_df.pivot_table(
            values=self.info[2], columns=self.info[0], index=self.info[1]
        )

    def focus(self, pairs=None, pair_names=["start_index", "index_range"], var=[]):
        """Return sum of squared errors (pred vs actual)."""
        if var == []:
            var = [
                "start_index",
                "stop_index",
                "index_range",
                "start_loc",
                "stop_loc",
                "diff",
                "conc",
                "error",
                "stats",
            ]
        df = self.composite.gen_df(var=list(np.unique(var + pair_names)))

        self.focus_df = df.pivot_table(index=pair_names)
        for check in var:
            if check not in self.focus_df.columns and check not in pair_names:
                self.focus_df[check] = df[check].to_numpy()
        if pairs is not None:
            self.focus_df = self.focus_df.loc[self.focus_df.index.intersection(pairs)]
        self.focus_df = self.focus_df[~self.focus_df.index.duplicated(keep="first")]

        return self.focus_df

    def pks_generate(
        self,
        num_of_peaks=2,
        stop_max=2.5,
        start_max=2,
        start_min=2,
        range_min=1,
        overlap=True,
    ):
        """Return sum of squared errors (pred vs actual)."""
        limited_df = (
            self.family_df[
                (self.family_df["stop_loc"] < stop_max)
                & (self.family_df["start_loc"] < start_max)
                & (self.family_df["index_range"] >= range_min)
                & (self.family_df["start_index"] > start_min)
            ]
            .copy()
            .dropna()
        )

        limited_df[["min", "max"]] = 0

        limited_df = limited_df.sort_values(by=["index_range", "start_index"])
        limited_df = limited_df.reset_index(drop=True)

        limited_df["min"][
            (limited_df["error"].shift(1) > limited_df["error"])
            & (limited_df["error"].shift(-1) > limited_df["error"])
        ] = -1
        limited_df["max"][
            (limited_df["error"].shift(1) < limited_df["error"])
            & (limited_df["error"].shift(-1) < limited_df["error"])
        ] = 1

        limited_df["max"][
            (limited_df["stop_index"] == limited_df["stop_index"].max())
        ] = 0
        limited_df["max"][
            (limited_df["start_index"] == limited_df["start_index"].min())
        ] = 0

        self.limited_error = pivot_cleaner(
            limited_df.pivot_table(
                index="index_range", columns="start_index", values="error"
            )
        )

        start_min_df = pivot_cleaner(
            limited_df.pivot_table(
                index="index_range", columns="start_index", values="min"
            )
        )
        start_max_df = pivot_cleaner(
            limited_df.pivot_table(
                index="index_range", columns="start_index", values="max"
            )
        )
        self.comb_start_df = start_min_df + start_max_df

        valley_leng = pd.DataFrame()
        valley_leng["min_sum"] = start_min_df.sum()
        valley_leng["max_sum"] = start_max_df.sum()
        valley_leng["comb_sum"] = valley_leng["min_sum"] + valley_leng["max_sum"]
        valley_leng["comb_res"] = valley_leng["comb_sum"]

        for x in range(len(valley_leng) - 1):
            if valley_leng.iloc[x, 3] < 0:
                outer_int = x
                inner_int = x + 1
                while (
                    inner_int < len(valley_leng) and valley_leng.iloc[inner_int, 3] < 0
                ):
                    valley_leng.iloc[outer_int, 3] += valley_leng.iloc[inner_int, 3]
                    valley_leng.iloc[inner_int, 3] = 0
                    inner_int += 1
        ave_peak = int(valley_leng["comb_res"][valley_leng["comb_res"] < 0].median())

        min_peaks = valley_leng["comb_res"][(valley_leng["comb_res"] <= ave_peak)]

        peak_df = pd.DataFrame(
            columns=["start_index", "count", "left_bound", "right_bound", "best_range"]
        )
        peak_df["start_index"] = min_peaks.index.to_numpy()
        peak_df["count"] = min_peaks.to_numpy()
        peak_df["left_bound"] = peak_df["start_index"].shift(1)
        peak_df["right_bound"] = peak_df["start_index"].shift(-1)

        peak_df.iloc[0, 2] = 0
        peak_df.iloc[-1, 3] = limited_df["stop_index"].max()

        limited_df = limited_df.sort_values(by=["start_index", "index_range"])
        limited_df = limited_df.reset_index(drop=True)
        limited_df["change"] = limited_df["error"].pct_change(1)

        limited_df["change"] = limited_df["change"].fillna(0)
        limited_df["change"][limited_df["change"] >= 0] = 1
        limited_df["change"][limited_df["change"] < 0] = 0

        grp_all = limited_df.groupby(["start_index"], as_index=False, sort=False)
        no_chng = grp_all.agg({"change": "nunique", "index_range": "idxmax"})[
            grp_all["change"].nunique()["change"] <= 1
        ]

        limited_df["change"][no_chng["index_range"]] = 0

        grp_mins = limited_df[limited_df["change"] == 0].groupby(
            ["start_index"], as_index=False, sort=False
        )
        by_range = grp_mins[self.family_df.columns].tail(1)

        peak_df["best_range"] = by_range["index_range"][
            by_range["start_index"].isin(peak_df["start_index"])
        ].to_numpy()

        max_df = limited_df[["start_index", "index_range", "error", "max"]]

        max_df = max_df.sort_values(by=["index_range", "start_index"])

        range_df = pd.DataFrame(columns=["index_range"])
        range_df["index_range"] = max_df["index_range"].unique()

        peak_dict = {}
        for n in range(len(min_peaks)):
            peak_dict[n] = list(
                limited_df[["start_index", "index_range"]][
                    limited_df["start_index"] == min_peaks.index[n]
                ].itertuples(index=False, name=None)
            )
            max_df["start_shift_%d" % n] = max_df["start_index"] - min_peaks.index[n]

            for i, x in enumerate(range_df["index_range"]):
                range_df.loc[i, "start_min_%d" % n] = max_df["start_index"][
                    (max_df["start_shift_%d" % n] < 0)
                    & (max_df["max"] > 0)
                    & (max_df["index_range"] == x)
                ].max()
                if np.isnan(range_df.loc[i, "start_min_%d" % n]).all():
                    range_df["start_min_%d" % n] = range_df["start_min_%d" % n].fillna(
                        min_peaks.index[n]
                    )
                range_df.loc[i, "start_max_%d" % n] = max_df["start_index"][
                    (max_df["start_shift_%d" % n] > 0)
                    & (max_df["max"] > 0)
                    & (max_df["index_range"] == x)
                ].min()
                if np.isnan(range_df.loc[i, "start_max_%d" % n]).all():
                    range_df["start_max_%d" % n] = range_df["start_max_%d" % n].fillna(
                        min_peaks.index[n]
                    )
        range_df = range_df.astype(int)

        range_dict = {
            i: (
                list(
                    range_df.loc[:, ("start_min_%d" % i, "index_range")].itertuples(
                        index=False, name=None
                    )
                )
                + list(
                    range_df.loc[:, ("start_max_%d" % i, "index_range")].itertuples(
                        index=False, name=None
                    )
                )
            )
            for i, n in enumerate(min_peaks.index)
        }

        region_dict = {}
        region_dict["all"] = []

        for n in range(min(num_of_peaks, len(min_peaks))):
            region_dict[n] = peak_dict[n] + range_dict[n]
            region_dict["all"] += region_dict[n]
        by_error = limited_df.loc[
            grp_all["error"].idxmin()["error"], self.family_df.columns
        ]

        self.region_dict = region_dict
        self.peak_dict = peak_dict
        self.peak_df = peak_df

        self.focus_dict = {"focus" + str(x): None for x in self.peak_df.index}
        self.focii_dict = {"focii" + str(x): None for x in self.peak_df.index}

        return (
            {"ridge pairs": self.region_dict, "valley pairs": self.peak_dict},
            {"valley lengths": valley_leng, "by_range": by_range, "by_error": by_error},
            self.peak_df,
        )

    def pks_analyze(
        self,
        peak=0,
        min_start=None,
        max_start=None,
        min_range=3,
        peak_range=None,
        max_range=None,
        pair_set=None,
        old_range=False,
        pair_names=["start_index", "index_range"],
        **kwargs
    ):
        """Evaluate for removal."""

        if not hasattr(self, "region_dict") or not hasattr(self, "peak_dict"):
            self.pks_generate()
        if pair_set is None or not old_range:
            pair_set = peak
        if old_range:
            pairs = self.region_dict[pair_set]
        else:
            pairs = self.peak_dict[pair_set]
        if min_start is None:
            min_start = self.peak_df.iloc[peak, 2]
        if max_start is None:
            max_start = self.peak_df.iloc[peak, 3]
        if peak_range is None or peak_range > self.limited_error.index.max():
            peak_range = self.peak_df.iloc[peak, 4]
        if max_range is None or max_range > np.array(pairs)[:, 1].max():
            max_range = np.array(pairs)[:, 1].max()
        peak_range = rt.find_nearest(
            self.limited_error.index.to_numpy(), peak_range, False
        )
        # converts error below peak value to negatives and changes to 1's & 0's
        # uses loc to get location
        ranges = (
            self.limited_error
            - self.limited_error.loc[peak_range, self.peak_df.iloc[peak, 0]]
        )
        ranges[ranges <= 0] = 1
        ranges[ranges < 1] = 0
        # logic: term1: lowest last max aka valley < expected start
        # and range value + 1 of valley <= max possible range
        # convert to 1.  essentially makes all values below ridge to be 12
        while (
            ranges[::-1].idxmax().idxmin() <= self.peak_df.iloc[peak, 0]
            and ranges[::-1].idxmax().min() + 1
            <= ranges[self.peak_df.iloc[peak, 0]].dropna().index.max()
        ):
            idx = ranges.index.get_loc(ranges[::-1].idxmax().min())
            ranges.loc[ranges.index[idx + 1], ranges[::-1].idxmax().idxmin()] = 1
            ranges = ranges.sort_index()
        ridgeline = ranges[::-1].idxmax()

        ridge = pd.DataFrame(
            [
                [
                    (
                        1
                        if (y <= ridgeline[x] and x <= ranges[::-1].idxmax().idxmin())
                        else 0
                    )
                    for x in ridgeline.index
                ]
                for y in ranges.index
            ],
            columns=ranges.columns,
            index=ranges.index,
        )

        new_ranges = [
            (x, y) for x in ridge.columns for y in ridge.index if ridge.loc[y, x]
        ]
        pairs = list(pd.Series((pairs + new_ranges)).unique())
        # makes sure that pair set start over input min and stop over input max
        pairs_new = [
            x
            for x in pairs
            if x[0] >= min_start and x[0] <= max_start and x[1] <= max_range
        ]

        focii = self.focus(pairs=pairs_new, pair_names=pair_names)
        focii = focii[
            (focii["diff"] != 1e-11)
            & (focii.index.get_level_values(0) < max_start)
            & (  # makes  sure that start is under desired stop (stay in range)
                focii.index.get_level_values(1) > min_range
            )
            & (  # makes sure that range always long enough
                focii["stop_index"] > self.peak_df.iloc[peak, 0]
            )
        ]  # makes sure that stop never before the local min -> range always straddles local min

        focus_stats = focii.describe()
        self.focus_dict["focus" + str(peak)] = focus_stats
        self.focii_dict["focii" + str(peak)] = focii
        return focus_stats, focii

    def auto_focus(
        self,
        min_start=None,
        max_start=None,
        min_range=2,
        peak_range=None,
        max_range=None,
        pair_set=None,
        old_range=False,
    ):

        for peak in self.peak_df.index:
            if pair_set != "all":
                pair_set = peak
            focus_stats, focii = self.pks_analyze(
                peak=peak,
                min_start=min_start,
                max_start=max_start,
                min_range=min_range,
                peak_range=peak_range,
                max_range=max_range,
                pair_set=pair_set,
                old_range=old_range,
            )
            self.auto_focus_df = self.focus_report(peak)
        return self.auto_focus_df.copy()

    def focus_report(self, peak=0, df=None, focus_stats=None, focii=None):
        if not hasattr(self, "auto_focus_df"):
            df = pd.DataFrame(
                columns=[
                    "count",
                    "error",
                    "error std",
                    "diff",
                    "diff std",
                    "conc",
                    "conc std",
                    "min index",
                    "min loc",
                    "Start indices",
                    "Range indices",
                    "Stop indices",
                    "Start locs",
                    "Range locs",
                    "Stop locs",
                ],
                index=self.peak_df.index,
                dtype=float,
            )
            df["min index"] = self.peak_df["start_index"]
            df["min loc"] = [self.depth[x] for x in df["min index"]]
            df = df.fillna(0)
        elif df is None:
            df = self.auto_focus_df.copy()
        if focus_stats is None:
            focus_stats = self.focus_dict["focus" + str(peak)]
        if focii is None:
            focii = self.focii_dict["focii" + str(peak)]
        df.loc[peak, "count"] = focus_stats.loc["count", "diff"]
        df.loc[peak, "error"] = focus_stats.loc["mean", "error"]
        df.loc[peak, "error std"] = focus_stats.loc["std", "error"]
        df.loc[peak, "diff"] = focus_stats.loc["mean", "diff"]
        df.loc[peak, "diff std"] = focus_stats.loc["std", "diff"]
        df.loc[peak, "conc"] = float(focus_stats.loc["mean", "conc"])
        df.loc[peak, "conc std"] = focus_stats.loc["std", "conc"]

        df.loc[:, "Start indices":"Stop locs"] = df.loc[
            :, "Start indices":"Stop locs"
        ].astype(object)
        df.at[peak, "Start indices"] = [
            focii.index.get_level_values(0).min(),
            rt.find_nearest(self.depth, focus_stats.loc["mean", "start_loc"]),
            focii.index.get_level_values(0).max(),
        ]
        df.at[peak, "Range indices"] = [
            focii.index.get_level_values(1).min(),
            self.peak_df.iloc[peak, 4],
            focii.index.get_level_values(1).max(),
        ]
        df.at[peak, "Stop indices"] = [
            int(focus_stats.loc["min", "stop_index"]),
            int(focus_stats.loc["mean", "stop_index"]),
            int(focus_stats.loc["max", "stop_index"]),
        ]

        df.at[peak, "Start locs"] = [
            focus_stats.loc["min", "start_loc"],
            focus_stats.loc["mean", "start_loc"],
            focus_stats.loc["max", "start_loc"],
        ]
        df.at[peak, "Range locs"] = [
            self.depth[df.loc[peak, "Range indices"][0]],
            self.depth[df.loc[peak, "Range indices"][1]],
            self.depth[df.loc[peak, "Range indices"][2]],
        ]
        df.at[peak, "Stop locs"] = [
            focus_stats.loc["min", "stop_loc"],
            focus_stats.loc["mean", "stop_loc"],
            focus_stats.loc["max", "stop_loc"],
        ]

        return df

    def check_error(self, **kwargs):
        """Evaluate for removal."""
        self.error_matrix = self.obj_matrix.applymap(
            lambda x: (
                ProfileOps(x.data, x.pred, x.start_index, x.stop_index)
                if not isinstance(x, (int, np.integer))
                else 1
            )
        )

        return self.error_matrix.applymap(
            lambda x: x.err(**kwargs) if not isinstance(x, (int, np.integer)) else 1
        )

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
                self.profile_list.append(
                    FitProfile(sims_obj, start_index=start, stop_index=stop)
                )
                profile = self.profile_list[-1].pred
                if start <= 1:
                    start = 0
                if stop == 199:
                    stop = 200
                [
                    (self.indexed.append(x), self.profile_num.append(num))
                    for x in profile[start:stop]
                ]

                num += 1
            self.stitched_res = BaseProfile(sims_obj)
            self.stitched_res.pred = np.array(self.indexed)
            self.stitched_res._data["Range number"] = np.array(self.profile_num)

            return self.stitched_res


class Plotter(Analysis):
    """Return sum of squared errors (pred vs actual)."""

    def __init__(self, obj=None, info=[None, None, None], **kwargs):
        super().__init__(obj, info)

    def map_plot(
        self,
        name=None,
        info=[None, None, None],
        matrix=None,
        conv=[1, 1],
        zlog=True,
        **kwargs
    ):
        """Return sum of squared errors (pred vs actual)."""
        conv_info = None
        if "depth_range" in info:
            for i, x in enumerate(info):
                if x == "depth_range":
                    info[i] = "index_range"
                    conv_info = i
        if matrix is None and not hasattr(self, "composite"):
            print("Insert matrix!")
            return
        elif matrix is None:
            self.info = info
            to_plot = self.matrix
        else:
            to_plot = matrix
        if name is None:
            name = "Fit Profile"
        to_plot = pivot_cleaner(to_plot)

        if conv_info == 0 and self.depth is not None:
            try:
                to_plot.columns = [self.depth[x] for x in to_plot.columns]
            except IndexError:
                to_plot.columns = [self.depth[x - 1] for x in to_plot.columns]
        if conv_info == 1 and self.depth is not None:
            try:
                to_plot.index = [self.depth[x] for x in to_plot.index]
            except IndexError:
                to_plot.index = [self.depth[x - 1] for x in to_plot.index]
        # standard plot information
        plt_kwargs = {
            "name": name,
            "xname": "Start Point (um)",
            "yname": "End Point (um)",
            "zname": "Error",
            "cmap": "kindlmann",
        }
        plt_kwargs.update({key: kwargs[key] for key in kwargs})

        rt.map_plt(
            to_plot.columns * conv[0],
            to_plot.index * conv[1],
            np.ma.masked_invalid(to_plot.to_numpy()),
            **plt_kwargs
        )

        plt.show()

    def prof_plot(self, name=None, data_in=None, **kwargs):
        """Return sum of squared errors (pred vs actual)."""
        multi_plot = False
        if data_in is None and not hasattr(self, "data"):
            print("Insert data!")
            return
        elif data_in is None:
            to_plot = self.data
        else:
            if data_in.index.names[0] is not None:
                data_in = data_in.reset_index()
                if "pred" in data_in.columns:
                    multi_plot = True
            to_plot = data_in
        if name is None:
            name = "Residual Plot"
        plt_kwargs = {
            "name": name,
            "xname": "Depth (um)",
            "yname": "Residuals",
            "palette": "kindlmann",
        }

        plt_kwargs.update({key: kwargs[key] for key in kwargs})
        if not multi_plot:
            rt.scatter(data=to_plot, **plt_kwargs)
        else:
            for pair in to_plot.index:
                plot_dict = to_plot.loc[pair, :].to_dict()
                plot_df = pd.DataFrame(plot_dict)
                rt.scatter(data=plot_df, **plt_kwargs)
        plt.show()
