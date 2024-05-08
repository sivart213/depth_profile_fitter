# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:24:26 2022

@author: j2cle
"""
# %% import section
import numpy as np
import pandas as pd
import xarray as xr
import General_functions as gf
from itertools import islice, combinations
import Units_Primary3 as up
import matplotlib.pyplot as plt
import seaborn as sns
import abc
import re
from scipy.special import erfc
from scipy import stats
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
from sklearn import metrics
from functools import partial
from dataclasses import field, fields, astuple, dataclass, InitVar

from scipy.optimize import curve_fit
import warnings


warnings.simplefilter("ignore", np.RankWarning)
warnings.filterwarnings("ignore")


def lin_test(x, y, lim=0.025):
    """Return sum of squared errors (pred vs actual)."""
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


def depth_conv(data_in, unit, layer_act, layer_meas):
    """Return sum of squared errors (pred vs actual)."""
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
            data_out = gf.tocm(data_in, unit)
    return data_out


def linear(x, coeffs):
    """Return sum of squared errors (pred vs actual)."""
    return coeffs[1] + coeffs[0] * x


def arr_like(arr, val=0):
    try:
        info = arr.dims.mapping
        zero = np.ones(list(info.values())) * val
    except AttributeError:
        zero = np.ones_like(arr) * val
    return xr.DataArray(zero, coords=arr.coords)


def set_like(arr, val=0, name="results"):
    return arr_like(arr, val).to_dataset(name=name)


def grow_arr(func, num, dim="new"):
    for n in range(num):
        temp_arr = func(n)
        if n == 0:
            arr = xr.concat([temp_arr], dim=dim)
        else:
            arr = xr.concat([arr, temp_arr], dim=dim)
    return arr


def gen_groups(bins, ref_arr, arr, blur=0, new=False):
    ref_temp_arr = xr.DataArray(
        gaussian_filter(ref_arr, sigma=blur), coords=ref_arr.coords
    )
    while np.isnan(ref_temp_arr).all() and blur > 0:
        blur -= 0.5
        ref_temp_arr = xr.DataArray(
            gaussian_filter(ref_arr, sigma=blur), coords=ref_arr.coords
        )
    if not np.isnan(ref_temp_arr).all():
        ref_arr = ref_temp_arr

    def func(loc):
        return xr.where(
            (bins[loc].left < ref_arr) & (ref_arr <= bins[loc].right), 1, np.nan
        )

    tmp_arr = grow_arr(func, len(bins), "grp")
    # for loc in range(len(bins)):
    #     tmp_arr = xr.where((bins[loc].left < ref_arr) & (ref_arr <= bins[loc].right), 1, np.nan)
    #     arr = grow_arr(arr, tmp_arr, loc, 'grp')
    return tmp_arr


class ImportFunc:
    def __init__(self, *args, func=""):
        self.func = func
        self.func(*args)

    @property
    def func(self):
        """Calculate constant, may shift to use the depth range instead."""
        return self._func

    @func.setter
    def func(self, value):
        if hasattr(self, value):
            self._func = getattr(self, value)
        else:
            self._func = getattr(self, "error")

    def nrel_d(self, *args):
        self.data_raw = pd.read_excel(
            args[0], sheet_name=args[1], usecols=args[2]
        ).dropna()
        return self.data_raw

    def asu_raw(self, *args):
        header_in = pd.read_csv(
            args[0], delimiter="\t", header=None, skiprows=14, nrows=2
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
        self.data_raw = (
            pd.read_csv(
                args[0],
                delimiter="\t",
                header=None,
                names=headers,
                index_col=False,
                skiprows=16,
            )
            .dropna()
            .astype(float)
        )
        return self.data_raw

    def rice_treated(self, *args):
        header_in = pd.read_csv(
            args[0], delimiter="\t", header=None, skiprows=2, nrows=3
        ).dropna(axis=1, how="all")
        header_in = header_in.fillna(method="ffill", axis=1)
        headers = [
            header_in.iloc[0, x] + " " + header_in.iloc[2, x]
            for x in range(header_in.shape[1])
        ]
        head_rem = ["#", "(nm)", "(Background Corrected)", "entration / atom cm^-3"]
        for rem in head_rem:
            headers = [x.replace(rem, "").strip() for x in headers]
        headers = [x.replace(" ", "_").strip().lower() for x in headers]

        self.data_raw = pd.read_csv(
            args[0],
            delimiter="\t",
            header=None,
            names=headers,
            index_col=False,
            skiprows=5,
        )
        return self.data_raw

    def rice_semi_treated(self, *args):
        return self.rice_treated(*args)

    def rice_raw(self, *args):

        self.data_raw = pd.read_csv(
            args[0],
            delimiter="\s+",
            header=None,
            index_col=[0, 1, 2],
            names=["x", "y", "z", "intens"],
            skiprows=10,
            dtype="int",
        )
        return self.data_raw


class ConvFunc:
    def __init__(self, raw_data, params, func="", **kwargs):
        self.data = pd.DataFrame(np.ones((len(raw_data), 2)), columns=["Depth", "Na"])
        self.params = params
        self.func = func
        self.func(raw_data, **kwargs)

    @property
    def func(self):
        """Calculate constant, may shift to use the depth range instead."""
        return self._func

    @func.setter
    def func(self, value):
        if hasattr(self, value):
            self._func = getattr(self, value)
        else:
            self._func = getattr(self, "error")

    def error(self, *args):
        print("available func not set")
        return

    def gen_col(self, df, a, b):
        return df[
            df.columns[[(a in x.lower()) and (b in x.lower()) for x in df.columns]]
        ].to_numpy(copy=True)

    def nrel_d(self, raw):
        self.data["Depth"] = depth_conv(
            self.gen_col(raw, "na", "x"),
            self.params["X unit"],
            self.params["Layer (actual)"],
            self.params["Layer (profile)"],
        )
        self.data["Na"] = (
            self.gen_col(raw, "na", "y")
            / self.gen_col(raw, "12c", "y").mean()
            * self.params["RSF"]
        )
        return self.data

    def asu_raw(self, raw):
        rate = self.params["Max X"] / self.gen_col(raw, "na", "time").max()
        self.data["Depth"] = gf.tocm(
            self.gen_col(raw, "na", "time") * rate, self.params["X unit"]
        )
        self.data["Na"] = (
            self.gen_col(raw, "na", "c/s")
            / self.gen_col(raw, "12c", "c/s").mean()
            * self.params["RSF"]
        )
        return self.data

    def rice_semi_treated(self, raw):
        self.data["Depth"] = gf.tocm(
            self.gen_col(raw, "depth", "dep"), self.params["X unit"]
        )

        if "counts" in self.params["Y unit"] and not np.isnan(self.params["RSF"]):
            self.data["Na"] = (
                self.gen_col(raw, "na+", "intens")
                / self.gen_col(raw, "c_2h_5", "intens").mean()
                * self.params["RSF"]
            )
        elif "counts" in self.params["Y unit"] and not np.isnan(self.params["SF"]):
            self.data["Na"] = self.gen_col(raw, "na+", "intens") * self.params["SF"]
        else:
            self.data["Na"] = self.gen_col(raw, "na+", "intens")
        return self.data

    def rice_treated(self, raw):
        self.data["Depth"] = gf.tocm(
            self.gen_col(raw, "depth", "dep"), self.params["X unit"]
        )
        self.data["Na"] = self.gen_col(raw, "na+", "conc")
        return self.data

    def rice_raw(self, raw, **kwargs):
        res_dict = PixelConv(raw, self.params, **kwargs).res_dict
        samp = self.params["Sample"]
        self.data = {}
        for key, val in res_dict.items():
            if "conc" in key:
                val = val.loc[:, (val != 0).any(axis=0)]
                val = val.set_index(
                    up.Length(val.index.to_numpy(), "um").cm
                ).reset_index()
                res_alt = {
                    f"{samp}-{key}-{col}": val[["index", col]].rename(
                        columns={"index": "Depth", col: "Na"}
                    )
                    for col in val.columns[1:]
                }
                self.data = {**self.data, **res_alt}
        return self.data


class PixelConv:
    def __init__(
        self,
        raw_data,
        params,
        devs=2,
        surf_lyr=0.2,
        back_lyr=0.2,
        slc_strt=2,
        slc_stp=5,
        num_bins=10,
        set_sigma=2,
        set_mult=0.5,
        save=True,
        **kwargs,
    ):
        self.devs = devs
        self.surf_lyr = surf_lyr
        self.back_lyr = back_lyr
        self.slc_strt = slc_strt
        self.slc_stp = slc_stp
        self.num_bins = num_bins
        self.set_sigma = set_sigma
        self.set_mult = set_mult

        self.params = params
        self.sample = self.params["Sample"]

        self.outpath = (
            "C:\\Users\\j2cle\\Work Docs\\Data\\Analysis\\SIMS\\RICE TOF\\220425 Man"
        )

        data_raw = ImportFunc(
            self.params["Reference File"], func="rice_treated"
        ).data_raw

        data_ref = data_raw.to_xarray()
        data_ref = data_ref.rename({"index": "z"})
        data_ref["depth"].data = up.Length(data_ref.depth.to_numpy(), "nm").um
        data_ref = data_ref.set_coords("depth")

        self.sf = (data_raw["na+_conc"] / data_raw["na+_intensity"]).mean()

        meas_depth = self.params["Max X"]

        self.matrix_ds = raw_data
        self.data_2d = self._matrix_ds[["x", "y"]]

        peak_m = xr.where(
            (
                self.data_2d.grad
                < int(self.data_2d.grad.mean() + self.data_2d.grad.std() * self.devs)
            ),
            1,
            np.nan,
        )
        peak_m = peak_m * xr.where(
            (
                self.data_2d.flat
                < int(self.data_2d.flat.mean() + self.data_2d.flat.std() * self.devs)
            ),
            1,
            np.nan,
        )

        # 1d datasets
        z_eval = self.matrix_ds[["z"]].to_dataframe()
        z_eval["given_intens"] = data_raw["na+_intensity"].to_numpy()
        z_eval["sum_intens"] = self.matrix_ds.intens.sum(["x", "y"])
        z_eval["masked_intens"] = (self.matrix_ds.intens * peak_m).sum(["x", "y"])
        z_eval["delta"] = z_eval["given_intens"] - z_eval["sum_intens"]
        z_eval["mult"] = z_eval["given_intens"] / z_eval["sum_intens"]
        z_eval["sum_chng"] = z_eval["sum_intens"].pct_change(1)
        z_eval["masked_chng"] = z_eval["sum_intens"].pct_change(1)
        z_eval["sum_means"] = z_eval["sum_intens"][::-1].expanding(1).mean()
        z_eval["masked_means"] = z_eval["masked_intens"][::-1].expanding(1).mean()
        self.z_eval = z_eval

        bkg = (
            z_eval.to_xarray()
            .sum_intens.where(z_eval.to_xarray().sum_chng > 0, drop=True)
            .isel(z=1)
        )

        bins = {}
        flt_msk = self.data_2d.surf * peak_m

        bins = pd.cut(np.unique(flt_msk), self.num_bins).unique().dropna()

        def func(val):
            return gen_groups(bins, flt_msk, flt_msk, val)

        data_masks = grow_arr(func, self.set_sigma, "sigma")
        data_masks = data_masks.transpose("x", "y", "sigma", "grp")

        perc = (data_masks).sum(["x", "y"]) / 128 ** 2

        data_3d = self.matrix_ds[["x", "y", "z"]]
        data_3d = self.matrix_ds.intens * data_masks
        data_3d = data_3d.to_dataset(name="raw_intens")
        data_3d["raw_conc"] = data_3d.raw_intens * self.sf / perc
        data_3d["corr_intens"] = data_3d.raw_intens * z_eval["mult"].to_xarray()
        data_3d["corr_conc"] = data_3d.corr_intens * self.sf / perc
        self.data_3d = data_3d

        self.data_masks = data_masks

        # quantifications and final sorting
        res_dict = {}
        res_dict["bins"] = pd.DataFrame(bins)
        res_dict["perc"] = (
            perc.to_dataframe(name="percents").unstack(["sigma"]).droplevel(0, 1)
        )
        res_dict["proj_c"] = (
            self.data_1d.raw_conc.sel(z=slice(None, self.surf_lyr, None))
            .mean(["z"])
            .to_dataframe()
            .unstack("sigma")
            .droplevel(0, 1)
        )
        res_dict["proj_c_alt"] = (
            self.data_1d.corr_conc.sel(z=slice(None, self.surf_lyr, None))
            .mean(["z"])
            .to_dataframe()
            .unstack("sigma")
            .droplevel(0, 1)
        )
        self.res_dict = {
            **res_dict,
            **{
                f"{key}_s{sig}": self.data_1d[key]
                .sel(sigma=sig)
                .to_dataframe()
                .unstack(["grp"])
                .droplevel(0, 1)
                for key in self.data_1d.data_vars
                for sig in self.data_1d.sigma.to_numpy()
            },
        }

        if save:
            self.save()
        self.plots(**kwargs)

    @property
    def matrix_ds(self):
        """Calculate constant, may shift to use the depth range instead."""
        return self._matrix_ds

    @matrix_ds.setter
    def matrix_ds(self, value):
        if isinstance(value, pd.DataFrame):
            value = value.to_xarray()
        surf_len = up.Length(self.params["Raster len"], self.params["Raster unit"]).um
        value = value.assign_coords({"length": (value.x * surf_len / value.x.max())})
        value = value.assign_coords({"width": (value.y * surf_len / value.y.max())})
        value = value.assign_coords(
            {"depth": (value.z * self.params["Max X"] / value.z.max())}
        )
        self._matrix_ds = value.set_index({"x": "length", "y": "width", "z": "depth"})

    @property
    def data_2d(self):
        """Calculate constant, may shift to use the depth range instead."""
        return self._data_2d

    @data_2d.setter
    def data_2d(self, value):
        self._data_2d = value
        self._data_2d["flat"] = self._matrix_ds.intens.sum(["z"])
        self._data_2d["surf"] = self._matrix_ds.intens.sel(
            z=slice(None, self.surf_lyr, None)
        ).sum(["z"])
        self._data_2d["back"] = self._matrix_ds.intens.sel(
            z=slice(self.back_lyr, None, None)
        ).sum(["z"])
        self._data_2d["sliced"] = self._matrix_ds.intens.sel(
            z=slice(self.slc_strt, self.slc_stp)
        ).mean(["z"])
        self._data_2d["grad"] = (
            ["x", "y"],
            gaussian_gradient_magnitude(self.matrix_ds.intens.sum(["z"]), 1),
        )

    @property
    def data_1d(self):
        """Calculate constant, may shift to use the depth range instead."""
        if hasattr(self, "data_3d"):
            return self.data_3d.sum(["x", "y"])
        else:
            return xr.Dataset()

    def save(self):
        gf.save(self.res_dict, self.outpath, f"{self.sample}_info")

        maps = {
            f"{key}_s{sig}": self.data_3d[key]
            .sel(sigma=sig)
            .sum(["z", "grp"])
            .to_dataframe()
            .unstack("x")
            .droplevel(0, 1)
            for key in self.data_3d.data_vars
            for sig in self.data_3d.sigma.to_numpy()
        }
        maps["full1"] = (
            (self.matrix_ds * self.sf).sum(["z"]).intens.to_dataframe().unstack("x")
        )
        maps["full2"] = (
            (self.matrix_ds * self.sf * self.z_eval["mult"].to_xarray())
            .sum(["z"])
            .intens.to_dataframe()
            .unstack("x")
        )
        maps["mean1"] = (
            (self.matrix_ds * self.sf).mean(["z"]).intens.to_dataframe().unstack("x")
        )
        maps["mean2"] = (
            (self.matrix_ds * self.sf * self.z_eval["mult"].to_xarray())
            .mean(["z"])
            .intens.to_dataframe()
            .unstack("x")
        )
        maps["surf1"] = (
            (self.matrix_ds * self.sf)
            .sel(z=slice(None, self.surf_lyr, None))
            .sum(["z"])
            .intens.to_dataframe()
            .unstack("x")
        )
        maps["surf2"] = (
            (self.matrix_ds * self.sf * self.z_eval["mult"].to_xarray())
            .sel(z=slice(None, self.surf_lyr, None))
            .sum(["z"])
            .intens.to_dataframe()
            .unstack("x")
        )

        gf.save(maps, self.outpath, f"{self.sample}_maps")

        maps1 = {
            f"s{sig}_g{group}": self.data_3d.raw_conc.sel(grp=group, sigma=sig)
            .sum(["z"])
            .to_dataframe()
            .unstack("x")
            .droplevel(0, 1)
            for sig in self.data_3d.sigma.to_numpy()
            for group in self.data_3d.grp.to_numpy()
        }

        maps2 = {
            f"s{sig}_g{group}": self.data_3d.corr_conc.sel(grp=group, sigma=sig)
            .sum(["z"])
            .to_dataframe()
            .unstack("x")
            .droplevel(0, 1)
            for sig in self.data_3d.sigma.to_numpy()
            for group in self.data_3d.grp.to_numpy()
        }

        gf.save(maps1, self.outpath, f"{self.sample}_mapsv1")
        gf.save(maps2, self.outpath, f"{self.sample}_mapsv2")

    def plots(
        self, grp_plot=None, map_plot=None, plt_type=None, surf_plt=None, prof_plt=None
    ):
        if grp_plot is not None:
            self.data_masks.isel(sigma=grp_plot).plot(
                x="x", y="y", col="grp", col_wrap=int(self.num_bins / 3)
            )
        if map_plot is not None:
            if plt_type is None:
                plt_type = "raw_intens"
            if "conc" in plt_type:
                (
                    np.log10(
                        self.data_3d[plt_type]
                        .sel(z=slice(None, self.surf_lyr, None), sigma=map_plot)
                        .sum(["z"])
                    ).plot(x="x", y="y", col="grp", col_wrap=int(self.num_bins / 3))
                )
            else:
                (
                    self.data_3d[plt_type]
                    .sel(z=slice(None, self.surf_lyr, None), sigma=map_plot)
                    .sum(["z"])
                    .plot(x="x", y="y", col="grp", col_wrap=int(self.num_bins / 3))
                )
        if surf_plt is not None:
            fig, ax = plt.subplots()
            self.data_2d[surf_plt].plot.contourf(x="x", y="y")
            plt.tight_layout()
        if prof_plt is not None:
            if plt_type is None:
                plt_type = "raw_intens"
            fig, ax = plt.subplots()
            self.data_1d.isel(sigma=prof_plt)[plt_type].plot.line(x="z")
            ax.set_xlim(0, 4)
            ax.set_ylim(1e16, 1e19)
            ax.set_yscale("log")
            plt.tight_layout()


class BulkImport:
    def __init__(self, call, **kwargs):
        prime_path = "C:\\Users\\j2cle\\Work Docs\\Data\\Analysis\\SIMS\\"
        active_log = (
            pd.read_excel(f"{prime_path}Active Log.xlsx", index_col=0, header=0)
            .dropna(axis=0, how="all")
            .fillna("")
        )
        called = (
            active_log[active_log.iloc[:, :5].isin(call)]
            .dropna(axis=1, how="all")
            .dropna(axis=0, how="any")
        )
        self.df_log = active_log.loc[called.index, :]

        self.not_called = [x for x in call if x not in self.df_log.to_numpy()]

        folders = pd.Series(
            [f"{x[0]}\\{x[1]}" for x in self.df_log[["Source", "Folder"]].to_numpy()],
            index=self.df_log.index,
        )
        files = pd.Series(
            [
                f"{prime_path}{x[0]}\\{x[1]}\\Files{x[2]}\\{x[3]}"
                for x in self.df_log[
                    ["Source", "Folder", "Sub Folder", "File"]
                ].to_numpy()
            ],
            index=self.df_log.index,
        )

        for logs in folders.unique():
            self.params_df = pd.read_excel(
                f"{prime_path}{logs}\Sample Log.xlsx", index_col=0, skiprows=1
            ).dropna(axis=0, how="all")
        self.raws = {
            sample: ImportFunc(
                files[sample],
                self.df_log.loc[sample, "Tab"],
                self.df_log.loc[sample, "Columns"],
                func=self.df_log["Import Info"][sample],
            ).data_raw
            for sample in self.df_log.index
        }

        self.datas = {}
        for sample in self.df_log.index:
            data = ConvFunc(
                self.raws[sample],
                self.params[sample],
                func=self.df_log["Import Info"][sample],
                **kwargs,
            ).data
            if isinstance(data, dict):
                self.datas = {**self.datas, **data}
            else:
                self.datas[sample] = data

    @property
    def params_df(self):
        """Calculate constant, may shift to use the depth range instead."""
        return self._params_df

    @params_df.setter
    def params_df(self, value):
        ind = value.index.join(self.df_log.index, how="inner")
        value = value.loc[ind, :]
        if hasattr(self, "_params_df"):
            self._params_df = pd.concat([self._params_df, value], join="outer")
        else:
            self._params_df = value

    @property
    def params(self):
        """Calculate constant, may shift to use the depth range instead."""
        return self._params_df.T.to_dict("series")


class DataProfile:
    """Return sum of squared errors (pred vs actual)."""

    def __init__(self, slog, data, limit=False, loc=None, even=False, **kwargs):
        self.params = slog

        self.data = data

        if "Layer (actual)" in self.params.keys() and not np.isnan(
            self.params["Layer (actual)"]
        ):
            self.a_layer_cm = gf.tocm(
                self.params["Layer (actual)"], self.params["A-Layer unit"]
            )
        else:
            self.a_layer_cm = 0
        if "Fit depth/limit" in self.params.keys() and not np.isnan(
            self.params["Fit depth/limit"]
        ):
            self.fit_depth_cm = gf.tocm(
                self.params["Fit depth/limit"], self.params["Fit Dep unit"]
            )
        else:
            self.params["Fit depth/limit"] = lin_test(
                self.data["Depth"].to_numpy(), self.data["Na"].to_numpy(), 0.05
            )[1]
            self.params["Fit Dep unit"] = "cm"
        if "Layer (profile)" in self.params.keys() and not np.isnan(
            self.params["Layer (profile)"]
        ):
            self.p_layer_cm = gf.tocm(
                self.params["Layer (profile)"], self.params["P-Layer unit"]
            )
        self.data_bgd = pd.Series()

        self.limit_test()

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

    def limit_test(self, thresh=0.025):
        """Return sum of squared errors (pred vs actual)."""
        lin_loc, lin_lim = lin_test(
            self.data["Depth"].to_numpy(), self.data["Na"].to_numpy(), thresh
        )

        if lin_lim > self.fit_depth_cm * 1.1 or lin_lim < self.fit_depth_cm * 0.9:
            self.data_bgd["bgd_lim"] = gf.find_nearest(
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
                    **kwargs,
                )[:2]
                resid = self.data["Na"].to_numpy()[x:stop] - linear(
                    self.data["Depth"].to_numpy()[x:stop], coeff
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
            **kwargs,
        )[:2]
        self.data_bgd["P-value"] = self.p[self.data_bgd["bgd_ave"]]
        self.data_bgd["slope"] = coeff[0]
        self.data_bgd["intercept"] = coeff[1]

    @property
    def thick_cm(self):
        """Return sum of squared errors (pred vs actual)."""
        return gf.tocm(self.params["Thick"], self.params["Thick unit"])


test = BulkImport(["R-90", "Temp", "T-Pixel"])
obj = {}
for key, val in test.datas.items():
    if "conc" in key:
        params = [x for x in test.params.keys() if "Na+" in x][0]
    else:
        params = key
    obj[key] = DataProfile(test.params[params], val)
