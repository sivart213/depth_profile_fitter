# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:24:26 2022

@author: j2cle
"""
# %% import section
import os
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import utilities as ut
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
from dataclasses import InitVar, make_dataclass

warnings.simplefilter("ignore", np.RankWarning)
warnings.filterwarnings("ignore")


def make_dc(data, params, name="Profile"):
    list_in = [(ut.nameify(k), type(v), v) for k, v in params.to_dict().items()]
    list_in.append(("data", InitVar[pd.DataFrame], data))
    new_dc = make_dataclass(name, list_in)
    return new_dc


def lin_test(x, y, lim=0.025):
    """Return sum of squared errors (pred vs actual)."""
    line_info = np.array([np.polyfit(x[-n:], y[-n:], 1) for n in range(1, len(x))])

    delta = np.diff(line_info[int(0.1 * len(x)):, 0])
    delta = delta / max(abs(delta))
    bounds = np.where(delta < -lim)[0]
    if bounds[0] + len(bounds) - 1 == bounds[-1]:
        bound = len(x) - bounds[0]
    else:
        bound = (
            len(x)
            - [bounds[n] for n in range(1, len(bounds)) if bounds[n - 1] + 1 != bounds[n]][-1]
        )
    return bound, x[bound]


def depth_conv(data_in, unit, layer_act, layer_meas):
    """Return sum of squared errors (pred vs actual)."""
    data_out = data_in
    if unit != "s":
        if not pd.isnull(layer_meas):
            data_out[data_in < layer_meas] = data_in[data_in < layer_meas] * layer_act / layer_meas
            data_out[data_in >= layer_meas] = (
                (data_in[data_in >= layer_meas] - layer_meas)
                * ((max(data_in) - layer_act) / (max(data_in) - layer_meas))
            ) + layer_act
        if unit != "cm":
            data_out = ut.Length(data_in, unit).cm
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
    ref_temp_arr = xr.DataArray(gaussian_filter(ref_arr, sigma=blur), coords=ref_arr.coords)
    while np.isnan(ref_temp_arr).all() and blur > 0:
        blur -= 0.5
        ref_temp_arr = xr.DataArray(gaussian_filter(ref_arr, sigma=blur), coords=ref_arr.coords)
    if not np.isnan(ref_temp_arr).all():
        ref_arr = ref_temp_arr

    def func(loc):
        return xr.where((bins[loc].left < ref_arr) & (ref_arr <= bins[loc].right), 1, np.nan)

    tmp_arr = grow_arr(func, len(bins), "grp")

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
        self.data_raw = pd.read_excel(args[0], sheet_name=args[1], usecols=args[2]).dropna()
        return self.data_raw

    def asu_raw(self, *args):
        header_in = pd.read_csv(args[0], delimiter="\t", header=None, skiprows=14, nrows=2).dropna(
            axis=1, how="all"
        )
        header_temp = (
            header_in.iloc[0, :].dropna().to_list() + header_in.iloc[0, :].dropna().to_list()
        )
        header_in.iloc[0, : len(header_temp)] = sorted(
            header_temp, key=lambda y: header_temp.index(y)
        )
        header_in = header_in.dropna(axis=1)
        headers = [
            header_in.iloc[0, x] + " " + header_in.iloc[1, x] for x in range(header_in.shape[1])
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
        header_in = pd.read_csv(args[0], delimiter="\t", header=None, skiprows=2, nrows=3).dropna(
            axis=1, how="all"
        )
        header_in = header_in.fillna(method="ffill", axis=1)
        headers = [
            header_in.iloc[0, x] + " " + header_in.iloc[2, x] for x in range(header_in.shape[1])
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
        return df[df.columns[[(a in x.lower()) and (b in x.lower()) for x in df.columns]]].to_numpy(
            copy=True
        )

    def nrel_d(self, raw):
        self.data["Depth"] = depth_conv(
            self.gen_col(raw, "na", "x"),
            self.params["X unit"],
            self.params["Layer (actual)"],
            self.params["Layer (profile)"],
        )
        self.data["Na"] = (
            self.gen_col(raw, "na", "y") / self.gen_col(raw, "12c", "y").mean() * self.params["RSF"]
        )
        return self.data

    def asu_raw(self, raw):
        rate = self.params["Max X"] / self.gen_col(raw, "na", "time").max()
        self.data["Depth"] = ut.Length(
            self.gen_col(raw, "na", "time") * rate, self.params["X unit"]
        ).cm
        self.data["Na"] = (
            self.gen_col(raw, "na", "c/s")
            / self.gen_col(raw, "12c", "c/s").mean()
            * self.params["RSF"]
        )
        return self.data

    def rice_semi_treated(self, raw):
        self.data["Depth"] = ut.Length(self.gen_col(raw, "depth", "dep"), self.params["X unit"]).cm

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
        self.data["Depth"] = ut.Length(self.gen_col(raw, "depth", "dep"), self.params["X unit"]).cm
        self.data["Na"] = self.gen_col(raw, "na+", "conc")
        return self.data

    def rice_raw(self, raw, **kwargs):
        res_dict = PixelConv(raw, self.params, **kwargs).res_dict
        samp = self.params["Sample"]
        self.data = {}
        for key, val in res_dict.items():
            if "conc" in key:
                key_str = re.sub("conc_", "", key)
                val = val.loc[:, (val != 0).any(axis=0)]
                val = val.set_index(ut.Length(val.index.to_numpy(), "um").cm).reset_index()
                res_alt = {
                    f"{samp}-{key_str}-{col}": val[["index", col]].rename(
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
        corr=False,
        path=None,
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
        self.corr = corr
        self.save_res = save

        self.params = params
        self.sample = self.params["Sample"]

        if path is not None:
            self.outpath = path
        else:
            self.outpath = ut.pathify(
                "work", "Data", "Analysis", "SIMS", "RICE", "220425 Man", "Results"
            )
        self.matrix_ds = raw_data

        data_ref_raw = ImportFunc(
            self.params["Reference File"], func="rice_treated"
        ).data_raw.dropna(axis=1)

        data_ref = data_ref_raw.to_xarray()
        data_ref = data_ref.rename({"index": "z"})
        data_ref["depth"].data = ut.Length(data_ref.depth.to_numpy(), "nm").um
        data_ref = data_ref.set_coords("depth")
        data_ref = data_ref.set_index({"z": "depth"})
        self.data_ref = data_ref.interp(z=list(self.matrix_ds.sum(["x", "y"]).z.to_numpy()))

        ref_conc_list = [key for key in self.data_ref.variables.mapping.keys() if "conc" in key]

        perf_match = None
        ref_match = None
        near_match = None
        no_match = []
        if len(ref_conc_list) >= 1:
            for n in ref_conc_list:
                if params.Ion.lower() in n.lower():
                    perf_match = n
                elif params.Ref_Ion.lower() in n.lower():
                    ref_match = n
                elif params.Ion.lower()[:2] in n.lower():
                    near_match = n
                else:
                    no_match.append(n)

        if perf_match is not None:
            self.data_ref_conc = perf_match
        elif ref_match is not None:
            self.data_ref_conc = ref_match
        elif near_match is not None:
            self.data_ref_conc = near_match
        else:
            self.data_ref_conc = None

        self.data_ref_intens = [
            key
            for key in self.data_ref.variables.mapping.keys()
            if params.Ion.lower() in key and "intens" in key
        ][0]

        if not np.isnan(params["SF"]):
            self.sf = params["SF"]
        elif self.data_ref_conc is not None:
            self.sf = (
                self.data_ref[self.data_ref_conc] / self.data_ref[self.data_ref_intens]
            ).mean()
        else:
            self.sf = 5e14

        self.calculate()
        if save:
            self.save()

    def calculate(self, **kwargs):
        self.__dict__.update(kwargs)
        # this should be post updatable
        if not np.isnan(self.devs):
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
        else:
            peak_m = arr_like(self.data_2d, 1)

        # 1d datasets: has peak_m so should be param
        z_eval = self.matrix_ds[["z"]].to_dataframe()
        z_eval["given_intens"] = self.data_ref[self.data_ref_intens].to_numpy()
        z_eval["sum_intens"] = self.matrix_ds.intens.sum(["x", "y"])
        z_eval["masked_intens"] = (self.matrix_ds.intens * peak_m).sum(["x", "y"])
        z_eval["delta"] = z_eval["given_intens"] - z_eval["sum_intens"]
        z_eval["mult"] = z_eval["given_intens"] / z_eval["sum_intens"]
        z_eval["sum_chng"] = z_eval["sum_intens"].pct_change(1)
        z_eval["masked_chng"] = z_eval["sum_intens"].pct_change(1)
        z_eval["sum_means"] = z_eval["sum_intens"][::-1].expanding(1).mean()
        z_eval["masked_means"] = z_eval["masked_intens"][::-1].expanding(1).mean()
        self.z_eval = z_eval

        bins = {}
        flt_msk = self.data_2d.surf * peak_m
        try:
            bins = pd.cut(np.unique(flt_msk), self.num_bins).unique().dropna()
        except ValueError:
            peak_m = arr_like(self.data_2d, 1)
            flt_msk = self.data_2d.surf
            bins = pd.cut(np.unique(self.data_2d.surf), self.num_bins).unique().dropna()

        def func(val):
            return gen_groups(bins, flt_msk, flt_msk, val)

        data_masks = grow_arr(func, self.set_sigma, "sigma")
        data_masks = data_masks.transpose("x", "y", "sigma", "grp")

        perc = (data_masks).sum(["x", "y"]) / 128**2

        data_3d = self.matrix_ds[["x", "y", "z"]]
        data_3d = self.matrix_ds.intens * data_masks
        data_3d = data_3d.to_dataset(name="raw_intens")
        data_3d["raw_conc"] = data_3d.raw_intens * self.sf / perc
        data_3d["corr_intens"] = data_3d.raw_intens * z_eval["mult"].to_xarray()
        if self.corr:
            data_3d["corr_conc"] = data_3d.corr_intens * self.sf / perc
        self.data_3d = data_3d

        self.data_masks = data_masks

        # quantifications and final sorting
        res_dict = {}
        res_dict["bins"] = pd.DataFrame(bins)
        res_dict["perc"] = perc.to_dataframe(name="percents").unstack(["sigma"]).droplevel(0, 1)
        res_dict["proj_c"] = (
            self.data_1d.raw_conc.sel(z=slice(None, self.surf_lyr, None))
            .mean(["z"])
            .to_dataframe()
            .unstack("sigma")
            .droplevel(0, 1)
        )
        if self.corr:
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
                f"{key}_{self.params.Ion.lower()}_s{sig}": self.data_1d[key]
                .sel(sigma=sig)
                .to_dataframe()
                .unstack(["grp"])
                .droplevel(0, 1)
                for key in self.data_1d.data_vars
                for sig in self.data_1d.sigma.to_numpy()
            },
        }

    @property
    def matrix_ds(self):
        """Calculate constant, may shift to use the depth range instead."""
        return self._matrix_ds

    @matrix_ds.setter
    def matrix_ds(self, value):
        if isinstance(value, pd.DataFrame):
            value = value.to_xarray()
        surf_len = ut.Length(self.params["Raster len"], self.params["Raster unit"]).um
        value = value.assign_coords({"length": (value.x * surf_len / value.x.max())})
        value = value.assign_coords({"width": (value.y * surf_len / value.y.max())})
        value = value.assign_coords({"depth": (value.z * self.params["Max X"] / value.z.max())})
        self._matrix_ds = value.set_index({"x": "length", "y": "width", "z": "depth"})

    @property
    def data_2d(self):
        """Calculate constant, may shift to use the depth range instead."""
        _data_2d = self._matrix_ds[["x", "y"]]
        _data_2d["flat"] = self._matrix_ds.intens.sum(["z"])
        _data_2d["surf"] = self._matrix_ds.intens.sel(z=slice(None, self.surf_lyr, None)).sum(["z"])
        _data_2d["back"] = self._matrix_ds.intens.sel(z=slice(self.back_lyr, None, None)).sum(["z"])
        _data_2d["sliced"] = self._matrix_ds.intens.sel(z=slice(self.slc_strt, self.slc_stp)).mean(
            ["z"]
        )
        _data_2d["grad"] = (
            ["x", "y"],
            gaussian_gradient_magnitude(self.matrix_ds.intens.sum(["z"]), 1),
        )
        return _data_2d

    @property
    def data_1d(self):
        """Calculate constant, may shift to use the depth range instead."""
        if hasattr(self, "data_3d"):
            return self.data_3d.sum(["x", "y"])
        else:
            return xr.Dataset()

    def save(self):
        ut.save(self.res_dict, self.outpath, f"{self.sample}_{self.params.Ion}_info")

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
        maps["full1"] = (self.matrix_ds * self.sf).sum(["z"]).intens.to_dataframe().unstack("x")
        maps["full2"] = (
            (self.matrix_ds * self.sf * self.z_eval["mult"].to_xarray())
            .sum(["z"])
            .intens.to_dataframe()
            .unstack("x")
        )
        maps["mean1"] = (self.matrix_ds * self.sf).mean(["z"]).intens.to_dataframe().unstack("x")
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

        ut.save(maps, self.outpath, f"{self.sample}_{self.params.Ion}_maps")

        maps1 = {
            f"s{sig}_g{group}": self.data_3d.raw_conc.sel(grp=group, sigma=sig)
            .sum(["z"])
            .to_dataframe()
            .unstack("x")
            .droplevel(0, 1)
            for sig in self.data_3d.sigma.to_numpy()
            for group in self.data_3d.grp.to_numpy()
        }
        ut.save(maps1, self.outpath, f"{self.sample}_{self.params.Ion}_mapsv1")

        if self.corr:
            maps2 = {
                f"s{sig}_g{group}": self.data_3d.corr_conc.sel(grp=group, sigma=sig)
                .sum(["z"])
                .to_dataframe()
                .unstack("x")
                .droplevel(0, 1)
                for sig in self.data_3d.sigma.to_numpy()
                for group in self.data_3d.grp.to_numpy()
            }

            ut.save(maps2, self.outpath, f"{self.sample}_{self.params.Ion}_mapsv2")

    def plots(self, grp_plot=None, map_plot=None, plt_type=None, surf_plt=None, prof_plt=None):
        if grp_plot is not None:
            self.data_masks.isel(sigma=grp_plot).plot(
                x="x", y="y", col="grp", col_wrap=round(self.num_bins / 3)
            )
            if self.save_res:
                plt.savefig(os.sep.join((self.outpath, f"{self.sample}_{self.params.Ion}_grp.png")))
                plt.close()
        if map_plot is not None:
            if plt_type is None:
                plt_type = "raw_intens"
            if "conc" in plt_type:
                (
                    np.log10(
                        self.data_3d[plt_type]
                        .sel(z=slice(None, self.surf_lyr, None), sigma=map_plot)
                        .sum(["z"])
                    ).plot(x="x", y="y", col="grp", col_wrap=round(self.num_bins / 3))
                )
            else:
                (
                    self.data_3d[plt_type]
                    .sel(z=slice(None, self.surf_lyr, None), sigma=map_plot)
                    .sum(["z"])
                    .plot(x="x", y="y", col="grp", col_wrap=round(self.num_bins / 3))
                )
            if self.save_res:
                plt.savefig(os.sep.join((self.outpath, f"{self.sample}_{self.params.Ion}_map.png")))
                plt.close()

        if surf_plt is not None:
            fig, ax = plt.subplots()
            self.data_2d[surf_plt].plot.contourf(x="x", y="y")
            plt.tight_layout()
            if self.save_res:
                plt.savefig(
                    os.sep.join((self.outpath, f"{self.sample}_{self.params.Ion}_surf.png"))
                )
                plt.close()

        if prof_plt is not None:
            if plt_type is None:
                plt_type = "raw_intens"
            fig, ax = plt.subplots()
            self.data_1d.isel(sigma=prof_plt)[plt_type].plot.line(x="z")
            ax.set_xlim(0, 4)
            ax.set_ylim(1e16, 1e19)
            ax.set_yscale("log")
            plt.tight_layout()
            if self.save_res:
                plt.savefig(
                    os.sep.join((self.outpath, f"{self.sample}_{self.params.Ion}_prof.png"))
                )
                plt.close()


class BulkImport:
    def __init__(self, call, calc=False, folder="Bulk_import", **kwargs):
        self.calc = calc
        self.jar = ut.PickleJar(folder=folder, **kwargs)

        prime_path = ut.pathify("work", "Data", "Analysis", "SIMS")
        active_log = (
            pd.read_excel(os.sep.join((prime_path, "Active Log.xlsx")), index_col=0, header=0)
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
            [os.sep.join(x) for x in self.df_log[["Source", "Folder"]].to_numpy()],
            index=self.df_log.index,
        )
        files = pd.Series(
            [
                os.sep.join((prime_path, *x[:2], "Files", *x[2:]))
                for x in self.df_log[["Source", "Folder", "Sub Folder", "File"]].to_numpy()
            ],
            index=self.df_log.index,
        )

        for logs in folders.unique():
            self.params_df = pd.read_excel(
                os.sep.join((prime_path, logs, "Sample Log.xlsx")), index_col=0, skiprows=1
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

    def to_data(self, **kwargs):
        db = self.jar.database
        for sample in self.df_log.index:
            calc = self.calc
            if len(db) == 0:
                calc = True
            if not calc:
                if self.df_log["Import Info"][sample] == "rice_raw":
                    samp = self.df_log.Sample[sample]
                    meas = self.df_log.Measurement[sample]
                    raw = ut.slugify(f"{samp}-raw_{meas}")
                    corr = ut.slugify(f"{samp}-corr_{meas}")
                    datas = db[db.str.contains(f"{raw}|{corr}")]
                else:
                    datas = db[db.str.startswith(f"{sample}")]
                if len(datas) == 0:
                    calc = True
                else:
                    self.datas = datas
            if calc:
                data = ConvFunc(
                    self.raws[sample],
                    self.params[sample],
                    func=self.df_log["Import Info"][sample],
                    **kwargs,
                ).data
                if isinstance(data, dict):
                    self.jar.append(data)
                    self.datas = list(data.keys())
                else:
                    self.jar[sample] = data
                    self.datas = data
        return

    def to_obj(self, **kwargs):
        db = self.jar.database
        for sample in self.df_log.index:
            calc = self.calc
            if len(db) == 0:
                calc = True
            if not calc and self.df_log["Import Info"][sample] == "rice_raw":
                objs = db[db.str.contains(f"{ut.slugify(sample)}_obj")]
                if len(objs) == 0:
                    calc = True
                else:
                    self.objs = objs
            if calc and self.df_log["Import Info"][sample] == "rice_raw":
                obj = PixelConv(self.raws[sample], self.params[sample], **kwargs)
                self.jar[f"{sample}_obj"] = obj
                self.objs = f"{ut.slugify(sample)}_obj"
        return

    @property
    def datas(self):
        """Calculate constant, may shift to use the depth range instead."""
        if not hasattr(self, "_datas"):
            self._datas = pd.Series()
        return self._datas

    @datas.setter
    def datas(self, value):
        """Calculate constant, may shift to use the depth range instead."""
        self._datas = pd.Series(
            pd.concat([self.datas, pd.Series(value)], ignore_index=True).unique()
        )

    @property
    def datas_dict(self):
        """Calculate constant, may shift to use the depth range instead."""
        return self.jar.to_dict(self.datas)

    @property
    def objs(self):
        """Calculate constant, may shift to use the depth range instead."""
        if not hasattr(self, "_objs"):
            self._objs = pd.Series()
        return self._objs

    @objs.setter
    def objs(self, value):
        """Calculate constant, may shift to use the depth range instead."""
        self._objs = pd.Series(pd.concat([self.objs, pd.Series(value)], ignore_index=True).unique())

    @property
    def objs_dict(self):
        """Calculate constant, may shift to use the depth range instead."""
        return self.jar.to_dict(self.objs)

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

        if "Layer (actual)" in self.params.keys() and not np.isnan(self.params["Layer (actual)"]):
            self.a_layer_cm = ut.Length(
                self.params["Layer (actual)"], self.params["A-Layer unit"]
            ).cm
        else:
            self.a_layer_cm = 0
        if "Fit depth/limit" in self.params.keys() and not np.isnan(self.params["Fit depth/limit"]):
            self.fit_depth_cm = ut.Length(
                self.params["Fit depth/limit"], self.params["Fit Dep unit"]
            ).cm
        else:
            self.params["Fit depth/limit"] = lin_test(
                self.data["Depth"].to_numpy(), self.data["Na"].to_numpy(), 0.05
            )[1]
            self.params["Fit Dep unit"] = "cm"
        if "Layer (profile)" in self.params.keys() and not np.isnan(self.params["Layer (profile)"]):
            self.p_layer_cm = ut.Length(
                self.params["Layer (profile)"], self.params["P-Layer unit"]
            ).cm
        self.data_bgd = pd.Series()

        self.limit_test()

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
            self.data_bgd["bgd_lim"] = ut.find_nearest(
                self.data["Depth"].to_numpy(), self.fit_depth_cm
            )
        else:
            self.data_bgd["bgd_lim"] = lin_loc

    @property
    def thick_cm(self):
        """Return sum of squared errors (pred vs actual)."""
        return ut.Length(self.params["Thick"], self.params["Thick unit"]).cm
