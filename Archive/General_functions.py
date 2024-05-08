# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import scipy.special as scs
import numpy as np
import pandas as pd
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import unicodedata
import re
import os
import dill
import difflib
from scipy.optimize import curve_fit
from sklearn import metrics
from scipy import stats
from matplotlib import ticker
from datetime import datetime as dt
import inspect
import warnings
warnings.simplefilter("ignore", np.RankWarning)
warnings.filterwarnings("ignore")
style.use('seaborn-colorblind')


KB_EV = 8.617333262145e-5  # ev/k
KB_J = 1.380649e-23  # J/k
N_MOL = 6.02214076e23  # 1/mol
H_J = 6.62607015e-34  # Js
H_EV = 4.135667662e-15  # eVs
HBAR_J = 1.054571800e-34  # js
HBAR_EV = 6.582119514e-16  # eVs
HC_JM = 1.98644568e-25  # jm
HC_EVUM = 1.23984193  # eVum
HBARC_JM = 3.16152649e-26  # jm
HBARC_EVUM = 0.19732697  # eVum
C = 299792458  # m/s
QE = 1.60217662e-19  # Coulombs
M0 = 9.10938356e-31  # kG
PERM = 8.8541878128e-12  # F/m

csv = pd.read_csv('C:\\Users\\j2cle\\Work Docs\\Work Documents\\kindlmann-tables\\kindlmann-table-float-1024.csv')
col_arr = (csv.iloc[:, 1:].to_numpy())
mpl.cm.register_cmap('kindlmann', mpl.colors.LinearSegmentedColormap.from_list('kindlmann', col_arr))
mpl.cm.register_cmap('kindlmann_r',mpl.colors.LinearSegmentedColormap.from_list('kindlmann', col_arr).reversed())

# %% Equations
def eqn_sets(params, **kwargs):

    results = []
    for key, vals in params.items():
        if isinstance(vals, (np.ndarray, list, tuple)):
            tmp = [eqn({**params, **{key: val}}, **kwargs) for val in vals]
            results.append(tmp)
    if results == []:
        results = eqn(params, **kwargs)
    return results


def eqn(params, target="D", eqns="x-1", as_set=True):

    x = sym.Symbol("x", positive=True)
    params[target] = x

    if not isinstance(eqns, str):
        expr = sym.parsing.sympy_parser.parse_expr(eqns.pop(0)[1])
        expr = expr.subs(eqns)
    else:
        expr = sym.parsing.sympy_parser.parse_expr(eqns)
    expr = expr.subs(params)

    try:
        res = sym.solveset(expr, x, domain=sym.S.Reals)
        if isinstance(res, sym.sets.sets.EmptySet):
            res = sym.FiniteSet(*sym.solve(expr))
    except Exception:
        res = sym.FiniteSet(*sym.solve(expr))
    if not as_set:
        res = float(list(res)[0])
    return res



# def Diffusivity(D0, Ea, T, Ea_unit="eV", T_unit="C"):
#     """ Calculate. generic discription """
#     if Ea_unit == "J":
#         KB = KB_J
#     else:
#         KB = KB_EV
#     if T_unit == "C":
#         T = CtoK(T)
#     return D0 * np.exp(-Ea / (KB * T))


def Diffuse_t(D, x, Ns=1e19, Nf=1e12):
    """ Calculate. generic discription """
    return ((x / (2 * scs.erfinv(1 - Nf / Ns))) ** 2) / D


def Diffuse_D(t, x, Ns=1e19, Nf=1e12):
    """ Calculate. generic discription """
    return ((x / (2 * scs.erfinv(1 - Nf / Ns))) ** 2) / t

# def np_diffusion(pre_fac=None, actv_en=None, temp=None, result=None):
#     """ Calculate. generic discription """
#     if result is None:
#         return pre_fac * np.exp(-1 * actv_en / (KB_EV * temp))
#     else:
#         params = {'temp':temp, "pre_fac":pre_fac, 'actv_en':actv_en, 'result':result, 'boltz':KB_EV}
#         target = [key for key, val in params.items() if val is None]
#         if len(target) > 1:
#             return
#         eqn = "pre_fac*exp(-actv_en/(boltz*temp))"
#         return eqn_sets(params, target=target[0], eqns=eqn, as_set=False)

# def arrh(T, pre_fac, E_A):
#     """ Calculate. generic discription """
#     return pre_fac * np.exp(-E_A / (KB_EV * T))

def arrh(pre_fac=None, actv_en=None, temp=None, result=None):
    """ Calculate. generic discription """
    if result is None:
        return pre_fac * np.exp(-1 * actv_en / (KB_EV * temp))
    else:
        params = {'temp':temp, "pre_fac":pre_fac, 'actv_en':actv_en, 'result':result, 'boltz':KB_EV}
        target = [key for key, val in params.items() if val is None]
        if len(target) > 1:
            return
        eqn = "pre_fac*exp(-actv_en/(boltz*temp))"
        return eqn_sets(params, target=target[0], eqns=eqn, as_set=False)


def capacitance(rel_perm=None, area=None, thick=None, perm_const=None, cap=None):
    """ Calculate. generic discription """
    if perm_const is None:
        perm_const = PERM
    if cap is None:
        return rel_perm * perm_const * area / thick
    else:
        params = {'cap':cap, "perm":perm_const, 'rel_perm':rel_perm, 'thick':thick, 'area':area}
        target = [key for key, val in params.items() if val is None]
        if len(target) > 1:
            return
        eqn = "rel_perm*perm*area/thick-cap"
        return eqn_sets(params, target=target[0], eqns=eqn, as_set=False)


def resistance(rho=None, area=None, thick=None, res=None):
    """ Calculate. generic discription """
    if res is None:
        return rho * thick / area
    else:
        params = {'res':res, 'rho':rho, 'thick':thick, 'area':area}
        target = [key for key, val in params.items() if val is None]
        if len(target) > 1:
            return
        eqn = "rho*thick/area-res"
        return eqn_sets(params, target=target[0], eqns=eqn, as_set=False)

# def impedance(rho=None, area=None, thick=None, res=None):
#     """ Calculate. generic discription """
#     if res is None:
#         return rho * thick / area
#     else:
#         params = {'res':res, 'rho':rho, 'thick':thick, 'area':area}
#         target = [key for key, val in params.items() if val is None]
#         if len(target) > 1:
#             return
#         eqn = "rho*thick/area-res"
#         return eqn_sets(params, target=target[0], eqns=eqn, as_set=False)

# def admitance(rho=None, area=None, thick=None, res=None):
#     """ Calculate. generic discription """
#     if res is None:
#         return rho * thick / area
#     else:
#         params = {'res':res, 'rho':rho, 'thick':thick, 'area':area}
#         target = [key for key, val in params.items() if val is None]
#         if len(target) > 1:
#             return
#         eqn = "rho*thick/area-res"
#         return eqn_sets(params, target=target[0], eqns=eqn, as_set=False)

# def ohms(rho=None, area=None, thick=None, res=None):
#     """ Calculate. generic discription """
#     if res is None:
#         return rho * thick / area
#     else:
#         params = {'res':res, 'rho':rho, 'thick':thick, 'area':area}
#         target = [key for key, val in params.items() if val is None]
#         if len(target) > 1:
#             return
#         eqn = "rho*thick/area-res"
#         return eqn_sets(params, target=target[0], eqns=eqn, as_set=False)

# %% Converters

def StoD(t):
    """ Calculate. generic discription """
    return t * 1.15741e-5


def CtoK(T):
    """ Calculate. generic discription """
    return T + 273.15


def tocm(num, unit=0, sqr=0, inv=0):
    """ Calculate. generic discription """
    if unit == "um":
        cor = 1e-4
    elif unit == "mm":
        cor = 0.1
    elif unit == "nm":
        cor = 1e-7
    elif unit == "pm":
        cor = 1e-10
    elif unit == "A":
        cor = 1e-8
    elif unit == "m":
        cor = 100
    elif unit == "in":
        cor = 2.54
    else:
        cor = 1
    if sqr:
        cor = cor ** 2
    if inv:
        cor = 1 / cor
    cor_num = num * cor

    return cor_num


def fromcm(num, unit=0, sqr=0, inv=0):
    """ Calculate. generic discription """
    if unit == "um":
        cor = 1e4
    elif unit == "mm":
        cor = 10
    elif unit == "nm":
        cor = 1e7
    elif unit == "pm":
        cor = 1e10
    elif unit == "A":
        cor = 1e8
    elif unit == "m":
        cor = 1e-2
    elif unit == "in":
        cor = 1 / 2.54
    else:
        cor = 1
    if sqr:
        cor = cor ** 2
    if inv:
        cor = 1 / cor
    cor_num = num * cor

    return cor_num


def SciNote(num, exp=2):
    """ Calculate. generic discription """
    if exp == 2:
        num = "{:.2E}".format(num)
    elif exp == 3:
        num = "{:.3E}".format(num)
    elif exp == 4:
        num = "{:.4E}".format(num)
    elif exp == 5:
        num = "{:.5E}".format(num)
    else:
        print("Out of range")
        num = "{:.1E}".format(num)
    return num


def Sig_figs(number, digits=3):
    """ Calculate. generic discription """
    digits = digits - 1
    power = "{:e}".format(number).split("e")[1]
    return round(number, -(int(power) - digits))


def myround(x, base=5):
    """ Calculate. generic discription """
    return base * round(float(x) / base)


def closest(K, lst):
    """ Calculate. generic discription """
    # lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return lst[idx]


def myprint(filename, *info):
    """ Calculate. generic discription """
    print_file = open(filename, "a+")
    args = ""
    for arg in info:
        args = args + str(arg)
    print(args)
    print(args, file=print_file)
    print_file.close()
    return


def find_nearest(array, value, index=True):
    """ Calculate. generic discription """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if index:
        return idx
    else:
        return array[idx]

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def nameify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '_', value).strip('-_')


def save(data, path=None, name=None):
    if path is None:
        path = 'C:\\Users\\j2cle\\Work Docs\\Data\\Analysis\\Auto' + dt.now().strftime('%Y%m%d')
    if name is None:
        name = 'data_' + dt.now().strftime('%H_%M')
    if not os.path.exists(path):
        os.makedirs(path)

    if isinstance(data, (list, np.ndarray)):
        if isinstance(data[0], (pd.DataFrame, pd.Series)):
            data = {x: data[x] for x in range(len(data))}
        else:
            data = pd.DataFrame(data)

    if isinstance(data, (dict)):
        if not isinstance(data[list(data.keys())[0]], (pd.DataFrame, pd.Series)):
            data = pd.DataFrame(data)

    if isinstance(data, (pd.DataFrame, pd.Series)):
        data.to_excel(f'{path}/{slugify(name)}.xlsx', merge_cells=False)
    elif isinstance(data, (dict)):
        # writer = pd.ExcelWriter(f'{path}/{slugify(name)}.xlsx', engine='openpyxl')
        # for key, df in data.items():
        #     df.to_excel(writer, sheet_name=key)
        # writer.save()
        with pd.ExcelWriter(f'{path}/{slugify(name)}.xlsx') as writer:
            for key, df in data.items():
                df.to_excel(writer, sheet_name=key, merge_cells=False)


class PickleJar():
    def __init__(self, data=None, folder='Auto', path=None, history=False, **kwargs):
        self.history = history
        self.folder = folder
        if path is not None:
            self.path = path
        if data is not None:
            self.append(data)

    @property
    def database(self):
        """Return sum of squared errors (pred vs actual)."""
        for _database in os.walk(self.path):
            break
        return pd.Series(_database[2])

    @property
    def path(self):
        """Return sum of squared errors (pred vs actual)."""
        if not hasattr(self, '_path'):
            self._path = 'C:\\Users\\j2cle\\Work Docs\\Data\\Analysis\\Pickles'
            self._path = f'{self._path}\{self.folder}'
            if not os.path.exists(self._path):
                os.makedirs(self._path)
        return self._path

    @path.setter
    def path(self, value):
        self._path = value
        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def __setitem__(self, name, data):
        db = self.database
        name = slugify(name)
        if self.history and len(self.database) != 0:
            self.shift(name)

        with open(f'{self.path}\{name}', 'wb') as dill_file:
            dill.dump(data, dill_file)


    def __getitem__(self, name):
        if isinstance(name, (int, np.integer, float, np.float)) and int(name) < len(self.database):
            name = self.database[int(name)]
        else:
            name = slugify(name)

        if not self.database.isin([name]).any():
            name = difflib.get_close_matches(name,self.database)[0]
        with open(f'{self.path}\{slugify(name)}', 'rb') as dill_file:
            data = dill.load(dill_file)
        return data

    def shift(self, name):
        if len(self.database) == 0:
            return

        db = self.database[self.database.str.startswith(name)]
        itr = len(db[db.str.startswith(name)])
        if itr > 0:
            old = self.__getitem__(name)
            self.__setitem__(f'{name} ({itr})', old)

    def pickler(self, value):
        db = self.database

        if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
            name = value[0]
            data = value[1]
        elif isinstance(value, dict) and len(value) == 1:
            name = list(value.keys())[0]
            data = list(value.values())[0]
        else:
            data = value
            if len(db) == 0:
                itr = 0
            else:
                itr = len(db[db.str.startswith('data')])
            name = f'data ({itr})'

        self.__setitem__(name, data)

    def append(self, value):
        db = self.database
        if isinstance(value, dict):
            [self.pickler((key, val))  for key, val in value.items()]
        elif isinstance(value, (tuple, list, np.ndarray, pd.Series)) and len(np.array(value)[0]) == 2:
            [self.pickler(val) for val in value]
        else:
            self.pickler(value)

    def to_dict(self, value):
        if isinstance(value, dict):
            val_dict = {key: self.__getitem__(key) for key in value.keys()}
        elif isinstance(value, (tuple, list, np.ndarray, pd.Series)):
            if np.array(value).ndim == 1:
                val_dict = {val: self.__getitem__(val) for val in value}
            else:
                val_dict = {val[0]: self.__getitem__(val[0]) for val in value}
        else:
            val_dict = {value: self.__getitem__(value)}
        return val_dict

    def queary(self, value):
        if not isinstance(value, (tuple, list, np.ndarray)):
            value = [value]

        if len(self.database) == 0:
            return []
        res = self.database
        for val in value:
            res = res[res.str.contains(val)]
        return res

def curve_fit_wrap(fcn, pnts, **params):
    pnts = np.array(pnts)
    params = {**{"method":"trf", "x_scale": "jac", "xtol": 1e-12, "jac": "3-point"}, **params}
    fit = curve_fit(fcn, pnts[:,0], pnts[:,1], **params)
    return [fit[0], np.sqrt(np.diag(fit[1]))]



def map_plt(x, y, z,
            xscale='linear', yscale='linear', zscale='log',
            xlimit=[0, 0], ylimit=[0, 0], zlimit=[0, 0],
            levels=50, name="",
            xname="X", yname="Y", zname="Z", ztick=10, save=None, show=True,
            **kwargs):
    """ Calculate. generic discription """

    if xlimit == [0, 0]:
        xlimit = [min(x), max(x)]
    if ylimit == [0, 0]:
        ylimit = [min(y), max(y)]
    if zlimit[0] <= 0:
        zlimit = [z[z > 0].min(), z[z > 0].max()]
        # zlimit = [max(0, z.min()), z.max()]

    if 'log' in zscale:
        lvls = np.logspace(np.log10(zlimit[0]), np.log10(zlimit[1]), levels)
        tick_loc = ticker.LogLocator()
    else:
        lvls = np.linspace(zlimit[0], zlimit[1], levels)
        tick_loc = ticker.MaxNLocator()

    fig, ax = plt.subplots()
    csa = ax.contourf(
        x,
        y,
        z,
        lvls,
        locator=tick_loc,
        **kwargs,
    )

    ax.set_xlabel(xname, fontname="Arial", fontsize=18, fontweight="bold")
    ax.set_xlim(xlimit[0], xlimit[1])
    ax.set_ylabel(yname, fontname="Arial", fontsize=18, fontweight="bold")
    ax.set_ylim(ylimit[0], ylimit[1])
    # if "both" in logs.lower() or "x" in logs.lower():
    ax.set_xscale(xscale)
    # if "both" in logs.lower() or "y" in logs.lower():
    ax.set_yscale(yscale)
    ax.set_title(name, fontname="Arial", fontsize=20, fontweight="bold")

    for tick in ax.get_xticklabels():
        tick.set_fontname("Arial")
        tick.set_fontweight("bold")
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_fontweight("bold")
        tick.set_fontsize(12)


    if zscale == 'log':
        cbar = fig.colorbar(csa)
        cbar.locator = ticker.LogLocator(**ztick)
        cbar.set_ticks(cbar.locator.tick_values(zlimit[0], zlimit[1]))
    elif zscale == 'linlog':
        cbar = fig.colorbar(csa,format=ticker.LogFormatter(**ztick))
    else:
        cbar = fig.colorbar(csa)
    cbar.minorticks_off()
    cbar.set_label(zname, fontname="Arial", fontsize=18, fontweight="bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_fontweight("bold")
        tick.set_fontsize(12)
    plt.tight_layout()
    # xx,yy,zz = limitcontour(x,y,z,xlim=xlimit)

    # visual_levels = [1, 4, 7, 30, 365, 365*10]
    # lv_lbls = ['1 d', '4 d', '1 w', '1 mo', '1 yr', '10 yr']
    # ax = plt.gca()
    # csb = ax.contour(xx,yy,zz,visual_levels, colors='w',locator=ticker.LogLocator(),linestyles='--',norm=LogNorm(),linewidths=1.25)
    # csb.levels = lv_lbls

    # ax.clabel(csb, csb.levels, inline=True, fontsize=14, manual=False)
    if save is not None:
        if not os.path.exists(save):
            os.makedirs(save)
        plt.savefig(f'{save}/{slugify(name)}.png')

    if not show:
        plt.close()

    return


def scatter(data, x='index', y=None,
            xscale='linear', yscale='linear',
            xlimit=None, ylimit=None,
            name='Residual Plot', xname=None, yname=None, zname=None,
            hline=None, colorbar=False, save=None, show=True, fig=None, ax=None, **kwargs):
    """ Calculate. generic discription """
    sns.set_theme(context='talk', style='dark')

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if x == 'index':
        data['index'] = data.index
    if y is None:
        y = data.columns[0]

    if xlimit is None and all(x != data.index):
        xlimit = [data[x].min(), data[x].max()]
    elif xlimit is None:
        xlimit = [data.index.min(), data.index.max()]

    if ylimit is None:
        ylimit = [data[y].min(), data[y].max()]

    if xname is None:
        xname = x
    if yname is None:
        yname = y

    if fig is None and ax is None:
        fig, ax = plt.subplots()
    # Loading the dataset into the variable 'dataset'
    # Graph is created and stored in the variable 'graph' *added ax to g
    g = sns.scatterplot(x=x, y=y, data=data, ax=ax, **kwargs)
    # Drawing a horizontal line at point 1.25
    g.set(xlim=xlimit, ylim=ylimit, xscale=xscale, yscale=yscale)
    try:
        if hline is not None and (yscale != 'log' or any(np.array(hline) > 0)):
            for h in hline:
                ax.axhline(h, color='k', linestyle=':')
    except TypeError:
        if hline is not None and (yscale != 'log' or hline > 0):
            ax.axhline(hline, color='k', linestyle=':')

    g.set_xlabel(xname, fontname="Arial", fontsize=18, fontweight="bold")
    g.set_ylabel(yname, fontname="Arial", fontsize=18, fontweight="bold")
    ax.set_title(name, fontname="Arial", fontsize=20, fontweight="bold")

    for tick in ax.get_xticklabels():
        tick.set_fontname("Arial")
        tick.set_fontweight("bold")
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_fontweight("bold")
        tick.set_fontsize(12)

    if colorbar:
        if zname is None:
            zname = kwargs['hue']
        norm = plt.Normalize(data[kwargs['hue']].min(), data[kwargs['hue']].max())
        sm = plt.cm.ScalarMappable(cmap=kwargs['palette'], norm=norm)
        sm.set_array([])


        # Remove the legend and add a colorbar
        ax.get_legend().remove()
        # ax.figure.colorbar(sm)
        cbar =  ax.figure.colorbar(sm)
        cbar.set_label(zname, fontname="Arial", fontsize=18, fontweight="bold")
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname("Arial")
            tick.set_fontweight("bold")
            tick.set_fontsize(12)


    #The plot is shown
    plt.tight_layout()
    if save is not None:
        if not os.path.exists(save):
            os.makedirs(save)
        plt.savefig(f'{save}/{slugify(name)}.png')

    if not show:
        plt.close()

def nyquist(data, freq=None, fit=None, band=None, bmin="min", bmax="max", title="Nyquist"):
    data = data.copy()
    # if freq is not None:
    #     data["freq"] = np.trunc(np.log10(freq))
    # else:
    #     data["freq"] = np.trunc(np.log10(data["freq"]))
    if freq is not None:
        data["freq"] = freq

    fig, ax = plt.subplots()
    sns.scatterplot(
        x="real",
        y="inv_imag",
        data=data,
        hue="freq",
        palette="kindlmann",
        hue_norm=plt.matplotlib.colors.LogNorm(),
        legend=False,
        edgecolor="none",
        ax=ax,
    )
    if fit is not None:
        ax.plot(fit["real"], fit["inv_imag"])
        if band is not None:
            ax.fill_between(band["real"], band[bmin], band[bmax], color='r', alpha=.5)

    ax.set_xlabel("Z' [Ohms]", fontname="Arial", fontsize=18, fontweight="bold")
    ax.set_ylabel("-Z'' [Ohms]", fontname="Arial", fontsize=18, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim", anchor="SW", share=True)
    ax.grid(True)
    ax.set_title(title, fontname="Arial", fontsize=18, fontweight="bold")

    norms = plt.matplotlib.colors.LogNorm(
        data["freq"].min(), data["freq"].max()
    )
    sm = plt.cm.ScalarMappable(cmap="kindlmann", norm=norms)
    cbar = ax.figure.colorbar(sm)
    cbar.set_label("Freq", fontname="Arial", fontsize=18, fontweight="bold")

    labels = np.unique(np.floor(np.log10(data["freq"])),return_index=True)[1]
    for label in labels:
        ax.annotate(data.loc[label,"freq"], (data.loc[label,"real"]+0.2, data.loc[label,"inv_imag"]+0.2))

    plt.show()

def bode(data, freq=None, top="mag", bot="phase", fit=None, band=None, bmin="min", bmax="max", title="bode"):
    if freq is not None:
        data["freq"] = freq
    labels = {"real": "Z' [Ohms]",
              "imag": "Z'' [Ohms]",
              "inv_imag": "-Z'' [Ohms]",
              "mag": "|Z| [Ohms]",
              "phase": "Phase [deg]",
              "inv_phase": "Inv Phase [deg]",
              }

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    sns.scatterplot(
        x="freq",
        y=top,
        data=data,
        edgecolor="none",
        ax=ax1,
    )
    sns.scatterplot(
        x="freq",
        y=bot,
        data=data,
        legend=False,
        edgecolor="none",
        ax=ax2,
    )
    if fit is not None:
        # sns.lineplot(x=data["freq"], y=top, data=fit, ax=ax1)
        # sns.lineplot(x=data["freq"], y=bot, data=fit, ax=ax2)
        ax1.plot(fit["freq"], fit[top])
        ax2.plot(fit["freq"], fit[bot])

        if band is not None:
            ax1.fill_between(data["freq"], band[top][bmin], band[top][bmax], color='r', alpha=.4)
            ax2.fill_between(data["freq"], band[bot][bmin], band[bot][bmax], color='r', alpha=.4)


    ax1.set(
        xscale="log",
        xlim=[data["freq"].min(), data["freq"].max()],
        yscale="log",
        ylim=[data[top].min(), data[top].max()],
    )
    # ax2.set(ylim=[-90, 90])
    if "phase" in bot.lower():
        ax2.yaxis.set_ticks(np.arange(-90, 90, 30))
    else:
        ax2.set(yscale="log", ylim=[data[bot].min(), data[bot].max()])

    ax2.set_xlabel("Frequency Hz", fontname="Arial", fontsize=18, fontweight="bold")
    ax1.set_ylabel(labels[top], fontname="Arial", fontsize=18, fontweight="bold")
    ax2.set_ylabel(labels[bot], fontname="Arial", fontsize=18, fontweight="bold")
    ax1.set_title(title, fontname="Arial", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
mats=['boro','boro alt','soda','soda alt','eva','eva_alt','sinx','si','air','poe_a','poe_b','poe_c']
mat_data_list=[[0.04495210144020945, 0.9702835437735396, tocm(0.125, 'in'), 4.6*tocm(PERM,'m', inv=True)],
          [0.6580761914650776, 0.8507365332956724, tocm(0.125, 'in'), 4.6*tocm(PERM,'m', inv=True)],
          [5644.5501772317775, 0.3590103059601377, tocm(0.125, 'in'), 7.5*tocm(PERM,'m', inv=True)],
          [0.023546001890093236, 0.9471251012618027, tocm(0.125, 'in'), 7.5*tocm(PERM,'m', inv=True)],
          [19863.639619529386, 0.6319537614631568, tocm(450, 'um'), 2.65*tocm(PERM,'m',inv=True)],
          [1e13, 0, tocm(450, 'um'), 2.65*tocm(PERM,'m',inv=True)],
          [1e13, 0, tocm(80, 'nm'), 7*tocm(PERM,'m',inv=True)],
          [0.1, 0, tocm(80, 'nm'), 11.68*tocm(PERM,'m',inv=True)],
          [1e18, 0, tocm(80, 'nm'), tocm(PERM,'m',inv=True)],
          [9e13, 0, tocm(80, 'nm'), tocm(PERM,'m',inv=True)],
          [4e14, 0, tocm(80, 'nm'), tocm(PERM,'m',inv=True)],
          [1e16, 0, tocm(80, 'nm'), tocm(PERM,'m',inv=True)],
          ]

mat_database = pd.DataFrame(mat_data_list,columns=['pre','ea','thick','perm'],index=mats)

diff_arrh = pd.DataFrame([[8.038e-8, 0.544],
                          [8.52e-7, 0.6],
                          [2.03e-8, 0.53],
                          [6.953e-8, 0.533]],
                         columns=['pre', 'ea'],
                         index=['ave', 'max', 'min', 'tof'])

mat_data = pd.read_excel('C:\\Users\\j2cle\\Work Docs\\Data\\Databases\\material_data.xlsx', index_col=[0, 1, 2])
