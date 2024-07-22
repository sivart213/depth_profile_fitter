# -*- coding: utf-8 -*-
"""
Insert module description/summary.

Provide any or all of the following:
1. extended summary
2. routine listings/functions/classes
3. see also
4. notes
5. references
6. examples

@author: j2cle
Created on Fri Jun 21 15:09:33 2024
"""

# %% Imports
import re
import os
import sys
import numpy as np
from pathlib import Path
import ctypes
import itertools
# Drive list
def find_drives(exclude_nonlocal=True, exclude_hidden=True, limit_drive_letters=True, **kwargs):
    if sys.platform.startswith("win"):
        import win32net
        resume = 0
        net_dr = []
        while 1:
            net_res, _, resume = win32net.NetUseEnum (None, 0, resume)
            for dr in net_res:
                net_dr.append(Path(dr["local"]))
                net_dr.append(Path(dr["remote"]))
            if not resume: break
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        drive_bitmask = ctypes.cdll.kernel32.GetLogicalDrives()
        drives = list(map(Path, map("{}:/".format, itertools.compress(letters,
                   map(lambda x:ord(x) - ord('0'), bin(drive_bitmask)[:1:-1])))))
        if exclude_nonlocal:
            drives = [dr for dr in drives if Path(dr.drive) not in net_dr]

    elif sys.platform.startswith("linu"):
        drives = [dr for dr in Path("/").glob("m*/*") if dr.is_dir() and any(dr.iterdir())]
        for drn, dr in enumerate(drives):
            dr_f = [x for x in os.listdir(dr)]
            while len(dr_f) == 1:
                drives[drn] = dr/dr_f[0]
                dr_f = [x for x in drives[drn].iterdir()]
        drives.append(Path.home())
        drives.append(Path("/"))
    elif sys.platform.startswith("darw"):
        drives = [dr for dr in Path("/").glob("Vol*/*") if dr.is_dir() and any(dr.iterdir())]
        for drn, dr in enumerate(drives):
            dr_f = [x for x in os.listdir(dr)]
            while len(dr_f) == 1:
                drives[drn] = dr/dr_f[0]
                dr_f = [x for x in drives[drn].iterdir()]
        drives.append(Path.home())
        drives.append(Path("/"))
    else:
        drives = [Path.home(), Path(Path.home().parts[0])]
    
    if exclude_nonlocal:
        drives = [dr for dr in drives if os.path.realpath(dr) == str(dr)]
    
    if exclude_hidden:
        drives = [dr for dr in drives if not str(dr).lower().endswith(".hidden")]
    
    return drives

def find_path(*dir_in, base=None, as_list=False, by_re=True, by_glob=False, **kwargs):
    """
    Find file path as quickly as possible.

    Currently creates a Path obj from the dir_in list and compares it to the base

    base options, cwd, home, sys.argv (originating file)

    relative -> NameError
    Parameters
    ----------
    dir_in : str(s), Path
        tuple of strings
    as_list : bool, optional
        Iterate p_find for each dir in returning a list of paths.
    kwargs :


    Returns
    -------
    dir_path : Path
        Return a windows or posix path object.
    """
    # Validate formatting
    if len(dir_in) == 1:
        if isinstance(dir_in[0], (list, np.ndarray)):
            dir_in = list(dir_in[0])
        elif isinstance(dir_in[0], (str, Path)):
            if not str(dir_in[0]).isprintable():
                dir_in = (repr(str(dir_in[0]))[1:-1],)
            dir_in = re.split(r"[\\/]+", str(dir_in[0]))

    if Path(*dir_in).exists():
        return Path(*dir_in)

    if as_list:
        return [find_path(d, **kwargs) for d in dir_in]
    # else:
    #     dir_in = []
    
    def overlap(path1, *path2):
        if len(path2) >= 1 and Path(*path2).parts[0] in Path(path1).parts:
            for b in path1.parents:
                if path2[0] not in b.parts:
                    path1 = b
                    break
        return path1
    drives = []

    # Get base path, if string it should either be the path, home, or cwd
    if isinstance(base, str):
        if base.lower() in ["local", "caller", "argv", "sys.argv"]:
            base = Path(sys.argv[0]).resolve().parent
        elif "drive" in base.lower():
            drives = find_drives(**kwargs)
            base_path = [p for d in drives for p in d.glob("*/" + str(Path(*dir_in)))]
            if base_path == []:
                base_path = [p for d in drives for p in d.glob("*/*/" + str(Path(*dir_in)))]
            if base_path == []:
                base = None
            else:
                base_path.sort(key=lambda x: len(Path(x).parts))
                base = base_path[0]

        else:
            base = getattr(Path, base)() if hasattr(Path, base) else Path(base)
    if base is None or not isinstance(base, Path) or not base.exists():
        base = Path.home() / "Documents"

    # if there may be overlap, shrink base path until there isn't overlap
    base = overlap(base, *dir_in)
    
    # try just merging without glob
    if (base / Path(*dir_in)).exists():
        return base / Path(*dir_in)
    if (base.parent / Path(*dir_in)).exists():
        return base.parent / Path(*dir_in)
    
    # Drive list
    if drives == []:
        drives = find_drives(**kwargs)
    
    bases_all = [base, Path.cwd(), Path.home()] + drives
    # bases_all = [base, Path.cwd(), Path.cwd().parent, Path.home()] + drives
    bases = []
    for b in bases_all:
        bases.append(overlap(b, *dir_in)) if b not in bases else None
    
    # bases.sort(key=lambda x: len(Path(x).parts), reverse=True)
    
    if "ftype" not in kwargs.keys():
        ftype = "file" if Path(*dir_in).suffix else "dir"
    else:
        ftype = kwargs.get("ftype", "dir")
    # paths_t = find_files(bases[3], "path", ftype, "Users", ignore=[str(b) for b in bases[:3]], recursive=True)
    # p_filt = lambda x: all(re.search(l, x.path) or x.path.find(str(l))+1 for l in Path(*dir_in).parts)
    paths = []
    if by_glob:
        paths = [p for b in bases for p in b.glob("*/" + str(Path(*dir_in)))]
    if by_re or not by_glob:
        for b in bases:
            paths = paths + find_files(b, "path", ftype, Path(*dir_in).parts,  recursive=False)
    
    # paths_t = find_files(bases[3], "path", ftype, "Users", ignore=[str(b) for b in bases[:3]], recursive=True)
    n = 0
    while n < len(bases) and paths == []:
        if by_glob:
            paths = list(bases[n].glob("**/" + str(Path(*dir_in))))
        if by_re or not by_glob:    
            paths = paths + find_files(bases[n], "path", ftype, Path(*dir_in).parts, ignore=bases[:n], recursive=True)
        n+=1
    
    paths.sort(key=lambda x: len(Path(x).parts))
    if len(paths) == 1:
        return paths[0]
    elif len(paths) > 1 and Path(paths[0]).exists():
        return paths[0]


    return base / Path(*dir_in)

def parse_path_str(arg):
    if isinstance(arg, (list, np.ndarray)):
        return list(filter(None, arg))
    elif isinstance(arg, (str, Path)):
        return list(filter(None, re.split(r"[\\/]+", str(repr(str(arg))[1:-1]))))
        # if not str(arg).isprintable():
            # dir_in = (repr(str(arg))[1:-1],)
        # return re.split(r"[\\/]+", str(repr(str(arg))[1:-1]))
        # else:
        #     return re.split(r"[\\/]+", str(arg))


def my_walk(path, ftype=None, recursive=True, ignore=None):
    """Recursively yield DirEntry objects (files only) for given directory."""
    try:
        for x in os.scandir(Path(path)):
            # if callable(ignore) and ignore(x):
            #     continue
            # elif isinstance(ignore, (list, np.ndarray)) and (x.path in ignore or Path(x.path) in ignore):
            #     continue
            if x.name.startswith(".") or x.name.startswith("$"):
                continue
            elif x.is_dir(follow_symlinks=False):
                if not ftype or "dir" in ftype.lower():
                    yield Path(x)
                if recursive:
                    yield from my_walk(x.path, ftype, True, ignore)  # see below for Python 2.x    
            elif not ftype or "file" in ftype.lower():
                yield Path(x)
    except (PermissionError, NotADirectoryError):
        pass


def my_filter(condition, gen, kill=False):
    """Recursively yield DirEntry objects (files only) for given directory."""
    try:
        if isinstance(gen, list):
            gen = iter(gen)
        while True:
            g = next(gen)
            match = condition
            if callable(condition):
                match = condition(g)
            if match:
                yield g
                if kill:
                    break
    except (StopIteration, AttributeError):
        return

def find_files(path, attr='', ftype=None, re_filter=None, ignore=[], recursive=False, kill=False, function=None):
    if not re_filter:
        re_filter = [r".*"]
    # if isinstance(re_filter, str):
    #     re_filter = lambda x: re.search(re_filter, str(x)) or str(x).find(str(re_filter))+1
    # elif isinstance(re_filter, Path):
    #     re_filter = lambda x: all(re.search(l, str(x)) or str(x).find(str(l))+1 for l in re_filter.parts)
        
    
    # filt_re = lambda x: all(re.search(l, str(x)) for l in re_filter)

    # filt_ign = lambda x: str(x) not in ignore
    
    file_search = my_walk(Path(path), ftype, recursive)
    if not re_filter:
        filesurvey = list(
            my_filter(
                lambda x: not any(str(x).startswith(str(i)) for i in ignore), 
                my_filter(
                    lambda x: all(re.search(l, str(x)) or str(x).find(str(l))+1 for l in re_filter), 
                    file_search, 
                    kill
                    )
                )
            )
    # file_search = my_walk(Path(path), ftype, recursive, ignore)
    filesurvey = list(my_filter(lambda x: not any(str(x).startswith(str(i)) for i in ignore), my_filter(lambda x: all(re.search(l, str(x)) or str(x).find(str(l))+1 for l in re_filter), file_search, kill)))
    # filesurvey.sort(key=lambda x: x.inode())
        
    if filesurvey == []:
        return filesurvey
    if hasattr(filesurvey[0], attr):
        if attr.lower() == "path":
            return [Path(str(f)) for f in filesurvey]
        return [getattr(f, attr) for f in filesurvey]
    if hasattr(filesurvey[0].stat(), attr):
        return [getattr(f.stat(), attr) for f in filesurvey]
    return filesurvey

def find_files2(path, *filters, attr='', ftype=None, re_filter=None, ignore=[], recursive=False, kill=False, function=None):
    # if not re_filter:
    #     re_filter = [r".*"]
    
    
    if isinstance(re_filter, str):
        re_filter = lambda x: re.search(re_filter, str(x)) or str(x).find(str(re_filter))+1
    elif isinstance(re_filter, Path):
        re_filter = lambda x: all(re.search(l, str(x)) or str(x).find(str(l))+1 for l in re_filter.parts)
        
    
    # filt_re = lambda x: all(re.search(l, str(x)) for l in re_filter)

    # filt_ign = lambda x: str(x) not in ignore
    
    filesurvey = my_walk(Path(path), ftype, recursive)
    
    for f in filters:
        if len(f) == 2:
            filesurvey = my_filter(f[0], filesurvey, f[1])
        else:
            filesurvey = my_filter(f, filesurvey)
    filesurvey = list(filesurvey)
    # if not re_filter:
    #     filesurvey = list(my_filter(lambda x: not any(str(x).startswith(str(i)) for i in ignore), my_filter(lambda x: all(re.search(l, str(x)) or str(x).find(str(l))+1 for l in re_filter), file_search, kill)))
    # file_search = my_walk(Path(path), ftype, recursive, ignore)
    # filesurvey = list(my_filter(lambda x: str(x) not in ignore, my_filter(lambda x: re.search(re_filter, str(x)), file_search, kill)))
    # filesurvey = list(my_filter(lambda x: str(x) not in ignore, my_filter(lambda x: all(re.search(l, str(x)) for l in re_filter), file_search, kill)))
    # filesurvey = list(my_filter(lambda x: Path(str(x)) not in ignore, my_filter(lambda x: all(re.search(l, str(x)) for l in re_filter), file_search, kill)))
    # filesurvey = list(my_filter(lambda x: not any(str(x).startswith(str(i)) for i in ignore), my_filter(lambda x: all(re.search(l, str(x)) or str(x).find(str(l))+1 for l in re_filter), file_search, kill)))
    # filesurvey = list(my_filter(lambda x: all(re.search(l, str(x)) for l in re_filter), my_filter(lambda x: str(x) not in ignore, file_search), kill))
    # filesurvey = list(my_filter(lambda x: all(re.search(l, str(x)) for l in re_filter), file_search, kill))
    # filesurvey.sort(key=lambda x: x.inode())
        
    if filesurvey == []:
        return filesurvey
    if hasattr(filesurvey[0], attr):
        if attr.lower() == "path":
            return [Path(f.path) for f in filesurvey]
        return [getattr(f, attr) for f in filesurvey]
    if hasattr(filesurvey[0].stat(), attr):
        return [getattr(f.stat(), attr) for f in filesurvey]
    return filesurvey


#%%
# my_path = find_path(r"Dropbox*\Work Docs\Data\Analysis\SIMS")

# my_path = find_path3(r"\Work Docs\Data\Analysis\SIMS")
# test = "Documents\Python\impedance_analysis\testing\circuits.ini"
# my_path = find_path_parts("D:/Documents/Python/impedance_analysis/testing/circuits.ini")
# my_path = find_path(r"Documents\Python\impedance_analysis\testing")
# my_path = find_path("Python\impedance_analysis\testing")

my_path = find_path(r"Analysis\SIMS", base=find_path(r"ASU Dropbox", base="drive"))
# test0  = find_files(Path().cwd().parent, "path", "dir", "eis_analysis",  recursive=True)
# test1  = find_files(Path().cwd().parent, "path", "dir", None,  recursive=True)




