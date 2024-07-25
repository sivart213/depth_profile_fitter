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
from pathlib import Path
import re
import os
import sys
import numpy as np
import ctypes
import itertools


# Drive list
def find_drives(exclude_nonlocal=True, exclude_hidden=True):
    """
    Finds and returns a list of drive paths available on the system.

    This function dynamically identifies available drives based on the operating system.
    For Windows, it includes both local and network drives. For Linux and Darwin (macOS),
    it searches for directories matching certain patterns and includes the home directory
    and root. The function can exclude non-local (network) drives and hidden drives based
    on the parameters provided.

    Parameters:
    - exclude_nonlocal (bool): If True, excludes network drives from the result. Default is True.
    - exclude_hidden (bool): If True, excludes drives that are marked as hidden. Default is True.

    Returns:
    - list: A list of pathlib.Path objects representing the paths to the drives found.

    Note:
    - On Windows, network drives are detected using the win32net module and local drives are
    identified using ctypes to call GetLogicalDrives. The function filters out non-local or
    hidden drives based on the parameters.
    - On Linux, it looks for directories under "/" that match "m*/*" and checks if they are
    directories with contents. It also adds the home directory and root ("/") to the list.
    - On Darwin (macOS), it performs a similar search under "/" for directories matching "Vol*/*".
    - For other platforms, it defaults to adding the home directory and its root.
    - The function also provides an option to exclude drives that are symbolic links or
    do not match their realpath, aiming to filter out non-local drives.
    - Hidden drives are determined by a simple check if the drive's name ends with ".hidden".
    """
    if sys.platform.startswith("win"):
        drives = detect_windows_drives(exclude_nonlocal=exclude_nonlocal)
    elif sys.platform.startswith("linu"):
        drives = detect_posix_drives("m*/*", exclude_nonlocal)
    elif sys.platform.startswith("darw"):
        drives = detect_posix_drives("Vol*/*", exclude_nonlocal)
    else:
        drives = [Path.home(), Path(Path.home().parts[0])]

    if exclude_hidden:
        drives = [dr for dr in drives if not str(dr).lower().endswith(".hidden")]

    return drives

def detect_windows_drives(letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ", exclude_nonlocal=True):
    """
    Detects and returns a list of drive paths available on a Windows system, with options to exclude non-local drives.

    This function identifies available drives by querying the system for logical drives and network drives. It uses
    the win32net module to enumerate network drives and ctypes to access the GetLogicalDrives Windows API function,
    which provides a bitmask representing the drives available on the system. The function can be configured to
    exclude non-local (network) drives based on the parameters provided.

    Parameters:
    - letters (str, optional): A string containing the uppercase alphabet letters used to check for drive presence.
      Default is "ABCDEFGHIJKLMNOPQRSTUVWXYZ".
    - exclude_nonlocal (bool, optional): If True, excludes network drives from the result. Default is True.

    Returns:
    - list of pathlib.Path: A list of Path objects representing the paths to the drives found on the system.
      Each Path object corresponds to a drive root (e.g., C:/).

    Note:
    - Network drives are detected using the win32net.NetUseEnum function, which enumerates all network connections.
      The function checks each connection's status to determine if it should be considered a drive.
    - Local drives are identified by converting the bitmask returned by GetLogicalDrives into drive letters.
    - If exclude_nonlocal is True, the function filters out drives that are mapped to network locations.
    """
    import win32net

    resume = 0
    net_dr = []
    while 1:
        net_res, _, resume = win32net.NetUseEnum(None, 0, resume)
        for dr in net_res:
            net_dr.append(Path(dr["local"]))
            net_dr.append(Path(dr["remote"]))
        if not resume:
            break

    drive_bitmask = ctypes.cdll.kernel32.GetLogicalDrives()
    drives = list(
        map(
            Path,
            map(
                "{}:/".format,
                itertools.compress(
                    letters,
                    map(lambda x: ord(x) - ord("0"), bin(drive_bitmask)[:1:-1]),
                ),
            ),
        )
    )
    if exclude_nonlocal:
        drives = [dr for dr in drives if Path(dr.drive) not in net_dr]
        drives = [dr for dr in drives if os.path.realpath(dr) == str(dr)]
    return drives

def detect_posix_drives(pattern="m*/*", exclude_nonlocal=True):
    """
    Detects and returns a list of drive paths available on POSIX-compliant systems (e.g., Linux, macOS).

    This function identifies available drives by searching for directories that match a specified pattern
    at the root ("/") directory level. It then checks if these directories are actual mount points by
    verifying they contain subdirectories. Optionally, it can exclude drives that are mounted from network
    locations based on the realpath comparison, aiming to filter out non-local drives.

    Parameters:
    - pattern (str, optional): The glob pattern used to identify potential drives at the root directory.
      Default is "m*/*", which aims to target mnt and media directories having at least one subdirectory typical of Linux structures.  Alternatively, utilize "Vol*/*" for macOS.
    - exclude_nonlocal (bool, optional): If True, excludes drives that do not have their realpath matching
      their path, which typically indicates network-mounted drives. Default is True.

    Returns:
    - list of pathlib.Path: A list of Path objects representing the mount points of the drives found on the
      system. Each Path object corresponds to a drive's mount point.

    Note:
    - The function initially searches for directories at the root ("/") that match the specified pattern.
    - It then filters these directories to include only those that contain at least one subdirectory,
      under the assumption that a valid drive mount point will have subdirectories.
    - The home directory and the root directory are always included in the list of drives.
    - If exclude_nonlocal is True, the function filters out drives that are mounted from network locations
      by comparing each drive's realpath to its original path. Drives with differing realpaths are considered
      non-local and excluded from the results.
    """
    drives = [
        dr for dr in Path("/").glob(pattern) if dr.is_dir() and any(dr.iterdir())
    ]
    for drn, dr in enumerate(drives):
        dr_f = [x for x in os.listdir(dr)]
        while len(dr_f) == 1:
            drives[drn] = dr / dr_f[0]
            dr_f = [x for x in drives[drn].iterdir()]
    drives.append(Path.home())
    drives.append(Path("/"))
    
    if exclude_nonlocal:
        drives = [dr for dr in drives if os.path.realpath(dr) == str(dr)]
    return drives


def find_path(*dir_in, base=None, as_list=False, by_re=True, by_glob=False, **kwargs):
    """
    Searches for paths matching the specified directory names within given base directories,
    optionally using glob patterns or regular expressions. If no paths are found directly,
    performs a recursive search within each base directory.

    Parameters:
    - dir_in (tuple): Directory names to be joined together and searched for. Can be a mix of strings and Path objects.
    - base (Path or str, optional): The base directory or directories to search within. If not specified, uses a default set of base directories.
    - as_list (bool, optional): If True, assumes dir_in is a list of targets.Returns a list of each result for a given dir_in value. Default is False.
    - by_re (bool, optional): If True, uses regular expressions for searching. Default is True.
    - by_glob (bool, optional): If True, uses glob patterns for searching. Overrides by_re if both are True. Default is False.
    - **kwargs: Additional keyword arguments, reserved for future use.

    Returns:
    - Path or list of Path: Depending on the value of `as_list`, either the first matching Path object or a list of all matching Path objects.

    The function first attempts to find paths directly within the specified base directories. If no matches are found,
    it recursively searches within each base directory. The search can be performed using either glob patterns or regular
    expressions, with glob patterns taking precedence if both are specified. The function sorts the found paths by their
    length, preferring shorter paths, and returns either the first match or a list of all matches based on the `as_list` parameter.
    """
    # Validate formatting
    dir_in = parse_path_str(dir_in)

    if Path(*dir_in).exists():
        return Path(*dir_in)

    if as_list:
        return [find_path(d, **kwargs) for d in dir_in]

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
            drives = find_drives(exclude_nonlocal=True, exclude_hidden=True, **kwargs)
            base_path = [p for d in drives for p in d.glob("*/" + str(Path(*dir_in)))]
            if base_path == []:
                base_path = [
                    p for d in drives for p in d.glob("*/*/" + str(Path(*dir_in)))
                ]
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

    # Try to find drives without exclusions
    if drives == []:
        drives = find_drives(**kwargs)

    # get list of possible bases
    bases_all = [base, Path.cwd(), Path.home()] + drives

    bases = []
    for b in bases_all:
        if b not in bases:
            bases.append(overlap(b, *dir_in))

    # define target type
    res_type = kwargs.get("res_type", "file" if Path(*dir_in).suffix else "dir")

    # Search one level down from available bases
    paths = []
    if by_glob:
        paths = [p for b in bases for p in b.glob("*/" + str(Path(*dir_in)))]
    if by_re or not by_glob:
        for b in bases:
            paths = paths + find_files(
                b, "path", res_type, Path(*dir_in).parts, recursive=False
            )

    # if paths are not found, do a recursive search
    n = 0
    while n < len(bases) and paths == []:
        if by_glob:
            paths = list(bases[n].glob("**/" + str(Path(*dir_in))))
        if by_re or not by_glob:
            paths = paths + find_files(
                bases[n],
                "path",
                res_type,
                Path(*dir_in).parts,
                ignore=bases[:n],
                recursive=True,
            )
        n += 1

    # Sort by shortest path and return 1st option
    paths.sort(key=lambda x: len(Path(x).parts))
    if len(paths) == 1:
        return paths[0]
    elif len(paths) > 1 and Path(paths[0]).exists():
        return paths[0]

    return base / Path(*dir_in)


def parse_path_str(arg):
    """
    Parses a path string or a list of path strings into a normalized list of path components.

    This function takes a single argument which can be a string, a pathlib.Path object, a list of strings,
    or a numpy.ndarray of strings, representing one or more paths. It normalizes these paths by splitting them
    into their individual components (directories and file names), filtering out any empty components or redundant
    separators. The function is designed to handle various path formats and separators, making it useful for
    cross-platform path manipulation.

    Parameters:
    - arg (str, Path, list, np.ndarray): The path or paths to be parsed. This can be a single path string,
      a pathlib.Path object, a list of path strings, or a numpy.ndarray of path strings.

    Returns:
    - list: A list of the path components extracted from the input. If the input is a list or an array,
      the output will be a flattened list of components from all the paths.

    Note:
    - The function uses regular expressions to split path strings on both forward slashes (`/`) and backslashes (`\\`),
      making it suitable for parsing paths from both Unix-like and Windows systems.
    - Path components are filtered to remove any empty strings that may result from consecutive separators or leading/trailing separators.
    - The function handles string representations of paths by stripping enclosing quotes before parsing, which is particularly
      useful when dealing with paths that contain spaces or special characters.
    """
    if isinstance(arg, (str, Path)):
        return list(filter(None, re.split(r"[\\/]+", str(repr(str(arg))[1:-1]))))
    elif isinstance(arg, (tuple, list, np.ndarray)):
        return list(filter(None, arg))
    return arg


def my_walk(path, res_type=None, recursive=True, ignore=None, ignore_hidden=True):
    """
    Recursively yields Path objects for files and/or directories within a given directory,
    based on specified criteria.

    This function traverses the directory tree starting from `path`, yielding Path objects
    for files and directories that match the specified criteria. It allows filtering based
    on resource type (files, directories), recursion control, and the ability to ignore
    specific paths or hidden files/directories.

    Parameters:
    - path (str or Path): The root directory from which to start walking.
    - res_type (str, optional): Specifies the type of resources to yield. Can be 'file', 'dir', or None.
      If None, both files and directories are yielded. Default is None.
    - recursive (bool, optional): If True, the function will recursively walk through subdirectories.
      If False, only the immediate children of `path` are processed. Default is True.
    - ignore (list, np.ndarray, or callable, optional): A list of paths to ignore, an array of paths to ignore,
      or a callable that takes a DirEntry object and returns True if it should be ignored. Default is None.
    - ignore_hidden (bool, optional): If True, hidden files and directories (those starting with '.' or '$')
      are ignored. Default is True.

    Yields:
    - Path: Path objects for each file or directory that matches the specified criteria.

    Note:
    - The function can handle large directories by yielding results as it walks the directory tree,
      rather than building a list of results in memory.
    - The `ignore` parameter provides flexibility in filtering out unwanted paths, either through a list,
      an array, or custom logic implemented in a callable.
    - Hidden files and directories are identified by their names starting with '.' or '$'.
    - PermissionError and NotADirectoryError are silently caught, allowing the walk to continue
      in case of inaccessible or invalid directories.
    """
    try:

        if isinstance(ignore, (list, np.ndarray)) and len(ignore) > 0:
            ignore_list = ignore
            ignore = (
                lambda var: var.path in ignore_list or Path(var.path) in ignore_list
            )
        elif not callable(ignore):
            ignore = lambda var: False

        for x in os.scandir(Path(path)):
            if (
                ignore_hidden and (x.name.startswith(".") or x.name.startswith("$"))
            ) or ignore(x):
                continue
            elif x.is_dir(follow_symlinks=False):
                if not res_type or "dir" in res_type.lower():
                    yield Path(x)
                if recursive:
                    yield from my_walk(x.path, res_type, True, ignore, ignore_hidden)
            elif not res_type or "file" in res_type.lower():
                yield Path(x)
    except (PermissionError, NotADirectoryError):
        pass


def my_filter(condition, gen, yield_first_match=False):
    """
    Filters items from a generator or list based on a specified condition, optionally yielding only the first match.

    This function applies a filtering condition to each item yielded by a generator or contained in a list. It yields
    items for which the condition evaluates to True. The condition can be a boolean value or a callable that takes an
    item and returns a boolean. If `yield_first_match` is True, the function yields the first matching item and then
    terminates; otherwise, it yields all matching items.

    Parameters:
    - condition (bool or callable): The condition to apply to each item. If a boolean, it directly determines whether
      to yield items. If a callable, it should accept an item and return a boolean indicating whether the item matches.
    - gen (generator or list): The generator or list from which to filter items. If a list is provided, it is converted
      to a generator.
    - yield_first_match (bool, optional): If True, the function yields the first item that matches the condition and
      then stops. If False, it yields all items that match the condition. Default is False.

    Yields:
    - The next item from `gen` that matches the condition specified by `condition`. If `yield_first_match` is True,
      only the first matching item is yielded.

    Note:
    - This function is designed to work with both generators and lists, providing flexibility in handling different
      types of iterable inputs.
    - The ability to yield only the first match can be useful in scenarios where only one matching item is needed,
      potentially improving performance by avoiding unnecessary processing.
    """
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
                if yield_first_match:
                    break
    except (StopIteration, AttributeError):
        return


def find_files(
    path,
    attr="",
    res_type=None,
    patterns=None,
    ignore=None,
    recursive=False,
    yield_first_match=False,
    yield_shortest_match=False,
):
    """
    Searches for files or directories within a given path that match specified patterns,
    optionally filtering by resource type and controlling recursion and result quantity.

    This function is designed to find files or directories that match a set of regular expression patterns within a specified path.
    It allows for filtering the search results based on the type of resource (file or directory), and can perform either a
    non-recursive (default) or recursive search. Additionally, it can be configured to yield only the first match found or the shortest match based on the specified criteria.

    Parameters:
    - path (str or Path): The root directory from which to start the search.
    - attr (str, optional): Additional attributes to filter the search. Default is an empty string.
    - res_type (str, optional): Specifies the type of resources to search for. Can be 'file', 'dir', or None. If None,
    both files and directories are included in the search results. Default is None.
    - patterns (str, Path, list of str, optional): Regular expression patterns to match against the names of the files
    or directories. If None, all files or directories are considered a match. Default is None.
    - ignore (str, Path, list of str, optional): Regular expression patterns for files or directories to ignore. Default is None.
    - recursive (bool, optional): If True, the function will search through all subdirectories of the given path. If False,
    only the immediate children of the path are considered. Default is False.
    - yield_first_match (bool, optional): If True, the function returns the first matching file or directory and stops the
    search. If False, all matching files or directories are returned. Default is False.
    - yield_shortest_match (bool, optional): If True, the function returns the match with the shortest path. This is useful
    when searching for the most relevant result. Default is False.

    Returns:
    - list: A list of Path objects for each file or directory that matches the specified criteria. If `yield_first_match`
      is True, the list contains at most one Path object.

    Note:
    - The function uses regular expressions for pattern matching, allowing for flexible and powerful search criteria.
    - If `patterns` is provided as a Path object, it is converted to a string representation before being used for matching.
    - The search is performed using a combination of `my_walk` for traversing directories and `my_filter` for applying
      the match criteria, demonstrating the use of helper functions for modular code design.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    # Example of parameter validation
    if res_type not in ['file', 'dir', None]:
        raise ValueError("res_type must be 'file', 'dir', or None")

    # Purpose: find files or dirs which match desired
    if isinstance(patterns, Path):
        patterns = parse_path_str(patterns)
    elif isinstance(patterns, str):
        patterns = [patterns]

    if patterns:
        compiled_patterns = [re.compile(pattern) for pattern in patterns]
        f_filter = lambda x: all(pattern.search(str(x)) for pattern in compiled_patterns)
    else:
        f_filter = lambda x: True  # No-op lambda, always returns True
        yield_first_match = False  # If no patterns, always return all matches

    if yield_first_match or callable(f_filter):
        filesurvey = list(
            my_filter(
                f_filter,
                my_walk(Path(path), res_type, recursive, ignore),
                yield_first_match,
            ),
        )
    else:
        filesurvey = list(my_walk(Path(path), res_type, recursive, ignore))
    
    if yield_shortest_match:
        filesurvey.sort(key=lambda x: x.inode())

    if filesurvey == []:
        return filesurvey
    if hasattr(filesurvey[0], attr):
        if attr.lower() == "path":
            return [Path(str(f)) for f in filesurvey]
        return [getattr(f, attr) for f in filesurvey]
    if hasattr(filesurvey[0].stat(), attr):
        return [getattr(f.stat(), attr) for f in filesurvey]
    return filesurvey

def f_find(path, search=False, res_type="path", re_filter=None):
    path = Path(path)
    if search:
        res = f_find(path.parent)
        return [r for r in res if r.parent == path.parent and r.stem == path.stem][0]

    filesurvey = []
    for row in os.walk(Path(path)):  # Walks through current path
        for filename in row[
            2
        ]:  # row[2] is the file name re.search(r"(.h5|.hdf5)$", str(file))
            full_path: Path = Path(row[0]) / Path(filename)  # row[0]
            if re_filter is None or re.search(re_filter, full_path.name):
                # if re_filter is None or re_filter in full_path.name:
                filesurvey.append(
                    dict(
                        path=full_path,
                        file=filename,
                        date=full_path.stat().st_mtime,
                        size=full_path.stat().st_size,
                    )
                )

    if res_type == "all" or filesurvey == []:
        return filesurvey
    else:
        try:
            return [f[res_type] for f in filesurvey]
        except KeyError:
            return filesurvey

# %%
# my_path = find_path(r"Dropbox*\Work Docs\Data\Analysis\SIMS")

# my_path = find_path3(r"\Work Docs\Data\Analysis\SIMS")
# test = "Documents\Python\impedance_analysis\testing\circuits.ini"
# my_path1 = parse_path_str(Path("D:/Documents/Python/impedance_analysis/testing/circuits.ini"))
# my_path2 = parse_path_str(Path(r"Documents\Python\impedance_analysis\testing"))
# my_path3 = parse_path_str(Path("Python\impedance_analysis\testing"))

# my_path = find_path(r"Analysis/SIMS", base=find_path(r"ASU Dropbox", base="drive"))
# test0  = find_files(Path().cwd().parent, "path", "dir", None,  recursive=True)
# test1  = find_files(Path().cwd().parent, "path", "dir", "eis_analysis",  recursive=True)
# test2  = find_files(Path().cwd().parent, "path", "dir", "eis_analysis", test0[20:], recursive=True)
# test3  = find_files(Path().cwd().parent, "path", "file", r"\d.py$",  recursive=True)
# test4  = find_files(Path().cwd().parent, "path", "file", r"\d.py$",  recursive=True, yield_first_match=True)

# my_path = find_path("Data","Analysis","Auto", base=find_path(r"ASU Dropbox", base="drive"))

my_folder_path = find_path(
        "impedance_analysis", "testing", "Data", base="cwd"
    )

files1 = f_find(my_folder_path / "Raw", re_filter="mfia_rc")
files2 = find_files(my_folder_path / "Raw", patterns="mfia_rc")