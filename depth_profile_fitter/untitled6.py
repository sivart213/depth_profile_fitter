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
Created on Thu Jun 27 11:38:21 2024
"""

# %% Imports
import numpy as np
import pandas as pd

# %% Code


# %% Operations
if __name__ == "__main__":
    from pathlib import Path
    import ctypes
    import itertools
    import os
    import string
    import platform
    
    # def get_available_drives():
    #     if 'Windows' not in platform.system():
    #         return []
    drive_bitmask = ctypes.cdll.kernel32.GetLogicalDrives()
    # res = list(itertools.compress(string.ascii_uppercase,
    #            map(lambda x:ord(x) - ord('0'), bin(drive_bitmask)[:1:-1])))
    res = list(itertools.compress(map(lambda x: '%s:/'%x, string.ascii_uppercase),
               map(lambda x:ord(x) - ord('0'), bin(drive_bitmask)[:1:-1])))