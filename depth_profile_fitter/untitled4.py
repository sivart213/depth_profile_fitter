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
Created on Fri Jun 21 12:35:57 2024
"""

# %% Imports
import numpy as np
import pandas as pd

# %% Code


# %% Operations
if __name__ == "__main__":
    from pathlib import Path
    import os
    
    def get_size(start_path = '.'):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                try:
                    fp = os.path.join(dirpath, f)
                    # skip if it is symbolic link
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
                except FileNotFoundError:
                    continue
    
        return total_size