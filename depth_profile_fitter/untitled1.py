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
Created on Tue Jun 11 17:03:22 2024
"""

# %% Imports
import numpy as np
import pandas as pd
from statistics import mode
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# %% Code

def is_num_array(x):
    x = np.array(x, ndmin=1)
    if len(x) == 0:
        return 0
    if x.shape == (1,):
        return isinstance(x[0], (float, int))+0
    elif len(x.shape) == 1:
        return sum([isinstance(y, (float, int))+0 for y in x])/len(x)
    else:
        return sum([is_num_array(y) for y in x])/len(x)

# %% Operations
if __name__ == "__main__":
    from pathlib import Path
    import io
    # file = Path(r"D:\Online\Dropbox (ASU)\Work Docs\Data\Analysis\SIMS\RICE\220323 RICE Upd\Files\A-E60.TXT")
    # file = Path(r"D:\Online\Dropbox (ASU)\Work Docs\Data\Analysis\IV\DOW1\210427_112121  DOW1-0-1.jvtst")
    file = Path(r"D:\Online\Dropbox (ASU)\Work Docs\Data\Analysis\SIMS\ASU\20201019\EVA_70_r1.dp_asc")
    # file = Path(r"D:\Online\Dropbox (ASU)\Work Docs\Data\Analysis\SIMS\RICE\220429 Pixels\Files\A-80\E-80-A_total.txt")
    
    # Import lines individually
    lines = []
    with open(file) as f:
        line_n = 0
        large_file = False
        for line in f:
            if line_n >= 10000:
                large_file = True
                break
            if not re.fullmatch(r"\W", line):
                lines.append(pd.read_csv(io.StringIO(line), header=None, sep=None, engine="python").infer_objects())
            else:
                lines.append(pd.DataFrame())
            line_n += 1
    # create a list of indixes based on where the length doesn't change
    lines_numeric = [is_num_array(l) for l in lines]
    # lines_bool = np.diff([len(list(f)) for f in lines]) != 0 + np.diff(lines_numeric) != 0
    lines_bool1 = np.diff([len(list(f)) for f in lines]) != 0
    lines_bool2 = np.diff(lines_numeric) != 0
    gr_ind = np.split([n for n in range(len(lines))], np.where(lines_bool1+lines_bool2)[0]+1)

    gr_df = [pd.concat(list(map(lines.__getitem__, d)), ignore_index=True) for d in gr_ind]
    gr_df = [pd.DataFrame(t.dropna(axis=1, how='all').dropna(axis=0, how='all').to_numpy()) for t in gr_df]
    
    ratings = []
    for n, t in enumerate(gr_df):
        num = is_num_array(t)*(1/3)
        amnt = len(t)/len(lines)*(1/3)
        pos = (n+1)/len(gr_df)*(1/3)
        ratings.append(num+amnt+pos)
    
    data_ind = ratings.index(max(ratings))
    
    data_raw = gr_df.pop(data_ind)
    
    if large_file:
        data_raw = pd.read_csv(file, skiprows=gr_ind[data_ind][0], header=None, sep=None, engine='python')

    n = data_ind - 1
    labs = []
    while gr_df[n].shape[1] == data_raw.shape[1] or abs(data_raw.shape[1]/2-gr_df[n].shape[1]) <= 1:
        if gr_df[n].shape[1] == data_raw.shape[1]/2:
            vals = []
            for v in gr_df[n].to_numpy()[0]:
                vals.append(v)
                vals.append(v)
            gr_df[n] = pd.DataFrame(vals).T
        labs = [gr_df.pop(n)] + labs
        n -= 1
    if labs == []:
        auto_lab = ["col_"+str(x+1) for x in range(data_raw.shape[1])]
        xyz_lab = ["x", 'y', 'z']
        if data_raw.shape[1] < 3:
            labels = auto_lab
        else:
            labels = xyz_lab + auto_lab[:-3]
    else:
        labels = pd.concat(labs, ignore_index=True).infer_objects().fillna("").astype(str).apply(lambda x: re.sub(r'^_', '', re.sub(r'_+', '_', x.str.cat(sep="_"))))

    data = pd.DataFrame(data_raw.to_numpy(), columns=labels).infer_objects()
    #%%
 
    header_raw = pd.concat(gr_df, ignore_index=True).infer_objects()
    header_rows = [row[1].dropna().to_numpy().astype(str) for row in header_raw.iterrows()]
    for n, h in enumerate(header_rows):
        df_str = "\t".join(h)
        df_str = re.sub(r"^[\W_]+\t*", "", df_str)
        sep_list = re.findall(r"\t\W\t", df_str)
        if len(sep_list) > 1 and len(sep_list) > len(np.unique(sep_list)):
            # header_rows[n] = np.array([re.sub(r"\\t", " ", x) for x in re.split(r"\\t\W\\t", df_str)])
            df_str = ", ".join([re.sub(r"\t", " ", x) for x in re.split(r"\t\W\t", df_str)])
        # if re.search(":", df_str):
        #     df_str = re.sub(":", "\t", re.sub(r"\s*\t\s*", " ", df_str), count=1)
        header_rows[n] = np.array(re.split(r"\t", df_str))
            
    header = [" ".join(row) for row in header_rows]
    # header = pd.DataFrame(header_rows)
    # header = pd.DataFrame(header.iloc[:,1:].T.to_numpy(), columns=header.iloc[:,0].to_numpy())
    

