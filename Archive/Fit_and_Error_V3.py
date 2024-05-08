# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 12:36:55 2021

@author: j2cle
"""

#%%
import numpy as np
import pandas as pd
import General_functions as gf
import matplotlib.pyplot as plt

from scipy.special import erfc
from scipy import stats
from functools import partial
import lmfit
from lmfit import Model
from scipy.stats.distributions import  t
import uncertainties as unc
import uncertainties.unumpy as unp

from scipy.optimize import curve_fit
from statsmodels.stats.weightstats import DescrStatsW
from scipy.signal import savgol_filter
import warnings
from sklearn.metrics import r2_score,mean_squared_log_error, mean_absolute_percentage_error
warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings("ignore")



#%%
def c_np(depth,diff,c_0,thick,temp,e_app,time):
    if diff <0:
        diff=10**diff
        c_0=10**c_0
    mob = diff/(gf.KB_EV*temp)
    term_B = erfc(-mob*e_app*time/(2*np.sqrt(diff*time)))
    return (c_0/(2*term_B)) * (erfc((depth - mob*e_app * time)/(2*np.sqrt(diff*time))) + erfc(-(depth-2*thick+mob*e_app*time)/(2*np.sqrt(diff*time))))

def c_np_array(depth,diff,c_0,thick,temp,e_app,time):
    mob = diff/(gf.KB_EV*temp)
    term_B = erfc(-mob*e_app*time/(2*np.sqrt(diff*time)))
    return (c_0/(2*term_B)) * (erfc((depth - mob*e_app * time)/(2*np.sqrt(diff*time))) + erfc(-(depth-2*thick+mob*e_app*time)/(2*np.sqrt(diff*time))))

def depth_conv(data_in,unit,layer_act,layer_meas):
    data_out=data_in
    if unit != 's':
        if not pd.isnull(layer_meas):
            data_out[data_in<layer_meas] = data_in[data_in<layer_meas]*layer_act/layer_meas
            data_out[data_in>=layer_meas] = (data_in[data_in>=layer_meas]-layer_meas)*((max(data_in)-layer_act)/(max(data_in)-layer_meas))+layer_act
                
            # if col(A)< a2 ? col(A)*a1/a2 : (col(A)-a2)*((max(A)-a1)/(max(A)-a2))+a1
        if unit != 'cm':
            data_out=gf.tocm(data_in,unit)
    
    return data_out


def lin_test(x,y,lim=0.025):
    line_info=np.array([np.polyfit(x[-n:],y[-n:],1) for n in range(1,len(x))])
    delta=np.diff(line_info[int(.1*len(x)):,0])
    delta=delta/max(abs(delta))
    bounds=np.where(delta<-lim)[0]
    if bounds[0]+len(bounds)-1 == bounds[-1]:
        bound = len(x)-bounds[0]
    else:
        bound = len(x)-[bounds[n] for n in range(1,len(bounds)) if bounds[n-1]+1!=bounds[n]][-1]
    return bound,x[bound]


def num_diff(x,y,order=1,ave=0,poly=3,trim=True):

    for i in range(order):
        if trim:
            m=1
            while abs(y[m-1])<abs(y[m]):
                y[:m]=y[m]
                m+=1
        if ave != 0:
            y = moving_average(y, ave)
            # x,y = smooth(x,y,ave,poly)
            
        dy = np.zeros(y.shape,float)
        dy[0:-1] = np.diff(y)/np.diff(x)
        dy[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
        y = dy
    return x,y

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def smooth(x,y,box,order=3):
    xy = savgol_filter((x,y),box,order)
    return xy[0,:].T,xy[1,:].T

    
#%%
class Profile:
    def __init__(self,slog):
        self.sample=slog['Sample']
        self.measurement=slog['Measurement']
        self.mtype=slog['Type']
        self.thick=slog['Thick']
        self.thick_unit=slog['Thick unit']
        self.thick_cm = gf.tocm(self.thick,self.thick_unit)
        self.a_layer=slog['Layer (actual)']
        self.a_layer_unit=slog['A-Layer unit']
        self.temp=slog['Temp']
        self.volt=slog['Volt']
        self.e_field=slog['E-field']
        self.stress_time=slog['Stress Time']
        self.fit_depth=slog['Fit depth/limit']
        self.fit_depth_unit=slog['Fit Dep unit']
        self.p_layer=slog['Layer (profile)']
        self.p_layer_unit=slog['P-Layer unit']
        self.res=slog['Resolution']
        self.res_unit=slog['Resolution unit']
        self.matrix=slog['Matrix']
        self.RSF=slog['RSF']
        self.SF=slog['SF']
        self.max_x=slog['Max X']
        self.x_unit=slog['X unit']
        self.y_unit=slog['Y unit']
        self.file_loc=slog['Data File Location']
        self.tab=slog['Tab']
        self.columns=slog['Columns']
        self.comments=slog['Comments']
        
        self.Data()
        
        
        # self.deriv1_x, self.deriv1_y = num_diff(self.data['Depth'].to_numpy(),self.data['Na'].to_numpy(),1,15,3,True)
        # self.deriv2_x, self.deriv2_y = num_diff(self.data['Depth'].to_numpy(),self.data['Na'].to_numpy(),1,4,3,True)
        
        if ~np.isnan(self.a_layer):
            self.a_layer_cm = gf.tocm(self.a_layer,self.a_layer_unit)
        else:
            self.a_layer_cm = 0
            
        if ~np.isnan(self.fit_depth):
            self.fit_depth_cm = gf.tocm(self.fit_depth,self.fit_depth_unit)
        else:
            self.fit_depth=lin_test(self.data['Depth'].to_numpy(),self.data['Na'].to_numpy(),0.05)[1]
            self.fit_depth_unit='cm'
            
        if ~np.isnan(self.p_layer):
            self.p_layer_cm = gf.tocm(self.p_layer,self.p_layer_unit)    
        
        self.Lim()
        # self.lim_loc = lin_test(self.data['Depth'].to_numpy(),self.data['Na'].to_numpy(),0.05)[0]
        
        
    
    def Data(self):
        if self.mtype == 'NREL MIMS':
            self.data_raw = pd.read_excel(self.file_loc, sheet_name=self.tab,usecols=self.columns).dropna()
        elif 'TOF' in self.mtype:
            self.header_in=pd.read_csv(self.file_loc,delimiter='\t',header=None,skiprows=2,nrows=3).dropna(axis=1,how='all')
            self.header_in=self.header_in.fillna(method='ffill',axis=1)
            self.headers = [self.header_in.iloc[0,x] +' '+ self.header_in.iloc[2,x] for x in range(self.header_in.shape[1])]
            self.data_raw = pd.read_csv(self.file_loc,delimiter='\t',header=None,names=self.headers,index_col=False,skiprows=5)
        
        self.data = pd.DataFrame(np.ones((len(self.data_raw),2)), columns=['Depth','Na'])
        self.data_cols=list(self.data_raw.columns)

        for col in self.data_cols:
            if self.mtype == 'NREL MIMS':
                if 'x' in col.lower() or 'depth' in col.lower():
                    self.data['Depth'] = depth_conv(self.data_raw[col].to_numpy(copy=True),self.x_unit,self.a_layer,self.p_layer)
                if 'na' in col.lower():
                    self.na_col = col
                if self.matrix in col:
                    self.data_matrix = self.data_raw[col].to_numpy()
            elif self.mtype == 'TOF':
                if 'x' in col.lower() or 'depth' in col.lower():
                    self.data['Depth'] = gf.tocm(self.data_raw[col].to_numpy(copy=True),self.x_unit)
                if 'na+' in col.lower() and 'conc' in col.lower():
                    self.na_col = col
                if self.matrix in col and 'inten' in col.lower():
                    self.data_matrix = self.data_raw[col].to_numpy()
            elif self.mtype == 'TOF Local':
                if 'x' in col.lower() or 'depth' in col.lower():
                    self.data['Depth'] = gf.tocm(self.data_raw[col].to_numpy(copy=True),self.x_unit)
                if 'na+' in col.lower() and 'inten' in col.lower():
                    self.na_col = col
                if self.matrix in col and 'inten' in col.lower():
                    self.data_matrix = self.data_raw[col].to_numpy()

        if 'counts' in self.y_unit and ~np.isnan(self.RSF):
            self.data['Na'] = self.data_raw[self.na_col].to_numpy()/np.mean(self.data_matrix)*self.RSF
        elif 'counts' in self.y_unit and ~np.isnan(self.SF):
            self.data['Na'] = self.data_raw[self.na_col].to_numpy()*self.SF
        else:
            self.data['Na'] = self.data_raw[self.na_col].to_numpy()
            
        
    def Lim(self,thresh=0.025):
        self.lin_loc, self.lin_lim = lin_test(self.data['Depth'].to_numpy(),self.data['Na'].to_numpy(),thresh)
        if self.lin_lim > self.fit_depth_cm*1.1 or self.lin_lim < self.fit_depth_cm*0.9:
            self.lim_loc = gf.find_nearest(self.data['Depth'].to_numpy(),self.fit_depth_cm)
        else:
            self.lim_loc = self.lin_loc
    
    def err_full(self,size=100):
        
        self.array_size=size
        
        self.C0_range=np.logspace(15,21,self.array_size) # init was  14 to 22
        self.D_range=np.logspace(-17,-11,self.array_size) # init was -18 to -10
        
        C0_3D,D_3D,depth_3D = np.meshgrid(self.C0_range, self.D_range, self.data['Depth'])
        
        # outline is [C0,D,array]
        self.C_pred_matrix = c_np_array(depth=depth_3D, diff=D_3D, c_0=C0_3D, thick=self.thick_cm, temp=gf.CtoK(self.temp), e_app=self.volt/self.thick_cm, time=self.stress_time)
        
        self.profiles_matrix = pd.DataFrame([[self.Sim_Profile(self.data['Na'],self.C_pred_matrix[y,x,:],self.D_range[y],self.C0_range[x],self.data['Depth']) for x in range(self.array_size)] for y in range(self.array_size)], columns=self.C0_range,index=self.D_range)
        
        self.error_matrix_bkg = self.profiles_matrix.applymap(lambda x: x.bkg_error())
        
        # finds the background region --> returns tuple
        self.minima_bkg = np.unravel_index(np.array(self.error_matrix_bkg).argmin(), np.array(self.error_matrix_bkg).shape)
        
        # finds the start of bkgrnd --> returns int
        self.bkg_index = self.profiles_matrix.iloc[self.minima_bkg].x0_index
        
        # forces range on low C0 end and recalculates
        self.profiles_matrix.applymap(lambda x: x.dir_error(self.bkg_index,len(self.data['Na'])) if x.C0 < self.C0_range[self.minima_bkg[1]] else None)
        
        # drop all(?) x0 to 0
        self.profiles_matrix.applymap(lambda x: setattr(x,'x0_index',0))
        
        self.minima_array = np.empty((0,2),int) #ndarray
        iterations = 0
        minima_index = self.minima_bkg # tuple
        x0_index_temp = self.bkg_index
        while x0_index_temp > np.array(self.data['Na']).argmax() and iterations < 30:
            x1_index_temp = x0_index_temp # int: the index point to end on 
            C0_start = minima_index[1]+1 # int: index point of C0 start on

            # run the fitting program for the new range of C0 --> no output
            self.profiles_matrix.applymap(lambda x: x.local_error(x1_index_temp,reset=True) if x.C0 > self.C0_range[minima_index[1]] else None)
            
            # generate a temporary error array --> df matrix
            err_temp = self.profiles_matrix.applymap(lambda x: x.error if x.C0 > self.C0_range[minima_index[1]] else 1)
            
            # find indexes (D&C0) of minimum value in range just tested --> tuple
            minima_index = np.unravel_index(np.array(err_temp).argmin(), np.array(err_temp).shape)
            
            # revert index to compensate for not reviewing lower C0.
            # minima_index = tuple((minima_index[0],(minima_index[1] + C0_start)))
            
            # get the x0 index of the minima location, only for data location
            x0_index_temp = self.profiles_matrix.iloc[minima_index].x0_index
            
            # store the (D&C0) minimum location --> can be used for any dataframe 
            self.minima_array = np.append(self.minima_array, [np.array(minima_index)],axis=0)
            
            # set x0 back to 0 in upcoming round prior to analysis
            self.profiles_matrix.applymap(lambda x: setattr(x,'x0_index',0) if x.C0 > self.C0_range[minima_index[1]] else None)
            
            iterations += 1
        
        self.error_alt_matrix = self.profiles_matrix.applymap(lambda x: x.error_alt)
        self.error_raw_matrix = self.profiles_matrix.applymap(lambda x: x.error_raw)
        self.error_final = self.profiles_matrix.applymap(lambda x: x.error)
        self.x0_index_matrix = self.profiles_matrix.applymap(lambda x: x.x0_index)
        self.x1_index_matrix = self.profiles_matrix.applymap(lambda x: x.x1_index)
                
        col = ['fit error','real error','D','C0','x0 index','x1 index','range (points)','x0','x1','range (um)','Depth (range)','Na (range)','Na (all)']
       
        self.fit_curves = pd.DataFrame([[self.profiles_matrix.iloc[tuple(self.minima_array[x])].error, self.profiles_matrix.iloc[tuple(self.minima_array[x])].error_alt,
                                        self.profiles_matrix.iloc[tuple(self.minima_array[x])].D, self.profiles_matrix.iloc[tuple(self.minima_array[x])].C0, self.profiles_matrix.iloc[tuple(self.minima_array[x])].x0_index,
                                        self.profiles_matrix.iloc[tuple(self.minima_array[x])].x1_index, self.profiles_matrix.iloc[tuple(self.minima_array[x])].index_range, self.profiles_matrix.iloc[tuple(self.minima_array[x])].x0_loc,
                                        self.profiles_matrix.iloc[tuple(self.minima_array[x])].x1_loc, self.profiles_matrix.iloc[tuple(self.minima_array[x])].depth_range, self.profiles_matrix.iloc[tuple(self.minima_array[x])].limited_depth,
                                        self.profiles_matrix.iloc[tuple(self.minima_array[x])].limited_conc, self.profiles_matrix.iloc[tuple(self.minima_array[x])].full_conc] for x in range(np.size(self.minima_array,axis=0))],columns = col) 
        

    
    class Sim_Profile:
        def __init__(self,true_in,pred_in,D_in,C0_in,x_in):
            
            # constant once set 
            self.D = D_in
            self.C0 = C0_in
            self.C_pred = np.log10(1+pred_in)
            self.C_true = np.log10(np.array(1+true_in))
            self.weights = np.array([(10**self.C_pred[x])/(10**self.C_true[x]) if self.C_pred[x]>self.C_true[x] else 1 for x in range(len(self.C_true))])
            self.error_raw = mean_absolute_percentage_error(self.C_true,self.C_pred,sample_weight=self.weights)
            self.x_array = x_in
            
            # changes throughout code
            self.x0_index = 0
            self.x1_index = len(self.C_true)
            self.error = 1/100

        @property
        def x0_index(self):
            return self._x0_index
        @x0_index.setter
        def x0_index(self,value):
            self._x0_index = value
            if hasattr('Sim_Profile','_x1_index'):
                self._error_alt = mean_absolute_percentage_error(self.C_true[self._x0_index:self.x1_index],self.C_pred[self._x0_index:self.x1_index],sample_weight=self.weights[self._x0_index:self.x1_index])
        
        @property
        def x1_index(self):
            return self._x1_index
        @x1_index.setter
        def x1_index(self,value):
            if self.x0_index >= value:
                self.x0_index = 0
            self._x1_index = value
            self._error_alt = mean_absolute_percentage_error(self.C_true[self.x0_index:self._x1_index],self.C_pred[self.x0_index:self._x1_index],sample_weight=self.weights[self.x0_index:self._x1_index])

        @property
        def error(self):
            return self._error
        @error.setter
        def error(self,value):
            self._error = value
            self._error_alt = mean_absolute_percentage_error(self.C_true[self.x0_index:self.x1_index],self.C_pred[self.x0_index:self.x1_index],sample_weight=self.weights[self.x0_index:self.x1_index])
        
        @property
        def error_alt(self):
            return self._error_alt
        @error_alt.setter
        def error_alt(self,value):
            self._error_alt = mean_absolute_percentage_error(self.C_true[self.x0_index:self.x1_index],self.C_pred[self.x0_index:self.x1_index],sample_weight=self.weights[self.x0_index:self.x1_index])


        @property
        def x0_loc(self):
            return gf.fromcm(self.x_array[self.x0_index],'um')
        @property
        def x1_loc(self):
            return gf.fromcm(self.x_array[self.x1_index],'um')
        @property
        def limited_depth(self):
            return self.x_array[self.x0_index:self.x1_index]
        @property
        def limited_conc(self):
            return 10**self.C_pred[self.x0_index:self.x1_index]
        @property
        def full_conc(self):
            return 10**self.C_pred
        @property
        def index_range(self):
            return self.x1_index-self.x0_index
        @property
        def depth_range(self):
            return self.x1_loc-self.x0_loc
        
        def bkg_error(self):
            # self.error = self.error_raw
            # err_last = self.error_raw*1/100
            err_last = self.error
            
            err_array = np.array([mean_absolute_percentage_error(self.C_true[start:],self.C_pred[start:],sample_weight=self.weights[start:])*1/(100*len(self.C_true[start:][(self.C_true>self.C_pred)[start:]])/len(self.C_true)) for start in range(len(self.C_true)-1)])

            if np.min(err_array) < err_last:
                self.x0_index = np.argmin(err_array)
                err_last = self.error
                self.error = np.min(err_array)
                
            return self.error
        
        def local_error(self,x_max=0,reset=True):
            if reset:
                self.error = 1/100
            
            err_last = self.error
            
            if x_max != 0:
                self.x1_index=x_max
            else:
                x_max = len(self.C_true)
                
            err_array = np.array([mean_absolute_percentage_error(self.C_true[start:x_max],self.C_pred[start:x_max],sample_weight=self.weights[start:x_max])*1/(100*len(self.C_true[start:x_max][(self.C_true>self.C_pred)[start:x_max]])/x_max) for start in range(x_max-1)])
                
            if np.min(err_array) < err_last:
                self.x0_index = np.argmin(err_array)
                err_last = self.error
                self.error = np.min(err_array)
        
        def dir_error(self,x_min,x_max):
            return mean_absolute_percentage_error(self.C_true[x_min:x_max],self.C_pred[x_min:x_max],sample_weight=self.weights[x_min:x_max])*1/(100*(x_max-x_min)/len(self.C_true))
        
        def dir_x0(self,x_min):
            if self.x0_index > x_min:
                self.x0_index = x_min
            

    
    def Plot_fit_Eval(self,D_in,C0_in,is_index=True):
        
        if is_index:
            D_ind=D_in
            C0_ind=C0_in
        else:
            D_ind=gf.find_nearest(self.D_range,D_in)
            C0_ind=gf.find_nearest(self.C0_range,C0_in)
        
        plt.figure(self.sample)
        plt.plot(gf.fromcm(self.data['Depth'].to_numpy(),'um'),self.data['Na'].to_numpy(),gf.fromcm(self.data['Depth'].to_numpy(),'um'),self.C_pred_matrix[D_ind,C0_ind])
        plt.xlabel('Depth')
        plt.ylabel('Conc')
        plt.yscale('log')
        plt.ylim(1e14,1e22)
        plt.xlim(0,8)
        plt.grid()
        plt.title(self.sample)
        plt.show()
        

#%%

mypath = "C:\\Users\\j2cle\\Work Docs\\Data\\Analysis\\SIMS\\"
figpath = 'C:\\Users\\j2cle\\Work Docs\\Data\\Analysis\\SIMS\\Fig_fits'
df_log = pd.read_excel(f'{mypath}/Sample Log for fitting.xlsx',index_col=0,skiprows=1).dropna(axis=0,how='all')


#%%  all analysis
if 0:
    df_log.drop(['R-60 SF ROI'],inplace=True)
    
sample_all={x: Profile(df_log.loc[x,:]) for x in df_log.index}

#%%
columns = ['Sample','Measurement','Type','Temp','D','D err','C0','C0 err','MSLE']
df_fit_res = pd.DataFrame(data=None,columns=columns)
#%%
sample=sample_all[df_log.index[18]]

sample.err_full(size=50)

#%%
err1=np.array(sample.error_final)
err2=np.array(sample.error_alt_matrix)
err3=np.array(sample.error_raw_matrix)
x0_mat=np.array(sample.x0_index_matrix)
x1_mat=np.array(sample.x1_index_matrix)


#%%
fit_res = sample.fit_curves

profiles = [x for x in fit_res['Na (all)']]

gf.Log_Map(sample.C0_range,sample.D_range,err1,name='final',xname='C0',yname='D',zname='MAPE',logs='both',levels=50,z_limit=[1e-5,1])
gf.Log_Map(sample.C0_range,sample.D_range,err2,name='re-err',xname='C0',yname='D',zname='MAPE',logs='both',levels=50,z_limit=[1e-5,1])

#%%

index_ranges = [x for x in fit_res[['x0 index','x1 index','range (points)']].values]
err_anal_range1 = np.array([(x,index_ranges[2][1]) for x in range(0,index_ranges[2][1])])
err_anal_range2 = np.array([(index_ranges[2][0],x) for x in range(index_ranges[2][0]+1,sample.bkg_index)])
err_anal_range3 = np.array([(x,x+index_ranges[2][2]) for x in range(sample.bkg_index) if (x+index_ranges[2][2]) < sample.bkg_index])

# err_anal1 = np.array([[]for y in range()])

# gf.Log_Map(sample.C0_range,sample.D_range,err_bkg,name='background',xname='C0',yname='D',zname='MAPE',logs='both',levels=50,z_limit=[1e-5,1])
# # gf.Log_Map(sample.C0_range,sample.D_range,err_perc,name='perc before',xname='C0',yname='D',zname='MAPE',logs='both',levels=50,z_limit=[1e-5,1])

