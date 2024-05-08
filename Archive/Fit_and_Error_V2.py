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
        self.C_pred_array = c_np_array(depth=depth_3D, diff=D_3D, c_0=C0_3D, thick=self.thick_cm, temp=gf.CtoK(self.temp), e_app=self.volt/self.thick_cm, time=self.stress_time)
        
        self.error_dicts = {(self.D_range[y],self.C0_range[x]) : self.Sim_Profile(self.data['Na'],self.C_pred_array[y,x,:],self.D_range[y],self.C0_range[x]) for x in range(self.array_size) for y in range(self.array_size)}
        
        # self.error_local = np.array([[self.error_dicts[(self.D_range[y],self.C0_range[x])].local_error() for x in range(self.array_size)] for y in range(self.array_size)])
        self.error_bkg = np.array([[self.error_dicts[(self.D_range[y],self.C0_range[x])].bkg_error() for x in range(self.array_size)] for y in range(self.array_size)])
        test=self.error_bkg
        # finds the 
        self.bkg_min = np.unravel_index(self.error_bkg.argmin(), self.error_bkg.shape)
        
        # finds the start of bkgrnd
        x0_temp = np.array([[self.error_dicts[(self.D_range[y],self.C0_range[x])].x0 for x in range(self.array_size)] for y in range(self.array_size)])[self.bkg_min]
        
        np.array([[self.error_dicts[(self.D_range[y],self.C0_range[x])].dir_x0(0) for x in range(self.array_size)] for y in range(self.array_size)])
        
        # forces range on low C0 end
        np.array([[self.error_dicts[(self.D_range[y],self.C0_range[x])].dir_error(x0_temp,len(self.data['Na'])) for x in range(len(self.C0_range[:self.bkg_min[1]]))] for y in range(self.array_size)])
        
        self.minima_array = []
        iterations = 0
        min_ind = self.bkg_min
        while x0_temp > np.array(self.data['Na']).argmax() and iterations < 30:
            x1_temp = x0_temp
            C0_start = min_ind[1]+1
            
            np.array([[self.error_dicts[(self.D_range[y],self.C0_range[x])].local_error(x1_temp,reset=True) for x in range(C0_start,self.array_size)] for y in range(self.array_size)])
            err_temp = np.array([[self.error_dicts[(self.D_range[y],self.C0_range[x])].error for x in range(self.array_size)] for y in range(self.array_size)])
            
            # find indexes (D&C0) of minimum value in range just tested
            min_ind = np.unravel_index(err_temp[:,C0_start:].argmin(), err_temp[:,C0_start:].shape)
            # revert index to compensate for not reviewing lower C0.
            min_ind = tuple((min_ind[0],(min_ind[1] + C0_start)))
            x0_temp = np.array([[self.error_dicts[(self.D_range[y],self.C0_range[x])].x0 for x in range(self.array_size)] for y in range(self.array_size)])[min_ind]
            self.minima_array = np.append(self.minima_array, min_ind)
            
            iterations += 1
        
        self.error_raw = np.array([[self.error_dicts[(self.D_range[y],self.C0_range[x])].error_raw for x in range(self.array_size)] for y in range(self.array_size)])
        self.error_final = np.array([[self.error_dicts[(self.D_range[y],self.C0_range[x])].error for x in range(self.array_size)] for y in range(self.array_size)])

        # self.error_array2 = np.array([[self.error_dicts[(self.D_range[y],self.C0_range[x])].error_raw2 for x in range(self.array_size)] for y in range(self.array_size)])
        # self.error_perc_array = np.array([[self.error_dicts[(self.D_range[y],self.C0_range[x])].error_perc for x in range(self.array_size)] for y in range(self.array_size)])
        
        # self.x0_array = np.array([[self.error_dicts[(self.D_range[y],self.C0_range[x])].x0 for x in range(self.array_size)] for y in range(self.array_size)])
        # self.x1_array = np.array([[self.error_dicts[(self.D_range[y],self.C0_range[x])].x1 for x in range(self.array_size)] for y in range(self.array_size)])
        # self.point_range = self.x1_array-self.x0_array
        
        
        # self.error_array_legacy = np.array([[mean_absolute_percentage_error(self.data['Na']**2,self.C_pred_array[y,x,:]**2) for x in range(self.array_size)]for y in range(self.array_size)])
        
        
        # for D in range(self.array_size):
        #     for C0 in range(self.array_size):
        #         for x1 in range(len(self.data['Depth'])):
        #             if change_start==0:
        
    class Sim_Profile:
        def __init__(self,true_in,pred_in,D_in,C0_in):
        
            self.D = D_in
            self.C0 = C0_in
            self.C_pred = np.log10(1+pred_in)
            self.C_true = np.log10(np.array(1+true_in))
            self.weights = np.array([(10**self.C_pred[x])/(10**self.C_true[x]) if self.C_pred[x]>self.C_true[x] else 1 for x in range(len(self.C_true))])
            # self.weights = np.ones_like(self.C_pred)
            self.error_raw = mean_absolute_percentage_error(self.C_true,self.C_pred,sample_weight=self.weights)
            # self.error = self.error_raw
            self.error = 1/100
            # self.error_raw2 = mean_absolute_percentage_error(self.C_true,self.C_pred)
            
            self.x0 = 0
            self.x0_bkg = 0
            self.x1 = len(self.C_true)
        
        def bkg_error(self):
            # self.error = self.error_raw
            # err_last = self.error_raw*1/100
            err_last = self.error
            
            err_array = np.array([mean_absolute_percentage_error(self.C_true[start:],self.C_pred[start:],sample_weight=self.weights[start:])*1/(100*len(self.C_true[start:][(self.C_true>self.C_pred)[start:]])/len(self.C_true)) for start in range(len(self.C_true)-1)])

            if np.min(err_array) < err_last:
                self.x0 = np.argmin(err_array)
                err_last = self.error
                self.error = np.min(err_array)
                
            return self.error
        
        def local_error(self,x_max=0,reset=False):
            if reset:
                self.error = 1/100
            err_last = self.error
            
            if x_max != 0:
                self.x1=x_max
            else:
                x_max = len(self.C_true)
                
            err_array = np.array([mean_absolute_percentage_error(self.C_true[start:x_max],self.C_pred[start:x_max],sample_weight=self.weights[start:x_max])*1/(100*len(self.C_true[start:x_max][(self.C_true>self.C_pred)[start:x_max]])/x_max) for start in range(x_max-1)])
                
            if np.min(err_array) < err_last:
                self.x0 = np.argmin(err_array)
                err_last = self.error
                self.error = np.min(err_array)
        
        def dir_error(self,x_min,x_max):
            self.x0 = x_min
            self.x1 = x_max
            self.error = mean_absolute_percentage_error(self.C_true[x_min:x_max],self.C_pred[x_min:x_max],sample_weight=self.weights[x_min:x_max])*1/(100*(x_max-x_min)/len(self.C_true))
        
        def dir_x0(self,x_min):
            if self.x0 > x_min:
                self.x0 = x_min
            
            
            # starts = np.array(range(0,len(self.C_true)-1))
            # ends = np.array(range(1,len(self.C_true)))
            # # starts_2D,ends_2D = np.meshgrid(starts, ends)
            # # errors = np.ones((len(starts),len(starts)))*self.error_raw
            # errors = np.array([[mean_absolute_percentage_error(self.C_true[x:y]**2,self.C_pred[x:y]**2) if x < y else self.error_raw for x in starts] for y in ends])
            # self.x0,self.x1 = (np.unravel_index(errors.argmin(), errors.shape))
            # self.x1 += 1
            # self.error = np.min(errors)
            
            

            
            # inc=int(len(self.C_true)/8)
            # for start in range(0,len(self.C_true)-2,inc):
            #     err_array = np.array([mean_absolute_percentage_error(self.C_true[start:end],self.C_pred[start:end],sample_weight=self.weights[start:end]) for end in range(start+2,len(self.C_true))])
                
            #     if np.min(err_array) < err_last:
            #         self.x0 = start
            #         self.x1 = np.argmin(err_array)+start+2
            #         err_last = self.error
            #         self.error = np.min(err_array)
                    
            # bot = self.x0-inc if self.x0-inc>0 else 0
            # top = self.x0+inc if self.x0+inc<len(self.C_true)-2 else len(self.C_true)-2
            
            # inc=int((top-bot)/10)
            # for start in range(bot,top,inc):
            #     err_array = np.array([mean_absolute_percentage_error(self.C_true[start:end],self.C_pred[start:end],sample_weight=self.weights[start:end]) for end in range(start+2,len(self.C_true))])
                
            #     if np.min(err_array) < err_last:
            #         self.x0 = start
            #         self.x1 = np.argmin(err_array)+start+2
            #         err_last = self.error
            #         self.error = np.min(err_array)
                    
            # bot = self.x0-inc if self.x0-inc>0 else 0
            # top = self.x0+inc if self.x0+inc<len(self.C_true)-2 else len(self.C_true)-2
            
            # for start in range(bot,top):
            #     err_array = np.array([mean_absolute_percentage_error(self.C_true[start:end],self.C_pred[start:end],sample_weight=self.weights[start:end]) for end in range(start+2,len(self.C_true))])
                
            #     if np.min(err_array) < err_last:
            #         self.x0 = start
            #         self.x1 = np.argmin(err_array)+start+2
            #         err_last = self.error
            #         self.error = np.min(err_array)
                    
            # return self.error
            
            
            # while err_last > self.error and self.x0 < len(self.C_true):
            #     self.x0 += 1
            #     err_last = self.error
            #     self.error = mean_absolute_percentage_error(self.C_true[self.x0:]**2,self.C_pred[self.x0:]**2)
            # self.x0 -= 1
            # if err_last < self.error:
            #     self.error = err_last
            # err_last = self.error*10
            # while err_last > self.error and self.x1 > 0:
            #     self.x1 -= 1
            #     err_last = self.error
            #     self.error = mean_absolute_percentage_error(self.C_true[:self.x1]**2,self.C_pred[:self.x1]**2)
            # self.x1 += 1
            # if err_last < self.error:
            #     self.error = err_last
            # return self.error
        
        

    
    def Plot_fit_Eval(self,D_in,C0_in,is_index=True):
        
        if is_index:
            D_ind=D_in
            C0_ind=C0_in
        else:
            D_ind=gf.find_nearest(self.D_range,D_in)
            C0_ind=gf.find_nearest(self.C0_range,C0_in)
        
        plt.figure(self.sample)
        plt.plot(gf.fromcm(self.data['Depth'].to_numpy(),'um'),self.data['Na'].to_numpy(),gf.fromcm(self.data['Depth'].to_numpy(),'um'),self.C_pred_array[D_ind,C0_ind])
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
test=sample.error_final
# #%%
# err_full = sample.error_array
# # err_full2 = sample.error_array2
# # err_loc = sample.error_local
# err_bkg = sample.error_bkg
# # err_perc = sample.error_perc_array

x0_array = np.array([[sample.error_dicts[(sample.D_range[y],sample.C0_range[x])].x0 for x in range(sample.array_size)] for y in range(sample.array_size)])
# x0_alt_array = np.array([[sample.error_dicts[(sample.D_range[y],sample.C0_range[x])].x0_bkg for x in range(sample.array_size)] for y in range(sample.array_size)])
x1_array = np.array([[sample.error_dicts[(sample.D_range[y],sample.C0_range[x])].x1 for x in range(sample.array_size)] for y in range(sample.array_size)])
point_range = x1_array-x0_array


#%%
# gf.Log_Map(sample.C0_range,sample.D_range,err_full,name='full',xname='C0',yname='D',zname='MAPE',logs='both',levels=50,z_limit=[1e-5,1])
# gf.Log_Map(sample.C0_range,sample.D_range,err_bkg,name='background',xname='C0',yname='D',zname='MAPE',logs='both',levels=50,z_limit=[1e-5,1])
# # gf.Log_Map(sample.C0_range,sample.D_range,err_perc,name='perc before',xname='C0',yname='D',zname='MAPE',logs='both',levels=50,z_limit=[1e-5,1])



# #%%
# for sample_name in df_log.index:
    
#     sample = sample_all[sample_name]
    
#     sample.Points(limited=True)
#     sample.Fit(limited=True)
    
#     df_fit_res=df_fit_res.append(sample.df_fit_res)
#     df_fit_res.loc[sample.df_fit_res.index,'Sample']=str(df_log.loc[sample_name,'Sample'])
#     df_fit_res.loc[sample.df_fit_res.index,'Measurement']=str(df_log.loc[sample_name,'Measurement'])
#     df_fit_res.loc[sample.df_fit_res.index,'Temp']=int(df_log.loc[sample_name,'Temp'])
#     df_fit_res.loc[sample.df_fit_res.index,'Type']=str(df_log.loc[sample_name,'Type'])
    
# #%%
# df_fit_res.boxplot(column='D',by='Temp')
# df_fit_res.boxplot(column='D',by=['Type','Temp'])
# df_fit_res.plot.scatter(x='Sample',y='D')


# #%%
# for sample_name in df_log.index:
    
#     sample = sample_all[sample_name]
    
#     plt.figure(sample_name)
#     plt.plot(gf.fromcm(sample.data['Depth'],'um'),sample.data['Na'])
#     for plots in range(len(sample.df_fit_res)):
#         plt.plot(gf.fromcm(sample.data['Depth'],'um'),sample.c_np_new(sample.data['Depth'],sample.df_fit_res.iloc[plots,0],sample.df_fit_res.iloc[plots,2]))
    
#     plt.xlabel('Depth [$\mu$m]')
#     plt.ylabel('Concentration [atoms/cm$^{3}$]')
#     plt.yscale('log')
#     plt.ylim(1e14,1e20)
#     plt.xlim(0,8)
#     plt.grid()
#     plt.title(sample_name)
#     newpath = figpath + '\\' + sample_name + '.png'
#     plt.savefig(newpath)
#     plt.close("all")    
# #%%
# arr_ind=df_fit_res.loc[:,:'Temp'].to_numpy()
# arr_ind=np.delete(arr_ind,1,axis=1)
# arr_ind=np.flip(arr_ind,axis=1)
# arr_ind=np.concatenate((arr_ind,np.array([df_fit_res.index]).T),axis=1)

# mindex=pd.MultiIndex.from_arrays(arr_ind.T,names=('Temp','Type','Sample','Run'))

# df_ord_res=pd.DataFrame(df_fit_res.loc[:,'D':].to_numpy(),index=mindex,columns=['D','D err','C0','C0 err','MSLE'])

# # df_Dat60=df_ord_res.loc[:,(('D','D err'),60)]
# # test1=df_ord_res.sort_index()

# df_Dbytemp=df_ord_res.loc[(slice(None),('NREL MIMS','TOF')),('D','D err')]
# # x=df_Dbytemp.index.get_level_values(0).to_numpy()

# #%%
# arrh = lambda x, *p: (p[0]*np.exp(p[1]/(gf.KB_EV*(x+273.15))))
# arrh_log = lambda x, *p: p[0]+(p[1]/(gf.KB_EV*(x+273.15)))

# result=curve_fit(arrh_log,df_Dbytemp.index.get_level_values(0).to_numpy(),np.log(df_Dbytemp['D'].to_numpy()),sigma=(df_Dbytemp['D err'].to_numpy())/(df_Dbytemp['D'].to_numpy()),absolute_sigma=False, p0=(np.log(1e-5),-0.75),bounds=((np.log(1e-20),-10),(np.log(10),10)),method='trf',jac='3-point',xtol=1e-12,ftol=1e-12,loss='soft_l1',x_scale='jac') #xtol=1e-16

# prefac=np.exp(result[0][0])
# prefac_err = np.sqrt(np.diag(result[1]))[0]*prefac
# act_en=result[0][1]
# act_en_err = np.sqrt(np.diag(result[1]))[1]

# if prefac_err>prefac:
#     print('log Fail')

# result2=curve_fit(arrh,df_Dbytemp.index.get_level_values(0).to_numpy(),(df_Dbytemp['D'].to_numpy()),sigma=(df_Dbytemp['D err'].to_numpy()),absolute_sigma=False, p0=(1e-10,-0.1),bounds=((1e-50,-10),(1,0)),method='trf',jac='3-point',xtol=1e-12,ftol=1e-12,loss='soft_l1') #xtol=1e-16

# prefac2=result2[0][0]
# prefac_err2 = np.sqrt(np.diag(result2[1]))[0]
# act_en2=result2[0][1]
# act_en_err2 = np.sqrt(np.diag(result2[1]))[1]

# if prefac_err2>prefac2:
#     print('dir Fail')

# alpha = .05 # 95% confidence interval = 100*(1-alpha)

# n = len(df_Dbytemp['D'])    # number of data points
# p = len(result2[0]) # number of parameters

# dof = max(0, n - p)

# tval = t.ppf(1.0-alpha/2., dof)

# ci_p=prefac_err2*tval
# ci_e=act_en_err2*tval

# pmax=prefac2+ci_p
# pmin=prefac2-ci_p
# actmax=act_en2+ci_e
# actmin=act_en2-ci_e


# #%%




# #%%
# popt, pcov = result

# a, b = unc.correlated_values(popt, pcov)

# px = np.linspace(50, 100, 100)
# # use unumpy.exp
# py = a+(b/(gf.KB_EV*(px+273.15)))

# nom = np.exp(unp.nominal_values(py))
# std = unp.std_devs(py)*nom

# tval = t.ppf(1.0-alpha/2., dof)

# ci=std*tval


# ci_max=nom+ci
# ci_min=nom-ci

# #%%
# popt, pcov = result2

# a, b = unc.correlated_values(popt, pcov)

# px = np.linspace(50, 100, 100)
# # use unumpy.exp
# py = a*unp.exp(b/(gf.KB_EV*(px+273.15)))

# nom = unp.nominal_values(py)
# std = unp.std_devs(py)

# ci=std*tval


# ci_max=nom+ci
# ci_min=nom-ci

# #%%
# popt, pcov = result

# ci = 0.95
# # Convert to percentile point of the normal distribution.
# # See: https://en.wikipedia.org/wiki/Standard_score
# pp = (1. + ci) / 2.
# # Convert to number of standard deviations.
# nstd = stats.norm.ppf(pp)
# perr = np.sqrt(np.diag(pcov))

# popt_up = popt + nstd * perr
# popt_dw = popt - nstd * perr

# px = np.linspace(25, 100, 100)
# # use unumpy.exp
# py_up = np.exp(popt_up[0]+(popt_up[1]/(gf.KB_EV*(px+273.15))))
# py_dw = np.exp(popt_dw[0]+(popt_dw[1]/(gf.KB_EV*(px+273.15))))



# #%%
# popt, pcov = result2

# ci = 0.95
# # Convert to percentile point of the normal distribution.
# # See: https://en.wikipedia.org/wiki/Standard_score
# pp = (1. + ci) / 2.
# # Convert to number of standard deviations.
# nstd = stats.norm.ppf(pp)
# perr = np.sqrt(np.diag(pcov))

# popt_up = popt + nstd * perr
# popt_dw = popt - nstd * perr

# px = np.linspace(25, 100, 100)
# # use unumpy.exp
# py_up = popt_up[0]*np.exp(popt_up[1]/(gf.KB_EV*(px+273.15)))
# py_dw = popt_dw[0]*np.exp(popt_dw[1]/(gf.KB_EV*(px+273.15)))

# #%%

# arrh2 = lambda x, p, e: (p*np.exp(e/(gf.KB_EV*(x+273.15))))
# amodel=Model(arrh2)
# amodel.set_param_hint('e',value=-0.75,min=-10,max=10)
# amodel.set_param_hint('p',value=0.05,min=1e-50)
# # params = amodel.make_params(p=0.05,e=-0.75)
# # params['e'].set(max=10)
# # params['e'].set(min=-10)
# # params['p'].set(max=np.inf)
# # params['p'].set(min=1e-50)

# result3=amodel.fit(df_Dbytemp['D'].to_numpy(),weights=(df_Dbytemp['D err'].to_numpy()), x=df_Dbytemp.index.get_level_values(0).to_numpy(),nan_policy='omit',method='least_squares')
# print(result3.fit_report())


# # #%%
# # columns = ['Type','Temp','D','D err','C0','C0 err','MSLE']

# # df_fit_res = pd.DataFrame(data=None,columns=columns)
# # for sample_name in df_log[cond].index:
    
# #     sample = sample_all[sample_name]
    
# #     if len(df_fit_res)==0:
# #         df_fit_res=sample.df_fit_res
# #     else:
# #         df_fit_res=df_fit_res.append(sample.df_fit_res)
        
        
# # #%%  group analysis

# # # cond1 = df_log['Temp'] == 80
# # # cond = df_log['Type'] == 'NREL MIMS'
# # cond = (df_log['Type'] == 'NREL MIMS') * (df_log['Temp'] == 80)
# # # cond = (df_log['Type'] == 'TOF') * (df_log['Temp'] == 70)
# # # cond = (df_log['Type'] == 'TOF Local') * (df_log['Sample'] == 'R-90 B2') * (df_log['Temp'] == 90)

# # sample_all={x: Profile(df_log.loc[x,:]) for x in df_log[cond].index}

# # columns = ['D','D err','C0','C0 err','MSLE']


# # plots=0
# # df_fit_res = pd.DataFrame(data=None,columns=columns)

# # for sample_name in df_log[cond].index:
    
# #     sample = sample_all[sample_name]
    
# #     sample.Points()
# #     sample.Fit(limited=False)
    
    
# #     if 1:
# #         plt.figure(sample_name)
# #         plt.plot(gf.fromcm(sample.data['Depth'],'um'),sample.data['Na'])
# #         for plots in range(len(sample.df_fit_res)):
# #             plt.plot(gf.fromcm(sample.data['Depth'],'um'),sample.c_np_new(sample.data['Depth'],sample.df_fit_res.iloc[plots,0],sample.df_fit_res.iloc[plots,2]))

    
# #         plt.xlabel('Depth [$\mu$m]')
# #         plt.ylabel('Concentration [atoms/cm$^{3}$]')
# #         plt.yscale('log')
# #         plt.ylim(1e14,1e20)
# #         plt.xlim(0,8)
# #         plt.grid()
# #         plt.title(sample_name)
# #         # newpath = figpath + '\\' + sample_name + '.png'
# #         # plt.savefig(newpath)
# #         # plt.close("all")
    
# #     if len(df_fit_res)==0:
# #         df_fit_res=sample.df_fit_res
# #     else:
# #         df_fit_res=df_fit_res.append(sample.df_fit_res)

# # # stats = DescrStatsW(df_fit_res['D'], weights=np.ceil(np.max(df_fit_res['MSLE']))-df_fit_res['MSLE'], ddof=0)
# # cstats=DescrStatsW((df_fit_res['C0']), weights=(np.ceil(np.max(df_fit_res['MSLE']))-df_fit_res['MSLE'])*np.min(df_fit_res['C0'])/df_fit_res['C0'], ddof=0)
# # dstats=DescrStatsW((df_fit_res['D']), weights=(np.ceil(np.max(df_fit_res['MSLE']))-df_fit_res['MSLE'])*np.min(df_fit_res['D'])/df_fit_res['D'], ddof=0)
# # # dstats=DescrStatsW(np.log10(df_fit_res['D']), weights=[0.001,0.001,1], ddof=0)
# # print(sample_name)
# # print(cstats.mean,cstats.std)
# # print(dstats.mean,dstats.std)



        
        
#     # def Points(self,limited=True,thresh=0.25,dist_perc=.33,dist_force=False):
        
#     #     range_perc = gf.tocm(self.fit_depth,self.fit_depth_unit)*dist_perc
#     #     resolution_in = gf.tocm(self.res,self.res_unit)
        
#     #     # Sets how big or small range is to be
#     #     if range_perc <= resolution_in:
#     #         self.poor_res = True
#     #     else:
#     #         self.poor_res = False
#     #     if range_perc*0.1 >= resolution_in:
#     #         self.great_res = True
#     #     else:
#     #         self.great_res = False
        
#     #     if limited and range_perc/2 <= resolution_in and range_perc*2 >= resolution_in:
#     #         c_dist=2.5
#     #         dist=resolution_in
            
#     #     # if we want limited and the resolution is higher than 30% then use 30%
#     #     elif limited and (self.poor_res or self.great_res):
#     #         c_dist=2.5
#     #         dist=range_perc
#     #     # if we dont want limited and the resolution is lower than 3%
#     #     elif not limited and self.great_res:
#     #         c_dist=2.5
#     #         dist=resolution_in
#     #     else:
#     #         c_dist=2.5
#     #         dist=gf.Sig_figs(max(np.diff(self.data['Depth'])),'um',1)*c_dist
            
#     #     self.dist_um = gf.fromcm(dist,'um')
        
#     #     pstart = gf.find_nearest(self.data['Depth'].to_numpy(),self.a_layer_cm)
#     #     pend = gf.find_nearest(self.data['Depth'].to_numpy(),self.data.iloc[self.lim_loc,0]-dist)
#     #     prange = gf.find_nearest(self.data['Depth'].to_numpy(),self.data.loc[self.lim_loc,'Depth']+dist)-self.lim_loc
        
#     #     # ps_iter is essentially the current start
#     #     ps_iter=pstart
#     #     # p_iter is essentially the increasing range
#     #     p_iter=0
#     #     self.starts = []
#     #     self.ends = []
#     #     r2testold=thresh
#     #     if self.mtype == 'NREL MIMS':
#     #         neglim=2
#     #     else:
#     #         neglim=1.25
        
#     #     # Searches the range
#     #     while ps_iter < pend:
#     #         # passes in the constants 
#     #         self.c_np_new = partial(c_np,thick=self.thick_cm,temp=gf.CtoK(self.temp),e_app=self.volt/self.thick_cm,time=self.stress_time)
#     #         # Performs a curve fit across the current range set by the start+minimum range+increasing range
#     #         fittemp=curve_fit(self.c_np_new, self.data.loc[ps_iter:ps_iter+p_iter+prange,'Depth'], self.data.loc[ps_iter:ps_iter+p_iter+prange,'Na'],p0=(-14,19),bounds=((-20,15),(-10,21)),x_scale='jac',xtol=1e-12,jac='3-point')[0]
#     #         # Takes result and generates a simulated cuve
#     #         ### fittemp=curve_fit(self.c_np_new, self.data.loc[ps_iter:ps_iter+p_iter+prange,'Depth'], self.data.loc[ps_iter:ps_iter+p_iter+prange,'Na'],p0=(1e-14,1e19),bounds=((1e-20,1e15),(1e-10,1e21)),x_scale='jac',jac='3-point')[0]
#     #         simcurve = self.c_np_new(self.data['Depth'],fittemp[0],fittemp[1])
#     #         # calculate the MSLE between the two 
#     #         r2test=mean_squared_log_error(simcurve.loc[ps_iter:ps_iter+p_iter+prange],self.data.loc[ps_iter:ps_iter+p_iter+prange,'Na'])
#     #         # begins evaluation, if error less than threshold and best fit*125% and the simulated cuve does not exceed the origional by 25%... simulated-origional should be neg
#     #         # evaluation is good
#     #         if r2test <= thresh and r2test <= r2testold*1.25 and np.all(simcurve-self.data['Na']*neglim<0):
#     #             # save the fit value, increase the width
#     #             r2testold=r2test
#     #             p_iter+=1
#     #             # if the start point isnt saved, save it
#     #             if self.data.iloc[ps_iter,0] not in self.starts:
#     #                 self.starts.append(self.data.iloc[ps_iter,0])
                    
#     #         # evaluation fails
#     #         else:
#     #             # if there's a start, store the end, start searching at that end or 
#     #             if self.data.iloc[ps_iter,0] in self.starts:
#     #                 # save end
#     #                 self.ends.append(self.data.iloc[ps_iter+p_iter+prange,0])
#     #                 # set new start point based on the current end point (current start+min range+increasing range)
#     #                 ps_iter=ps_iter+p_iter+prange
#     #                 # reset range incrimenter
#     #                 p_iter=0
#     #             else:
#     #                 # no current fit so bump start by one data point
#     #                 ps_iter+=1
                    
#     #     # create points 
#     #     self.points=[[self.starts[n],self.ends[n]] for n in range(len(self.starts))]
        
        
#     #     # compensates for low results, adding point ranges in alt regions
#     #     self.was_len=len(self.points)
#     #     if len(self.points)==0:
#     #         self.points.append([self.data.iloc[pstart,0],self.data.iloc[pstart+prange,0]])
#     #         self.points.append([self.data.iloc[int((self.lim_loc-pstart)*1/4+pstart),0],self.data.iloc[int((self.lim_loc-pstart)*1/4+pstart+prange),0]])
#     #         self.points.append([self.data.iloc[int((self.lim_loc-pstart)*2/4+pstart),0],self.data.iloc[int((self.lim_loc-pstart)*2/4+pstart+prange),0]])
#     #         self.points.append([self.data.iloc[self.lim_loc-prange,0],self.data.iloc[self.lim_loc,0]])
#     #         self.starts = list(np.array(self.points)[:,0])
#     #         self.ends = list(np.array(self.points)[:,1])
#     #         self.point_info='Was Null'
#     #     elif len(self.points)==2 and np.max(self.ends) > self.data.iloc[pend,0]:
#     #         self.points.append([np.min(self.ends),np.max(self.starts)])
#     #         self.point_info='Was 2 and added a center range'
#     #     elif len(self.points)==2:
#     #         self.points.append([np.max(self.ends),self.data.iloc[self.lim_loc,0]])  
#     #         self.point_info='Was 2 and added an end point'
#     #     elif len(self.points)==1 and np.min(self.starts) > self.data.iloc[pend-prange,0] and np.max(self.ends) > self.data.iloc[pend,0]:
#     #         self.points.append([self.data.iloc[pstart,0],self.starts[0]*1/2])
#     #         self.points.append([self.starts[0]*1/2,self.starts[0]])
#     #         self.point_info='Was 1 and added points near the start'
#     #     elif len(self.points)==1 and np.min(self.starts) < self.data.iloc[pend-prange,0] and np.max(self.ends) > self.data.iloc[pend,0]:
#     #         self.points.append([self.data.iloc[pstart,0],self.data.iloc[pend-prange,0]])
#     #         self.points.append([self.data.iloc[pend-prange,0],self.ends[0]])
#     #         self.point_info='Was 1 and added points near the middle'
#     #     elif len(self.points)==1 and np.min(self.ends) < self.data.iloc[pend-prange*2,0]:
#     #         self.points.append([self.ends[0],self.data.iloc[pend-prange,0]])
#     #         self.points.append([self.data.iloc[pend-prange,0],self.data.iloc[self.lim_loc,0]])
#     #         self.point_info='Was 1 and added points near the end'
#     #     elif len(self.points)==1:
#     #         self.points.append([np.min((self.data.iloc[pstart,0],self.starts[0])),(np.min((self.data.iloc[pstart,0],self.starts[0]))+dist)])
#     #         self.points.append([np.min((self.data.iloc[pend,0],self.ends[0])),(np.min((self.data.iloc[pend,0],self.ends[0]))+dist)])
            
#     #         self.point_info='Was 1 and added random points'

#     #     if np.min(self.points)>self.data.loc[pstart+prange*3,'Depth']:
#     #         self.points.append([self.data.iloc[pstart+prange,0],self.data.iloc[pstart+prange*2,0]])
    

#     #     if len(self.points) <= 4 and not limited:
#     #         points1=[[self.points[n][0]-dist,self.points[n][1]] for n in range(len(self.points)) if self.points[n][0]-dist > 0]
#     #         points2=[[self.points[n][0]+dist,self.points[n][1]] for n in range(len(self.points)) if self.points[n][0]+dist < self.points[n][1]]
#     #         points3=[[self.points[n][0],self.points[n][1]-dist] for n in range(len(self.points)) if self.points[n][0] < self.points[n][1]-dist]
#     #         points4=[[self.points[n][0],self.points[n][1]+dist] for n in range(len(self.points))]
#     #         self.points=self.points+points1+points2+points3+points4
            
#     #     if len(self.points) <= 5 and not limited:
#     #         points1=[[self.points[n][0]-dist/c_dist,self.points[n][1]] for n in range(len(self.points)) if self.points[n][0]-dist/c_dist > 0]
#     #         points2=[[self.points[n][0]+dist/c_dist,self.points[n][1]] for n in range(len(self.points)) if self.points[n][0]+dist/c_dist < self.points[n][1]]
#     #         points3=[[self.points[n][0],self.points[n][1]-dist/c_dist] for n in range(len(self.points)) if self.points[n][0] < self.points[n][1]-dist/c_dist]
#     #         points4=[[self.points[n][0],self.points[n][1]+dist/c_dist] for n in range(len(self.points))]
#     #         self.points=self.points+points1+points2+points3+points4
        
    
#     # def Fit(self,limited=True,perc=0):
#     #     # generate masks based on point ranges
#     #     self.res_columns = ['D','D err','C0','C0 err','MSLE']
#     #     self.masks_fit = [self.data['Depth'].between(x[0],x[1]) for x in self.points if sum(self.data['Depth'].between(x[0],x[1]))>1]
        
#     #     # backup region generation in case there's a problem with points or masks
#     #     if len(self.masks_fit) == 0:
#     #         self.masks_fit.append(self.data['Depth'].between(self.data.iloc[0,0],self.data.iloc[self.lim_loc,0]*1/3))
#     #         self.masks_fit.append(self.data['Depth'].between(self.data.iloc[self.lim_loc,0]*1/3,self.data.iloc[self.lim_loc,0]*2/3))
#     #         self.masks_fit.append(self.data['Depth'].between(self.data.iloc[self.lim_loc,0]*2/3,self.data.iloc[self.lim_loc,0]))
                                  
#     #     # send constants
#     #     self.c_np_new = partial(c_np,thick=self.thick_cm,temp=gf.CtoK(self.temp),e_app=self.volt/self.thick_cm,time=self.stress_time)
#     #     # uses log of vals to find fits
#     #     self.fits = [curve_fit(self.c_np_new, self.data.loc[mask,'Depth'], self.data.loc[mask,'Na'],p0=(-14,19),bounds=((-20,15),(-10,21)),x_scale='jac',xtol=1e-12,jac='3-point') for mask in self.masks_fit]
#     #     self.fits_reorg = [[10**self.fits[x][0][0],(10**(self.fits[x][0][0]+np.sqrt(np.diag(self.fits[x][1]))[0])-10**(self.fits[x][0][0]-np.sqrt(np.diag(self.fits[x][1]))[0]))/2,
#     #                     10**self.fits[x][0][1],(10**(self.fits[x][0][1]+np.sqrt(np.diag(self.fits[x][1]))[1])-10**(self.fits[x][0][1]-np.sqrt(np.diag(self.fits[x][1]))[1]))/2] for x in range(len(self.masks_fit))]

#     #     # self.fits = [curve_fit(self.c_np_new, self.data.loc[mask,'Depth'], self.data.loc[mask,'Na'],p0=(1e-14,1e19),bounds=((1e-20,1e15),(1e-10,1e21)),x_scale='jac', jac='3-point') for mask in self.masks_fit]
#     #     # self.fits_reorg = [[self.fits[x][0][0], np.sqrt(np.diag(self.fits[x][1]))[0], self.fits[x][0][1],np.sqrt(np.diag(self.fits[x][1]))[1]] for x in range(len(self.masks_fit))]
#     #     # only saves fits which have an actual fit and it the error isnt 5 times the values
#     #     self.df_fit_res = pd.DataFrame([x+[''] for x in self.fits_reorg if x[1] != 0 and x[3] !=0 and x[1] < x[0]*5 ],columns=self.res_columns)
#     #     for index in self.df_fit_res.index:
#     #         if self.df_fit_res.loc[index,'D']-self.df_fit_res.loc[index,'D err'] < 0:
#     #             self.df_fit_res.loc[index,'D err']=self.df_fit_res.loc[index,'D']*.75
#     #         if self.df_fit_res.loc[index,'C0']-self.df_fit_res.loc[index,'C0 err'] < 0:
#     #             self.df_fit_res.loc[index,'C0 err']=self.df_fit_res.loc[index,'C0']*.75    
                
#     #     self.df_fit_res.rename(index=lambda s:f'f{s}',inplace=True)
        
#     #     # generate curve from results
#     #     self.fitcurve = pd.DataFrame(np.array([self.c_np_new(self.data['Depth'],x[1][0],x[1][2]) for x in self.df_fit_res.iterrows()]).T,columns=[f'f{y}' for y in range(len(self.df_fit_res))])
        
#     #     # generate error
#     #     self.masks_err = [self.fitcurve[x]>self.data['Na']-self.data['Na']*.5 for x in self.fitcurve]
        
#     #     self.df_fit_res['MSLE'] = [mean_squared_log_error(self.fitcurve.loc[self.masks_err[x],f'f{x}'],self.data.loc[self.masks_err[x],'Na']) for x in range(len(self.df_fit_res))]

#     #     # if we want limited statitstics, this makes sure there's no overlaps
#     #     if limited:        
            
#     #         if perc!=0:
#     #             self.fitintervals = pd.DataFrame([[pd.Interval(x[1][0]-x[1][0]*perc,x[1][0]+x[1][0]*perc,closed='both'),pd.Interval(x[1][2]-x[1][2]*perc,x[1][2]+x[1][2]*perc,closed='both')] for x in self.df_fit_res.iterrows()],columns=['D','C0'],index=list(self.df_fit_res.index))
#     #         else:
#     #             self.fitintervals = pd.DataFrame([[pd.Interval(x[1][0]-x[1][1],x[1][0]+x[1][1],closed='both'),pd.Interval(x[1][2]-x[1][3],x[1][2]+x[1][3],closed='both')] for x in self.df_fit_res.iterrows()],columns=['D','C0'],index=list(self.df_fit_res.index))

#     #         self.fitlogic = pd.DataFrame().reindex_like(self.fitintervals)
#     #         self.fitlogic['D'] = [[x for x in list(self.fitintervals['D'].index) if self.fitintervals.loc[x,'D']!=y and y.overlaps(self.fitintervals.loc[x,'D'])] for y in self.fitintervals['D']]
#     #         self.fitlogic['C0'] = [[x for x in list(self.fitintervals['C0'].index) if self.fitintervals.loc[x,'C0']!=y and y.overlaps(self.fitintervals.loc[x,'C0'])] for y in self.fitintervals['C0']]
#     #         self.fitlogic['D']=self.fitlogic['D'].transform(lambda x: ', '.join(x))
#     #         self.fitlogic['C0']=self.fitlogic['C0'].transform(lambda x: ', '.join(x))
            
#     #         # evaluates overlaps, finds 
#     #         self.fit_ignore=[]
#     #         for index in self.fitlogic.index:
#     #             # if self.fitlogic.loc[index,'D']==self.fitlogic.loc[index,'C0'] and self.fitlogic.loc[index,'D'] != '' and index not in self.fit_ignore:
#     #             if self.fitlogic.loc[index,'D'] != '' and index not in self.fit_ignore:
#     #                 sisters=list(self.fitlogic.loc[index,'D'].split(', '))
#     #                 best_true=sum(self.data['Na'][:self.lim_loc]<self.fitcurve[index][:self.lim_loc])
#     #                 for tests in sisters:
#     #                     if tests not in self.fit_ignore:
#     #                         sister_true=sum(self.data['Na'][:self.lim_loc]<self.fitcurve[tests][:self.lim_loc])
#     #                         if sister_true<best_true:
#     #                             best_true=sister_true
#     #                             if index not in self.fit_ignore:
#     #                                 self.fit_ignore.append(index)
#     #                         elif self.df_fit_res.loc[tests,'MSLE']<self.df_fit_res.loc[index,'MSLE']:
#     #                             best_true=sister_true
#     #                             if index not in self.fit_ignore:
#     #                                 self.fit_ignore.append(index)
#     #                         else:
#     #                             if tests not in self.fit_ignore:
#     #                                 self.fit_ignore.append(tests)
                            
#     #         self.fit_print=[index for index in self.fitlogic.index if index not in self.fit_ignore]
#     #         self.df_fit_res = self.df_fit_res.loc[self.fit_print,:]
        
#     #     self.df_fit_res.set_axis([self.sample+' '+self.measurement+ f' f{x}' for x in range(len(self.df_fit_res))],inplace=True)