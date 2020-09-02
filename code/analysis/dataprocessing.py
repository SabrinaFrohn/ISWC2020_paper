# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 15:58:14 2018

@author: jward
"""

#%matplotlib 
##

import os
from zipfile import ZipFile
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from scipy.signal import correlate

def get_accelerometer_data( zipfilename, csvname='ACC.csv' ):
    
    
    zip_file = ZipFile( os.path.join(data_path, zipfilename))
    
    dfs = {text_file.filename: pd.read_csv(zip_file.open(text_file.filename))
           for text_file in zip_file.infolist()
           if text_file.filename in  csvname}  # endswith('.csv')}
    
    d = dfs[csvname]
    epoch = float(d.columns[0])
    fs = 32
    
    Acc = pd.DataFrame(d) # sampled at 32 Hz
    Acc.columns=pd.Index(['x','y','z'])
    Acc = Acc.mul(2/128) # convert into g
    Acc['epoch'] = np.arange(1,len(Acc)+1) / fs
    Acc['epoch'] = pd.to_numeric( 10e8 * (Acc['epoch'] + epoch), downcast='integer' )
    Acc['time'] = pd.DatetimeIndex(Acc['epoch'])
    Acc.set_index('time', inplace=True)
    Acc['R'] = Acc.loc[:,['x','y','z']].pow(2).sum(1).pow(0.5) 
    
    return Acc  


def get_series_data( zipfilename, csvname = 'EDA.csv'):
    
    zip_file = ZipFile( os.path.join(data_path, zipfilename))
    
    dfs = {text_file.filename: pd.read_csv(zip_file.open(text_file.filename))
           for text_file in zip_file.infolist()
           if text_file.filename in csvname}  # endswith('.csv')}

    d = dfs[csvname].loc[1:,:]
    epoch = float(d.columns[0])
    fs = float(dfs[csvname].ix[0])
    
    Dat = pd.DataFrame(d) # sampled at 32 Hz
    Dat.columns=pd.Index([csvname.strip('.csv')])
    Dat['epoch'] = np.arange(1,len(d)+1) / fs
    Dat['epoch'] = 10e8 * (Dat['epoch'] + epoch)
    Dat['time'] = pd.DatetimeIndex(Dat['epoch'])
    Dat.set_index('time', inplace=True)
 
    return Dat  


def collate_device_data( data_path ):
    Downloads = pd.read_csv( os.path.join(data_path,'downloaded_e4_zip_list.csv'))
    Acc = { dev: pd.DataFrame(columns=pd.Index(['x','y','z','R','epoch'])) for dev in Downloads['deviceID']}
    Eda= { dev: pd.DataFrame(columns=pd.Index(['EDA','epoch'])) for dev in Downloads['deviceID']}
    
    # load all data from all devices over all time into a single structure
    for idx in Downloads.index:    
        dev = Downloads.loc[idx,'deviceID']
        Acc[dev] = Acc[dev].append( 
            get_accelerometer_data( Downloads.loc[idx,'filename'] ))
        
    for dev in Acc.keys():
        Acc[dev].to_pickle( os.path.join(data_path, 'Acc', dev + '.pickle'))       
    
    del Acc
    
    for idx in Downloads.index:    
        dev = Downloads.loc[idx,'deviceID']  
        Eda[dev] = Eda[dev].append( 
            get_series_data( Downloads.loc[idx,'filename'] ))
        
    np.save(os.path.join(data_path,'E4_all_Eda.npy'),Eda)
    for dev in Eda.keys():
        Eda[dev].to_pickle( os.path.join(data_path, 'Eda', dev + '.pickle'))       



class Events:
     
    def __make_event_getter(self, event_name ):
        def get(self, e, d = None):
            if d is None:
                return self.events[self.events.event==e]
        get.__name__ = event_name                            
        return get
    
    def __init__(self, data_path, event_sheet= 'SatActors'):
        self.data_path = data_path
        
        # set up the mapping for each sensor to person-location for each experiment day
        EMap = pd.read_excel(os.path.join( data_path,'devices.xlsx'), sheet_name=event_sheet)
        
 
        self.events = EMap
#        for e in list(set(self.events['event'])):
#            self.get[e] = self.__make_event_getter(e) #self.events[self.events.event==e]          
    
    def get(self, event_name, d = None):
        e = self.events[self.events.event==event_name]
        if d is None:            
            return e

        t1 = e['datetime start']
        t2 = e['datetime stop']
        
        if len(e) > 1:
            D = {}
            for i,r in enumerate(e.iterrows()):
                t1 = r[1]['datetime start']
                t2 = r[1]['datetime stop']
                D[i] = d.loc[t1:t2,:]
            return D                        
        
        return d.loc[t1:t2,:]
            
        
        

class Devices:
    
 #   T1 = {'rosertest':'2018-03-07 13:00:00',
 #            'Mon':'2018-03-12 10:30:00', 'Tue': '2018-03-13 10:30:00','Wed': '2018-03-14 10:00:00','Thu': '2018-03-15 13:30:00', 'Fri':  '2018-03-16 09:35:00','Sat': '2018-03-17 10:30:00', # Bridge days
 #            'Fri-a':'2018-03-16 08:00:00', 'SatSync':'2018-03-17 10:02:30','SatSync2':'2018-03-17 10:03:38',
 #            'FriSync':'2018-03-16 09:39:00'}
 #   T2 =  {'rosertest':'2018-03-07 17:00:00',
 #            'Mon':'2018-03-12 12:30:00', 'Tue': '2018-03-13 13:00:00','Wed': '2018-03-14 12:30:00','Thu': '2018-03-15 16:15:00', 'Fri':  '2018-03-16 12:30:00', 'Sat': '2018-03-17 12:30:00', # Bridge days
 #            'Fri-a':'2018-03-16 09:00:00', 'SatSync':'2018-03-17 10:05:00','SatSync2':'2018-03-17 10:03:53',
 #            'FriSync':'2018-03-16 09:44:00'}  

    def __init__(self, data_path):
        self.data_path = data_path
        # set up the mapping for each sensor to person-location for each experiment day
        DayMap = pd.read_excel(os.path.join( data_path,'devices.xlsx'), sheet_name='TheatreDaysUse')
        DayMap.set_index('Number',inplace=True)
        DayMap.index = DayMap.index.astype('str')
        # obtain sync windows from excel sheet
        self.syncMap = pd.read_excel(os.path.join( data_path,'devices.xlsx'), sheet_name='Sync')
        # obtain day times from excel sheet
        self.timeMap = pd.read_excel(os.path.join( data_path,'devices.xlsx'), sheet_name='Times')

        
        self.DM = DayMap
        self.Offset = None        
        
        self._data = {}
        self._fs = {}
        self._ref_data = None
        self.day = None
        self.sync_device = ''
        self.devices = None
        self.sync_tag = ''
        self.sync_smooth_wnd = 1
        
        
        self.t1 = ''
        self.t2 = ''
        
    
    def ID(self, day, name, wrist='R'):
        tag = '%s-%s' % (wrist, name)
        ids = self.DM.loc[self.DM.loc[:,day]==tag,'SN']
        if ids.size == 1:            
            return ids.ix[0]
        else: 
            return ids

    def set_day( self, day, t1 = None, t2 = None ):
        if day in self.DM.columns:
            self.day = day
            self.devices = self.DM[day].dropna().unique()    
            if t1 is None:
                t1 = str(self.timeMap.loc[self.timeMap['day']==day, 'T1'].values[0])
                #t1 = self.T1[day]
            self.t1 = t1
            if t2 is None:
                t2 = str(self.timeMap.loc[self.timeMap['day']==day, 'T2'].values[0])
                # t2 = self.T2[day]                            
            self.t2 = t2        
        else:
            print('No such day in records: %s', day)                
            
        return self

        
    def get_video_time(self):
        return str(self.timeMap.loc[self.timeMap['day']==self.day, 'VideoStart'].values[0])

    def set_sync_device(self, name, wrist, time_tag_append = 'Sync', smooth_wnd=1 ):
        dev = '%s-%s' % (wrist, name)
        if dev in self.devices:
            self.sync_device = dev
            self.sync_tag = time_tag_append
            self.sync_smooth_wnd = smooth_wnd
            return True
        else:
            print('No such device for this day: %s', dev)                
            return False    

    def _read_data(self, day, name, wrist, data_type ):
        try:            
            dev = self.ID(day, name, wrist)
            print('%s,%s,%s,%s: %s' % (day, name, wrist, data_type,dev))
            return pd.read_pickle( os.path.join( self.data_path, data_type, dev + '.pickle'))
        except:
            print('Could not find data for: %s,%s,%s,%s' % (day, name, wrist, data_type))
            return None


    def sync_data(self, data_type, scope=2500, signal='R'):
  
        day = self.day
        #t1 = self.T1[day+ self.sync_tag] 
        #t2 = self.T2[day+ self.sync_tag] 
        t1 = str(self.syncMap.loc[self.syncMap['day']==day, 'T1'].values[0])
        t2 = str(self.syncMap.loc[self.syncMap['day']==day, 'T2'].values[0])
              
        # use Acc data to sync data_type

        d_test = self._data['Acc'].loc[t1:t2,signal].rolling(self.sync_smooth_wnd).mean().dropna()
        
        d_ref = self._read_data(self.day, self.sync_device.split('-')[1], self.sync_device.split('-')[0], 'Acc')
        d_ref = d_ref.loc[t1:t2,signal].rolling(self.sync_smooth_wnd).mean().dropna()
                             
        # search at least 2300 samples either side
        xcorr = [d_ref.corr(d_test.shift(i).dropna()) for i in np.arange(-scope,scope)]
        lag = np.argmax(xcorr) - scope
        
        acc_fs = self._data['Acc'].index[1] - self._data['Acc'].index[0]
        new_fs = self._data[data_type].index[1] - self._data[data_type].index[0]        
                
        print('lag (ACC): %s' % lag )
        # adjust lag to sample rate of thing to be shifted                        
        lag = int(lag * acc_fs / new_fs)        
            
        print('lag (%s): %s' % (data_type,lag) )
        self._data[data_type] = self._data[data_type].shift( lag )
        
        #d_test = d_test.shift(lag)        
        # d_ref.plot()
        # d_test.plot()
        
        return lag
    
    
    def sync_again(self, D, ref_tag = 'R-Jam-R', scope=500):
        ' re-apply the sync algorithm to the (already synced) datastructure, D'
        
        Tmp = D.iloc[0:scope*5,:] 
        
        for k in D.keys():
            xcorr = [Tmp[ref_tag].corr(Tmp[k].shift(i).dropna()) for i in np.arange(-scope,scope)]
            lag = np.argmax(xcorr) - scope
            D[k] = D[k].shift( lag )             
            print('applying lag %s to %s' % (lag,k) )
        
        return D
    

    def get_data(self, name, wrist='R', data_type='Acc', do_sync = True, offset = None, scope=2500 ): # sync data by default
                    
        self._data[data_type] = self._read_data(self.day, name, wrist, data_type)  # obtain all the data for this device and day                

        if offset is not None:            
            self._data[data_type] = self._data[data_type].shift(freq=offset)

        if do_sync:                                        
            if data_type not in 'Acc':
                self._data['Acc'] = self._read_data(self.day, name, wrist, 'Acc')                        
                if offset is not None:                   
                    self._data['Acc'] = self._data['Acc'].shift(freq=offset)

            lag = self.sync_data(data_type, signal = 'x', scope = scope)      
            print('shifting %s-%s by %s' % (wrist, name, lag))
        
        return self._data[data_type].loc[self.t1:self.t2] 


        
    def all_single_data(self, day, data_type='Acc', signal = 'R', do_sync = True ):
        # retrieve all data from all people for day for a single signal
        if signal is None:                
            d = pd.DataFrame({ wrist_name: 
                self.get_data(wrist_name.split('-')[1], wrist_name.split('-')[0], data_type, do_sync ) 
                for wrist_name in self.devices
                    if len(wrist_name.split('-'))==2 })         
        else:
            d = pd.DataFrame({ wrist_name: 
                self.get_data(wrist_name.split('-')[1], wrist_name.split('-')[0], data_type, do_sync )[signal] 
                for wrist_name in self.devices
                    if len(wrist_name.split('-'))==2 })         
            
        return d
    
    def all_data(self, day, data_type='Acc', do_sync = True, fields=None ):
        # retrieve all data from all people for day 
        D = pd.DataFrame()
        for wrist_name in self.devices:
            if len(wrist_name.split('-'))==2:
                wrist = wrist_name.split('-')[0]
                name = wrist_name.split('-')[1]        
                d = self.get_data(name, wrist, data_type, do_sync ) 
                for c in d.columns:
                    if (fields is None) or (c in fields):                        
                        D[wrist_name + '-' + c] = d[c]
        return D
    
    
    
    def markers(self, day, person='Jam' ):
        
        t1 = str(self.syncMap.loc[self.timeMap['day']==day, 'T1'].values[0])
        t2 = str(self.syncMap.loc[self.timeMap['day']==day, 'T2'].values[0])
        
        d = pd.read_excel(os.path.join( self.data_path,'devices.xlsx'), sheet_name=day+person)
        d = d.loc[d.datetime > t1,:]
        d = d.loc[d.datetime < t2,:]
        return d['datetime']
    
    
    

        return data
                
def plot_all_rss_data(data, day, txt = None):
    
    if day is 'Sat':
        Right = ['R-' + s for s in ['E5','E7','E8','E9','Gab','Fin','Pau','Oll','Tom','Taz']] # [c for c in data.columns if 'R-' in c]
        Left =  ['L-' + s for s in ['E5','E7','E8','E9','Gab','Fin','Pau','Oll','Tom','Taz']] #[c for c in data.columns if 'L-' in c]
    elif day is 'Fri':
        Right = ['R-' + s + '-R' for s in ['D5','D7','D8','Gab','Fin','Pau','Oll','Tom','Taz']] # [c for c in data.columns if 'R-' in c]
        Left =  ['L-' + s + '-R' for s in ['D5','D7','D8','Gab','Fin','Pau','Oll','Tom','Taz']] #[c for c in data.columns if 'L-' in c]
    elif day is 'Thu':
        Right = ['R-' + s + '-R' for s in ['C5','C8','C9','Gab','Fin','Pau','Oll','Tom','Taz']] # [c for c in data.columns if 'R-' in c]
        Left =  ['L-' + s + '-R' for s in ['C5','C8','C9','Gab','Fin','Pau','Oll','Tom','Taz']] #[c for c in data.columns if 'L-' in c]
        
    
    # find any markers
    markers = Dev.markers( day, person='Jam' )
    

    f, ax = plt.subplots(2, sharex=True, sharey=False)
    f.subplots_adjust(hspace=0)

    
    plt.setp(ax[0], yticks=np.arange(0,len(Right)+1), yticklabels=Right )
    plt.setp(ax[1], yticks=np.arange(0,len(Left)+1), yticklabels=Left )
        
    for i,x in enumerate(markers):
        ax[0].axvline(x=x)
        ax[0].text(x,-.8,str(i))
        ax[1].axvline(x=x)
        ax[1].text(x,-.8,str(i))
    
    [ax[0].plot(i + data[c]/3 -.5, label=c) 
        for i,c in enumerate(Right)] 
        
    [ax[1].plot(i + data[c]/3 -.5, label=c) 
        for i,c in enumerate(Left) ]
        
        
    if txt is None:
        txt = 'Bridge Dataset %s' % (day)
        
    ax[0].set_title( txt + ' (Right wrist)' )
    ax[1].set_title( txt + ' (Left wrist)' )
    
    f.tight_layout()
    f.savefig(os.path.join( data_path,'fig', txt+'.png' ), bbox_inches='tight')    
    
    return f
    

if __name__ == "__main__":

    ##
    # next step: read the zip files, synchronise and add all data into one big pandas df
    data_path = r'data'
    day = 'thu'

    #DeviceMap = {'A000A9':33, 'A0002B':4, 'A00030':2, 'A0009E':18, 'A003E1':20, 'A00034':10, 'A004D9':29, 'A00075':13, 'A003FE':37, 'A00214':35, 'A00100':19, 'A001CF':39, 'A00217':23, 'A000EE':31, 'A00213':30, 'A00062':3, 'A001F2':38, 'A001A6':26, 'A001D6':12, 'A0005A':7, 'A001D3':17,
    #             'A0004D':6,'A00132':27,'A0000D':5,'A0001D':9,'A001F0':14,'A00044':8,'A0037A':40}    
    #Actors = ['Taz','Tom','Pau','Gab','Oll','Fin']
    #Kids = ['A9','B8','C5','C8','C9','D5','D7','D8','E5','E7','E8','E9']
    
    
    # collate_device_data( data_path )  # use this to calculate datastructures Acc and Eda
    Dev = Devices( data_path )   
   
    Dev.set_day(day)
    Dev.set_sync_device('Jam','R',smooth_wnd=10)
      
 #   d=Dev.get_data('Oll','L')
 #   d=Dev.get_data('Fin','L')

    fn_data = os.path.join(data_path, 'Acc', 'all-' + day + '.pickle' )
    if False: 
        D = Dev.all_data(day, 'Acc', do_sync=True, fields=['x','y','z','R'])      
        # necessary timing hacks
        #! imporove this
        if day == 'Fri':
            d_d7 = Dev.get_data('D7', wrist='L', data_type='Acc', do_sync=True, offset = '3480s')
            D['L-D7-R'] =  d_d7['R']
            D['L-D7-x'] =  d_d7['x']
            D['L-D7-y'] =  d_d7['y']
            D['L-D7-z'] =  d_d7['z']
        if day == 'Sat':
            d_ = Dev.get_data('E7', wrist='L', data_type='Acc', do_sync=True, offset = '1920s')
            D['L-E7-R'] =  d_['R']
            D['L-E7-x'] =  d_['x']
            D['L-E7-y'] =  d_['y']
            D['L-E7-z'] =  d_['z']
            d_ = Dev.get_data('E5', wrist='L', data_type='Acc', do_sync=True, offset = '1s', scope = 10)
            d_ = d_.shift(-40)
            D['L-E5-R'] =  d_['R']
            D['L-E5-x'] =  d_['x']
            D['L-E5-y'] =  d_['y']
            D['L-E5-z'] =  d_['z']

        # save all the synced data    
        D.to_pickle(fn_data)
    
    if True:
        # plot synced data
        D =  pd.read_pickle( fn_data ) 
        data = D.loc[:,[c for c in D.columns if c.endswith('-R')]]    
        data.rename({k:k[:-2] for k in data.columns}, axis=1, inplace=True)
        data.to_pickle(os.path.join(data_path, 'Acc', day + '.pickle' ))
        f = plot_all_rss_data(data, day, txt = 'Bridge dataset %s ' % day)
        f = plot_all_rss_data(data['17-03-18 10:03':'17-03-18 10:05'], day, txt = 'Bridge dataset closeup %s ' % day)
    else:
        # plot unsynced data
        D = Dev.all_data(day, 'Acc', do_sync=False, fields=['x','y','z','R'])      
        data = D.loc[:,[c for c in D.columns if c.endswith('-R')]]    
        data.rename({k:k[:-2] for k in data.columns}, axis=1, inplace=True)
        f = plot_all_rss_data(data, day, txt = 'Bridge dataset unsynced %s ' % day)
        f = plot_all_rss_data(data['17-03-18 10:03':'17-03-18 10:05'], day, txt = 'Bridge dataset unsynced closeup %s ' % day)



    #data = Dev.all_single_data(day, 'Acc', signal = 'R', do_sync=True)
    # save all the RSS data 

   # A = Dev.sync_again( D, ref_tag = 'R-Jam-R', scope=500 )

 # this bit needs fixing...
#    eda = Dev.all_data(day, data_type='Eda', signal = None, do_sync=True)
#    eda.to_pickle(os.path.join(data_path, 'Eda', day + '.pickle' ))

    
#    data = Dev.accsync(d.copy(), day, sync_device='R-Jam')
    
    #data = D
##

    
    
    #manager = plt.get_current_fig_manager().window.showMaximized()



