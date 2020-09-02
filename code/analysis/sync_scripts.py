#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:05:44 2019

@author: jamieward
"""
import numpy as np


def sync_data( D, ref_tag, scope=200, ignore_buf=100):
    ''' 
        Synchronise data in the datastructure, D, to match D[ref_tag]
        Use ref_tag column as the reference data
        Scope specifies up until how many samples around the beginning of D to search for
          a sync gesture.
        ignore_buf specifies the buffer of samples at the start and end of the -scope and +scope range to ignore
    '''
  
    Tmp = D.iloc[0:scope,:].copy() 

    for k in D.keys():
            xcorr = [Tmp[ref_tag].corr(Tmp[k].shift(i).dropna()) for i in np.arange(-scope,scope)]        
            lag = np.argmax(np.nan_to_num(xcorr[ignore_buf:][:-ignore_buf]))
            D[k] = D[k].shift( lag )        
            print('applying lag %s to %s' % (lag,k) )
    
    return D


