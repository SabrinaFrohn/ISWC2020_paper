# encode utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import pomegranate as pom
import seaborn as sns

import pandas as pd
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize, minmax_scale

from scipy.stats import ttest_1samp, ttest_rel
from scipy.signal import butter, lfilter

#import jw_wavelet_scripts

import pycwt 
from pycwt.helpers import rect, fft, fft_kwargs
from scipy.signal import convolve2d


def butter_lowpass_filter(data, cutoff, fs, order=5 ):    
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5 ):    
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y

def detrend( d ):
    # use a polynomial fit to remove trends and normalise data
    t = range(0,len(d))
    p = np.polyfit(t,d, 1)
    dat_notrend = d - np.polyval(p, t)
    std = dat_notrend.std()  # Standard deviation
    var = std ** 2  # Variance
    dat_norm = dat_notrend / std  # Normalized dataset
    return dat_norm


def smooth( W, dt, dj, scales, deltaj0):
    """Smoothing function used in coherence analysis.
    Hack adaptation to let smoothing work with non-morlet wavelets
    Parameters
    ----------
    W :
    dt :
    dj :
    scales :
    deltaj0: taken from mother wavelet class object
    Returns
    -------
    T :
    """
    # The smoothing is performed by using a filter given by the absolute
    # value of the wavelet function at each scale, normalized to have a
    # total weight of unity, according to suggestions by Torrence &
    # Webster (1999) and by Grinsted et al. (2004).
    m, n = W.shape

    # Filter in time.
    k = 2 * np.pi * fft.fftfreq(fft_kwargs(W[0, :])['n'])
    k2 = k ** 2
    snorm = scales / dt
    # Smoothing by Gaussian window (absolute value of wavelet function)
    # using the convolution theorem: multiplication by Gaussian curve in
    # Fourier domain for each scale, outer product of scale and frequency
    F = np.exp(-0.5 * (snorm[:, np.newaxis] ** 2) * k2)  # Outer product
    smooth = fft.ifft(F * fft.fft(W, axis=1, **fft_kwargs(W[0, :])),
                      axis=1,  # Along Fourier frequencies
                      **fft_kwargs(W[0, :], overwrite_x=True))
    T = smooth[:, :n]  # Remove possibly padded region due to FFT

    if np.isreal(W).all():
        T = T.real

    # Filter in scale. For the Morlet wavelet it's simply a boxcar with
    # 0.6 width.
    wsize = deltaj0 / dj * 2
    win = rect(np.int(np.round(wsize)), normalize=True)
    T = convolve2d(T, win[:, np.newaxis], 'same')  # Scales are "vertical"

    return T


def wt( D, dj=1/12, s0=-1, J=-1,  wavelet='morlet', normalize=True ):
    
    """Wavelet transform for a single continuous vector y1
    Parameters
    ----------
    D : pandas Series with continuous timeseries index 
    dj : float, optional
        Spacing between discrete scales. Default value is 1/12.
        Smaller values will result in better scale resolution, but
        slower calculation and plot.
    s0 : float, optional
        Smallest scale of the wavelet. Default value is 2*dt.
    J : float, optional
        Number of scales less one. Scales range from s0 up to
        s0 * 2**(J * dj), which gives a total of (J + 1) scales.
        Default is J = (log2(N*dt/so))/dj.
        
    """
    
    dt = pd.Timedelta( D.index[1] - D.index[0] )
    #dt = 1e5/dt.value
    #assert(dt==0.01)
    dt = dt.value / 1e9

    wavelet = pycwt.wavelet._check_parameter_wavelet(wavelet)
    
    # Makes sure input signals are numpy arrays.
    y1 = np.array( D.T.values )
    
    # Checking some input parameters
    if s0 == -1:
        # Number of scales
        s0 = 2 * dt / wavelet.flambda()
    if J == -1:
        # Number of scales
        J = np.int(np.round(np.log2(y1.size * dt / s0) / dj))

    # Calculates the standard deviation of input signal.
    std1 = y1.std()
    # Normalizes signal, if appropriate.
    if normalize:
        y1_normal = (y1 - y1.mean()) / std1
    else:
        y1_normal = y1

    # Calculates the CWT of the time-series 
    _kwargs = dict(dj=dj, s0=s0, J=J, wavelet=wavelet)
    W1, sj, freq, coi, _, _ = pycwt.wavelet.cwt(y1_normal, dt, **_kwargs)
    #scales1 = np.ones([1, y1.size]) * sj[:, None]
    
    W = pd.DataFrame(data=W1.T, columns=sj, index=D.index)
    
    return W, coi # {'W':W1, 'scales':sj, 'freq':freq, 'coi':coi }


def wcompare( W1, W2, dj  ):
    """Wavelet coherence transform (WCT) and cross-wavelet comparison

    The WCT finds regions in time frequency space where the two time
    series co-vary, but do not necessarily have high power.
    
    Uses pre-computed wavelets (they must have matching sizes),
    and using the same parameters dt, dj, sj (explain!!!!)
    
    W1, W2:  matching input wavelet transforms in timeseries vs. scales
    dj : float
        Spacing between discrete scales. Default value is 1/12.
        Smaller values will result in better scale resolution, but
        slower calculation and plot.
    
    
    """
    
    wavelet = pycwt.wavelet._check_parameter_wavelet('morlet')
    
    assert(W1.shape==W2.shape)
    sj = W1.columns.values 
    dt = pd.Timedelta( W1.index[1] - W1.index[0] )
    #dt = 1e5/dt.value
    #assert(dt==0.01)
    dt = dt.value / 1e9
    data_len = W1.shape[0]
    scales = np.ones([1, data_len]) * sj[:, None]
    
    #print('%s' % (scales))
    
    _W1 = W1.T.values
    _W2 = W2.T.values
    
    S1 = smooth(np.abs(_W1) ** 2 / scales, dt, dj, sj, wavelet.deltaj0)
    S2 = smooth(np.abs(_W2) ** 2 / scales, dt, dj, sj, wavelet.deltaj0)

    # cross-wavelet transform 
    _W12 = _W1 * _W2.conj()

    #! Using a local adapted version of this to allow use with non-Morlet wavelets CHECK!
    S12 = smooth(_W12 / scales, dt, dj, sj, wavelet.deltaj0)
        
    _WCT = np.abs(S12) ** 2 / (S1 * S2)
    
    W12 = pd.DataFrame(data=_W12.T, columns=sj, index=W1.index)
    WCT = pd.DataFrame(data=_WCT.T, columns=sj, index=W1.index)
    
    
    return  WCT, W12
    

# plot a wavelet (W) alongside its constitent raw signals (D)
# also show the average wavelet power value calculation (over all frequencies)
def plot_wavelet_signal( W, D, title, W2= None ):
    # plot these outputs     comparing inverse wct with correlation and summed coherence
    plt.ion()    
    fig = plt.figure(title, **dict(figsize=(11, 8), dpi=72))

    period = W.columns.values
    freqs = 1/period
    t = W.reset_index(drop=True).index #np.array(np.arange(1,len(w)+1))#[x.value/10e8 for x in w.index]) # in seconds
    
    levels = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
    
    bx = plt.axes([0.1, 0.20, 0.85, 0.15])
    W.mean(axis=1).reset_index(drop=True).plot(ax=bx, label='W1')
        
    cx = plt.axes([0.1, 0.05, 0.85, 0.15], sharex=bx)
    D.reset_index(drop=True).plot(ax=cx, label='D')

    
    if W2 is not None:
        W2.mean(axis=1).reset_index(drop=True).plot(ax=bx, label='W2')
        dx = plt.axes([0.1, 0.70, 0.85, 0.30], sharex=bx)        
        p2 = dx.contourf(t, np.log2(period), np.log2(np.abs(W2).values.T ** 2), np.log2(levels),
        extend='both', cmap=plt.cm.viridis )
        ax = plt.axes([0.1, 0.38, 0.85, 0.30], sharex=bx)        
    else:
        ax = plt.axes([0.1, 0.58, 0.85, 0.45], sharex=bx)        

    ax.set_ybound(np.log2(min(period)), max(np.log2(period)))   
    p1=ax.contourf(t, np.log2(period), np.log2(np.abs(W).values.T ** 2), np.log2(levels),
        extend='both', cmap=plt.cm.viridis )

    ax.set_ybound(np.log2(min(period)), max(np.log2(period)))
    ax.set_ylabel('Period (seconds)')

    Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                               np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(Yticks))
    ax.set_yticklabels(Yticks)    
   

    # because the x labels have been reset to 1,2,3... reintroduce timing information as labels
    #ax.set_xticklabels( pd.TimedeltaIndex(D.index - D.index[0]) )
    
    fig.suptitle(title)

    return fig


def do_stats(freq_analysis_face, freq_analysis_back, sig_level=0.05, fdr_level=0.05):
    diff_face_back = freq_analysis_face - freq_analysis_back
    
    # calculate the stats -- using ttest_rel because the same particpants are being compared
    #stat, pval = ttest_rel( freq_analysis_face, freq_analysis_back, axis=1, nan_policy='omit' )
    t_stat, pval = ttest_1samp( diff_face_back, 0, axis=1, nan_policy='omit' )
    pval=np.nan_to_num(pval,0)

    # calculate cohen's d (https://www.socscistatistics.com/effectsize/default3.aspx)
    u_cohen_d = (freq_analysis_face.mean(1)-freq_analysis_back.mean(1)) / ((freq_analysis_face.std(1)**2 + freq_analysis_back.std(1)**2)/2).apply(np.sqrt)
    u_diff_sig = u_cohen_d[pval<sig_level]


    # FDR correction
    from statsmodels.stats.multitest import fdrcorrection
    reject_fdr, pval_fdr = fdrcorrection(pval, alpha=0.05, method='indep')
    u_diff_fdr = u_cohen_d[pval_fdr<fdr_level]
   
    return t_stat, pval, pval_fdr, u_cohen_d, u_diff_sig, u_diff_fdr 
    


def show_stats( A, B ):
    t_stat, pval, pval_fdr, u_cohen_d, u_diff_sig, u_diff_fdr_sig = do_stats(A,B,0.05,0.05)

    plt.plot(u_cohen_d)
    plt.plot(u_diff_sig,marker='x',color='b',linestyle='')
    plt.plot(u_diff_fdr_sig,marker='*',color='r',linestyle='')
    


def plot_stats( W_A, W_B, plan=None, sel_time = None, compare = 'all', ttl = '', ttl_ff = 'A', ttl_bb = 'B', show_support=False ):
    # Analysis 2: calculate statistics on dominant freqency bands for each condition
    # if no plan data specified, just compare A with B
    
  #  if plan is None:
    faceA = W_A.keys()
    faceB = W_B.keys()        
  #  else:
  #      faceA, faceB, ttl_ff, ttl_bb = get_direction(plan,compare)
        
    pairs = W_A.keys()

    freq_analysis_face, freq_analysis_back =pd.DataFrame(),pd.DataFrame()
    for p in pairs:
        try:
            if sel_time == None:       
                sel = range(1, min(W_A[p].shape[0], W_B[p].shape[0]))
            else:
                sel = sel_time

            if p in faceA:
                freq_analysis_face[p] = W_A[p].iloc[sel].mean(axis=0) 
                freq_analysis_back[p] = W_B[p].iloc[sel].mean(axis=0) 
            elif p in faceB:
                freq_analysis_face[p] = W_B[p].iloc[sel].mean(axis=0) 
                freq_analysis_back[p] = W_A[p].iloc[sel].mean(axis=0) 
        except:
            print('issue with pair %s and %s' % (p, sel_time))
            break
            
    # average coherence levels
    u_face = freq_analysis_face.mean(axis=1)
    u_back = freq_analysis_back.mean(axis=1)
    
    # difference matrix
    diff_face_back = freq_analysis_face - freq_analysis_back

    # Do some significance statistics on the results
    t_stat, pval, pval_fdr, u_cohen_d, u_diff_sig, u_diff_fdr = do_stats(freq_analysis_face,freq_analysis_back,0.05,0.05)
    
    if show_support:
        fig,axes = plt.subplots(3,1,sharex=True,sharey=True)

        freq_analysis_face.plot(title =ttl_ff, ax=axes[0])
        freq_analysis_back.plot(title=ttl_bb,  ax=axes[1])

        u_face.plot(ax=axes[2], label=ttl_ff)
        u_back.plot(ax=axes[2], label=ttl_bb)

        plt.legend()

        axes[1].invert_xaxis()
        axes[1].set_xscale('log')
        axes[1].set_xlabel('log period (s)')
        axes[1].set_xlim([100,0.1])
        

    #SEM error of the mean
    face_std_err2 = freq_analysis_face.std(1).div(np.sqrt(freq_analysis_face.shape[1]))
    back_std_err2 = freq_analysis_back.std(1).div(np.sqrt(freq_analysis_back.shape[1]))

    # prepare the 1st subplot showing coherence results
    fig,axes = plt.subplots(2,1,sharex=True,sharey=False)
    #diff_face_back.plot( ax=axes[0] )
    u_face.plot(ax=axes[0], label=ttl_ff, color = '#0000FF') # blue
    axes[0].fill_between( u_face.index, u_face-face_std_err2, u_face+face_std_err2, alpha=0.2, edgecolor='#0000FF', facecolor='#0000FF')

    u_back.plot(ax=axes[0], label=ttl_bb, color = '#FF0000', linestyle=':') # red
    axes[0].fill_between( u_back.index, u_back-back_std_err2, u_back+back_std_err2, alpha=0.2, edgecolor='#FF0000', facecolor='#FF0000')
       
    
    axes[0].invert_xaxis()
    axes[0].set_xscale('log')
#    axes[0].set_xlabel('log period (s)')

    if u_face.mean(0)>0.5:
        axes[0].set_ylim([0.1,1])
    elif u_face.mean(0)>0.25:
        axes[0].set_ylim([0.1,0.5])
    else:
        axes[0].set_ylim([0.1,0.3])
    axes[0].legend()
    axes[0].set_ylabel('avg. coherence')
    
    # 2nd subplot -- showing difference 
    u_cohen_d.plot( ax=axes[1], label='d')
    # overlay significance values on cohen's d difference measure
    try:
        u_diff_sig.plot( ax=axes[1], label='p<0.05', color='b', marker='.', linestyle='')
    except:
        pass
   
    try:
        u_diff_fdr.plot( ax=axes[1], label='p(FDR)<0.05', color='r', marker='*', linestyle='')
    except:
        pass

    axes[1].legend()
    axes[1].invert_xaxis()
    axes[1].set_xscale('log')
    axes[1].set_xlabel('log period (s)')
    axes[1].set_xlim([100,0.1])
    if u_cohen_d.apply(np.abs).max() < 1:
        axes[1].set_ylim([-1,1])
    axes[1].set_ylabel('effect size (d)')
    
    fig.suptitle(ttl)
    
    return pd.DataFrame({'pval':pval,'pval_fdr':pval_fdr,'u_cohen_d': u_cohen_d})
 
 