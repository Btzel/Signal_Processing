# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 12:53:59 2023

@author: BurakT

Note: Select your directory as current working directory (where the script exetuced in) before run (os.getcwd() get the current working directory at line 18..)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
import pandas as pd
import os
# get ready to use PPG signal
raw_signals = pd.read_csv(os.getcwd()+"/mimic_perform_af_csv/mimic_perform_af_001_data.csv")
ppg = raw_signals["PPG"].to_numpy()
# sampling frequency 360Hz
fs = 125

# generating time axis values
time = np.arange(ppg.size) / fs
winsize = fs * 4
winhop = fs * 2
i = 10

# Smoothing the PPG signal using a moving average filter
window_size = 10  # Adjust the window size as needed
ppg_smoothed = np.convolve(ppg, np.ones(window_size)/window_size, mode='same')

def compute_derivative(signal):
    return np.gradient(signal)
def butter_lowpass_filter(data, cutoff_frequency, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def baseline_correction(signal, window_size=101):
    # Apply a moving average to estimate the baseline
    baseline = np.convolve(signal, np.ones(window_size)/window_size, mode='same')

    # Subtract the baseline from the original signal
    corrected_signal = signal - baseline

    return corrected_signal

def findZeroCrossingsFPP(signal,peaks):
    peak_indices = peaks

    zero_before_peak_indices = []
    zero_after_peak_indices = []
    
    for peak_index in peak_indices:
        

        zero_crossings_before_peak = np.where(np.diff(np.signbit(signal[:peak_index])))[0]
        

        if len(zero_crossings_before_peak) > 0:
            zero_before_peak_index = zero_crossings_before_peak[-1]
            zero_before_peak_indices.append(zero_before_peak_index)
        else:
            zero_before_peak_indices.append(0)


        zero_crossings_after_peak = np.where(np.diff(np.signbit(signal[peak_index:])))[0]
        

        if len(zero_crossings_after_peak) > 0:
            zero_after_peak_index = zero_crossings_after_peak[0] + peak_index
            zero_after_peak_indices.append(zero_after_peak_index)
        else:
            zero_after_peak_indices.append(len(signal)-1)
        
    return zero_before_peak_indices,zero_after_peak_indices

def findZeroCrossingsFNP(signal,peaks):
    peak_indices = peaks 

    zero_before_negative_peak_indices = []
    zero_after_negative_peak_indices = []

    for peak_index in peak_indices:

        zero_crossings_before_peak = np.where(np.diff(np.signbit(-signal[:peak_index])))[0]
        

        if len(zero_crossings_before_peak) > 0:
            zero_before_peak_index = zero_crossings_before_peak[-1]
            zero_before_negative_peak_indices.append(zero_before_peak_index)
        else:
            zero_before_negative_peak_indices.append(0)
        

        zero_crossings_after_peak = np.where(np.diff(np.signbit(-signal[peak_index:])))[0]
        

        if len(zero_crossings_after_peak) > 0:
            zero_after_peak_index = zero_crossings_after_peak[0] + peak_index
            zero_after_negative_peak_indices.append(zero_after_peak_index)
        else:
            zero_after_negative_peak_indices.append(len(signal) - 1)
    
    return zero_before_negative_peak_indices,zero_after_negative_peak_indices

def findEKeyPoints(SPG,JPG,jpg_p0_peaks,apg_a_peaks):
    peaks, _ = find_peaks(SPG)
    
    init_q1_peaks = []
    init_q3_peaks = []
    conc_q1_peaks = []
    conc_q3_peaks = []
    conc_q1_q3_peaks = []
    conc_e_peaks = []
    
    peaks_after_points = {}
    for point_index in jpg_p0_peaks:
        try:
            peaks_after_point = [peak for peak in peaks if peak > point_index]
            peaks_after_points[point_index] = peaks_after_point
            if(len(peaks_after_points[point_index]) > 0):
                init_q1_peaks.append(peaks_after_points[point_index][0])
            if(len(peaks_after_points[point_index]) > 1):
                init_q3_peaks.append(peaks_after_points[point_index][1])
        except IndexError:('List index out of range')
    _,zero_amplitude_after_q3 = findZeroCrossingsFPP(SPG, init_q3_peaks)
    init_p3_peaks = zero_amplitude_after_q3
    _,zero_amplitude_after_p3 = findZeroCrossingsFPP(JPG, init_p3_peaks)
    init_e_peaks = zero_amplitude_after_p3

    
    peaks_after_points = {}
    for i in range(len(jpg_p0_peaks)):
        peaks_after_point = [peak for peak in peaks if peak > jpg_p0_peaks[i]]
        if(len(peaks_after_point) > 0):
            try:
                peaks_after_points[jpg_p0_peaks[i]] = peaks_after_point
                if i+1 in range(len(apg_a_peaks)):
                    if(init_e_peaks[i]-apg_a_peaks[i] > 0.55*(apg_a_peaks[i+1]-apg_a_peaks[i])):
                        if(len(peaks_after_points[jpg_p0_peaks[i]]) > 0):
                            conc_q1_q3_peaks.append(peaks_after_points[jpg_p0_peaks[i]][0]) 
                    else:
                        if(len(peaks_after_points[jpg_p0_peaks[i]]) > 0):
                            
                            conc_q1_peaks.append(peaks_after_points[jpg_p0_peaks[i]][0])
                        if(len(peaks_after_points[jpg_p0_peaks[i]]) > 1):
                            
                            conc_q3_peaks.append(peaks_after_points[jpg_p0_peaks[i]][1])
                else:
                    conc_q1_q3_peaks.append(peaks_after_points[jpg_p0_peaks[i]][0])
            except IndexError:('List index out of range')
            
    _,zero_amplitude_after_q1_q3 = findZeroCrossingsFPP(SPG, conc_q1_q3_peaks)
    _,zero_amplitude_after_q3 = findZeroCrossingsFPP(SPG, conc_q3_peaks)
    conc_separated_p3_peaks = zero_amplitude_after_q3
    conc_merged_p3_peaks = zero_amplitude_after_q1_q3
    _,temp_1_e_peaks = findZeroCrossingsFPP(JPG, conc_separated_p3_peaks)
    _,temp_2_e_peaks = findZeroCrossingsFPP(JPG, conc_merged_p3_peaks)
    conc_e_peaks = list(set(temp_1_e_peaks) | set(temp_2_e_peaks))
    
    
    
    return conc_e_peaks,conc_separated_p3_peaks,conc_merged_p3_peaks,conc_q1_q3_peaks,conc_q1_peaks,conc_q3_peaks

def findFKeyPoints(SPG,JPG,jpg_merged_p3_peaks,jpg_separated_p3_peaks,):
    spg_q4_peaks = []
    peaks,_ = find_peaks(-SPG)
    peaks_after_points = {}

    for point_index in jpg_merged_p3_peaks:
        peaks_after_point = [peak for peak in peaks if peak > point_index]
        peaks_after_points[point_index] = peaks_after_point
        if(len(peaks_after_points[point_index]) > 0):
            spg_q4_peaks.append(peaks_after_points[point_index][0])
    for point_index in jpg_separated_p3_peaks:
        peaks_after_point = [peak for peak in peaks if peak > point_index]
        peaks_after_points[point_index] = peaks_after_point
        if(len(peaks_after_points[point_index]) > 0):
            spg_q4_peaks.append(peaks_after_points[point_index][0])
    _,zero_amplitude_after_q4 = findZeroCrossingsFNP(SPG, spg_q4_peaks)
    jpg_p4_peaks = zero_amplitude_after_q4
    _,zero_amplitude_after_p4 = findZeroCrossingsFNP(JPG, jpg_p4_peaks)
    apg_f_peaks = zero_amplitude_after_p4
    ppg_distolic_peaks = apg_f_peaks
    
    return jpg_p4_peaks,apg_f_peaks,ppg_distolic_peaks,spg_q4_peaks

def PlottingTimeDifferencesOnPPG(PPG,ppg_o_peaks,ppg_s_peaks,ppg_n_peaks,ppg_d_peaks,lower,upper):
    ppg_o_peaks = np.sort(ppg_o_peaks)
    ppg_s_peaks = np.sort(ppg_s_peaks)
    ppg_d_peaks = np.sort(ppg_d_peaks)
    ppg_n_peaks = np.sort(ppg_n_peaks)
    
    for i in range(len(ppg_o_peaks)):
        
        ax2.plot([ppg_o_peaks[i], ppg_o_peaks[i]], [PPG[ppg_o_peaks[i]], -0.4], 'k-')
        global_time_value = time[lower + ppg_o_peaks[i]]  
        ax2.text(ppg_o_peaks[i]+5, PPG[ppg_o_peaks[i]] - 0.15, f'({global_time_value:.2f} s)', fontsize=7, color='black', ha='center', va='bottom') 
        
        if(i < len(ppg_o_peaks)-1):
            try:
                #PWD
                time_diff_pwd = time[ppg_o_peaks[i+1]-ppg_o_peaks[i]]
                ax2.plot([ppg_o_peaks[i], ppg_o_peaks[i+1]],[-0.4,-0.4],'k-')
                ax2.text((ppg_o_peaks[i]+ppg_o_peaks[i+1])/2,-0.4, f'PWD = {time_diff_pwd:.2f} s', fontsize=6, color='black', ha='center', va='bottom')
               
                #Systolic Phase Diastolic Phase
                time_diff_sp = time[ppg_s_peaks[i]-ppg_o_peaks[i]]
                time_diff_dp = time[ppg_o_peaks[i+1]-ppg_s_peaks[i]]
                ax2.plot([ppg_s_peaks[i], ppg_s_peaks[i]], [PPG[ppg_s_peaks[i]], 0], 'k-')
                ax2.plot([ppg_o_peaks[i], ppg_s_peaks[i]],[-0,-0],'k-')
                ax2.text((ppg_s_peaks[i]+ppg_o_peaks[i])/2,-0.2, f'Systolic Phase = {time_diff_sp:.2f} s', fontsize=5.5, color='black', ha='center', va='bottom')
                ax2.plot([ppg_s_peaks[i], ppg_o_peaks[i+1]],[-0,-0],'k-')
                ax2.text((ppg_s_peaks[i]+ppg_o_peaks[i+1])/2,-0.2, f'Diastolic Phase = {time_diff_dp:.2f} s', fontsize=5.5, color='black', ha='center', va='bottom')
                global_time_value = time[lower + ppg_s_peaks[i]]  
                ax2.text(ppg_s_peaks[i]-4.5, PPG[ppg_s_peaks[i]] - 0.2, f'({global_time_value:.2f} s)', fontsize=7, color='black', ha='center', va='bottom')
                
                #PPT
                time_diff_ppt = time[ppg_d_peaks[i]-ppg_s_peaks[i]]
                ax2.plot([ppg_s_peaks[i], ppg_s_peaks[i]], [PPG[ppg_s_peaks[i]], 0.9], 'k-')
                ax2.plot([ppg_d_peaks[i], ppg_d_peaks[i]], [PPG[ppg_d_peaks[i]], 0.9], 'k-')
                ax2.plot([ppg_s_peaks[i], ppg_d_peaks[i]],[0.9,0.9],'k-')
                ax2.text((ppg_s_peaks[i]+ppg_d_peaks[i])/2,0.75, f'PPT = {time_diff_ppt:.2f} s', fontsize=6, color='black', ha='center', va='bottom')
                
                #PWSP,PWDP,Dicrotich Notch
                ax2.plot([ppg_d_peaks[i], ppg_d_peaks[i]], [PPG[ppg_d_peaks[i]], 0], 'k-')
                ax2.plot([ppg_n_peaks[i], ppg_n_peaks[i]], [PPG[ppg_n_peaks[i]], 0], 'k-')
                amplitude_s = PPG[ppg_s_peaks[i]]
                amplitude_d = PPG[ppg_d_peaks[i]]
                amplitude_n = PPG[ppg_n_peaks[i]]
                ax2.text(ppg_s_peaks[i]+1.5, PPG[ppg_s_peaks[i]] /2 - 0.15 , f'{amplitude_s:.2f}', fontsize=6.5, color='black', ha='center', va='bottom',rotation=90)
                ax2.text(ppg_d_peaks[i]-1.2, PPG[ppg_d_peaks[i]] /2 - 0.12 , f'{amplitude_d:.2f}', fontsize=6.5, color='black', ha='center', va='bottom',rotation=90)
                ax2.text(ppg_n_peaks[i]-1.2, PPG[ppg_n_peaks[i]] /2 - 0.12 , f'{amplitude_n:.2f}', fontsize=6.5, color='black', ha='center', va='bottom',rotation=90)
                
                #PWA
                amplitude_pwa = PPG[ppg_s_peaks[i]]-PPG[ppg_o_peaks[i]]
                ax2.plot([ppg_o_peaks[  i], ppg_o_peaks[i]], [PPG[ppg_o_peaks[i]], PPG[ppg_s_peaks[i]]], 'k-')
                ax2.plot([ppg_s_peaks[i], ppg_o_peaks[i]], [PPG[ppg_s_peaks[i]], PPG[ppg_s_peaks[i]]], 'k-')
                ax2.plot([ppg_o_peaks[i], ppg_o_peaks[i]-3], [(PPG[ppg_s_peaks[i]]+PPG[ppg_o_peaks[i]])/2, PPG[ppg_s_peaks[i]]+0.2], 'k-')
                ax2.text(ppg_o_peaks[i]-3, PPG[ppg_s_peaks[i]]+0.22, f'PWA = {amplitude_pwa:.2f}', fontsize=6.5, color='black', ha='center', va='bottom')
                
                #Heart Rate ( BPM )
                ax2.plot([ppg_s_peaks[i], ppg_s_peaks[i]], [PPG[ppg_s_peaks[i]], 1.2], 'k-')
                if(i < len(ppg_o_peaks)-2):
                    heart_rate = (fs/(ppg_s_peaks[i+1] - ppg_s_peaks[i])) * 60
                    ax2.plot([ppg_s_peaks[i], ppg_s_peaks[i+1]],[1.2,1.2],'k-')
                    ax2.text((ppg_s_peaks[i]+ppg_s_peaks[i+1])/2,1, f'HR = {heart_rate:.2f} s', fontsize=6, color='black', ha='center', va='bottom')
            except IndexError:print('List index out of range')
                
           
            
def on_press(event):
    global i

    sys.stdout.flush()
   
    
    lower = i
    upper = i + winsize
    
    winhighlight = np.ones(len(ppg)) * -3
    winhighlight[lower:upper] = 4
    
    
    PPG = ppg_smoothed[lower:upper]
    VPG = compute_derivative(PPG)
    APG = compute_derivative(VPG)
    JPG = compute_derivative(APG)
    SPG = compute_derivative(JPG)
    corrected_vpg = baseline_correction(VPG)
    corrected_apg = baseline_correction(APG)
    corrected_jpg = baseline_correction(JPG)
    corrected_spg = baseline_correction(SPG)
    cutoff_frequency = 5 
    VPG = butter_lowpass_filter(corrected_vpg, cutoff_frequency, fs)
    APG = butter_lowpass_filter(corrected_apg, cutoff_frequency, fs)
    JPG = butter_lowpass_filter(corrected_jpg, cutoff_frequency, fs)
    SPG = butter_lowpass_filter(corrected_spg, cutoff_frequency, fs)
    
    
    
    
    vpg_u_peaks,_ = find_peaks(VPG,prominence=0.008)
    
    zero_amplitude_before_u,zero_amplitude_after_u=findZeroCrossingsFPP(VPG, vpg_u_peaks)
    ppg_s_peaks = zero_amplitude_after_u
    ppg_o_peaks = zero_amplitude_before_u
    
    jpg_p0_peaks,_ = find_peaks(-JPG,prominence=0.0003)
    
    zero_amplitude_before_p0,zero_amplitude_after_p0=findZeroCrossingsFNP(JPG, jpg_p0_peaks)
    apg_a_peaks = zero_amplitude_before_p0
    apg_b_peaks = zero_amplitude_after_p0
    
    apg_e_peaks,jpg_separated_p3_peaks,jpg_merged_p3_peaks,spg_merged_q1_q3_peaks,spg_q1_peaks,spg_q3_peaks = findEKeyPoints(SPG, JPG, jpg_p0_peaks, apg_a_peaks)
    ppg_dicrotic_notch_points = apg_e_peaks

    jpg_p4_peaks,apg_f_peaks,ppg_distolic_peaks,spg_q4_peaks = findFKeyPoints(SPG, JPG, jpg_merged_p3_peaks, jpg_separated_p3_peaks)
    
    
    
    ax1.cla()
    ax1.plot(time, ppg_smoothed, 'b', label='Smoothed PPG',linewidth = 0.75)  # Plot the smoothed signal
    ax1.plot(time, winhighlight, 'r')
    ax1.plot(time[lower:upper], ppg[lower:upper], 'r')
    ax1.grid()
    ax1.legend()
    ax1.set_title('Raw Signal',fontsize=10)
    
    ax2.cla()
    ax2.plot(PPG, 'b',linewidth=1)
    ax2.plot(ppg_s_peaks, PPG[ppg_s_peaks], 'ko',markersize = 5)
    ax2.plot(ppg_o_peaks, PPG[ppg_o_peaks], 'ko',markersize = 5)
    ax2.plot(ppg_dicrotic_notch_points,PPG[ppg_dicrotic_notch_points],'ko',markersize = 5)
    ax2.plot(ppg_distolic_peaks,PPG[ppg_distolic_peaks],'ko',markersize = 5)
    ax2.set_xticks([])
    #ax2.set_ylim(-0.4,1.2)
    ##Adding PPG Points
    for point in ppg_s_peaks:
        ax2.text(point+2, PPG[point] -0.2, 'S', fontsize=8, color='black', ha='center', va='bottom')
         
    for point in ppg_o_peaks:
        ax2.text(point-2, PPG[point] - 0.15, 'O', fontsize=8, color='black', ha='center', va='bottom')
        
    for point in ppg_dicrotic_notch_points:
        ax2.text(point+2, PPG[point]-  0.2, 'N', fontsize=8, color='black', ha='center', va='bottom')
        
    for point in ppg_distolic_peaks:
        ax2.text(point+2, PPG[point] - 0.2, 'D', fontsize=8, color='black', ha='center', va='bottom')
    ##Plotting Time Differences
    PlottingTimeDifferencesOnPPG(PPG, ppg_o_peaks,ppg_s_peaks,ppg_dicrotic_notch_points,ppg_distolic_peaks,lower,upper)


    ax2.grid()
    ax2.set_title('PPG Sliding Window',fontsize=10)
    
    ax3.cla()
    ax3.plot(VPG, 'r',linewidth=1)
    ax3.plot(vpg_u_peaks,VPG[vpg_u_peaks],'ko')
    for point in vpg_u_peaks:
        ax3.text(point, VPG[point] - 0.015, 'u', fontsize=12, color='black', ha='center', va='bottom')
    ax3.set_xticks([])
    ax3.grid()
    ax3.set_title('VPG Sliding Window',fontsize=10)
    
    ax4.cla()
    ax4.plot(APG,'y',linewidth=1)
    ax4.plot(apg_a_peaks,APG[apg_a_peaks],'ko')
    ax4.plot(apg_b_peaks,APG[apg_b_peaks],'ko')
    ax4.plot(apg_e_peaks,APG[apg_e_peaks],'ko')
    ax4.plot(apg_f_peaks,APG[apg_f_peaks],'ko')
    for point in apg_a_peaks:
        ax4.text(point, APG[point] - 0.002, 'a', fontsize=12, color='black', ha='center', va='bottom')
    for point in apg_b_peaks:
        ax4.text(point, APG[point] + 0.0002, 'b', fontsize=12, color='black', ha='center', va='bottom')
    for point in apg_e_peaks:
        ax4.text(point, APG[point] - 0.002, 'e', fontsize=12, color='black', ha='center', va='bottom')
    for point in apg_f_peaks:
        ax4.text(point, APG[point] + 0.0002, 'f', fontsize=12, color='black', ha='center', va='bottom')
    ax4.set_xticks([])
    ax4.grid()
    ax4.set_title('APG Sliding Window',fontsize=10)

    ax5.cla()
    ax5.plot(JPG, 'm',linewidth=1)
    ax5.plot(jpg_p0_peaks,JPG[jpg_p0_peaks], 'ko')
    ax5.plot(jpg_separated_p3_peaks,JPG[jpg_separated_p3_peaks], 'ko')
    ax5.plot(jpg_merged_p3_peaks,JPG[jpg_merged_p3_peaks],'ko')
    ax5.plot(jpg_p4_peaks,JPG[jpg_p4_peaks],'ko')
    ax5.set_xticks([])
    
    for point in jpg_p0_peaks:
        ax5.text(point, JPG[point] + 0.0001, 'p0', fontsize=12, color='black', ha='center', va='bottom')
    for point in jpg_separated_p3_peaks:
        ax5.text(point, JPG[point] - 0.0003, 'p3', fontsize=12, color='black', ha='center', va='bottom')
    for point in jpg_merged_p3_peaks:
        ax5.text(point, JPG[point] - 0.0003, 'p3', fontsize=12, color='black', ha='center', va='bottom')
    for point in jpg_p4_peaks:
        ax5.text(point, JPG[point] + 0.0001, 'p4', fontsize=12, color='black', ha='center', va='bottom')
    ax5.grid()
    ax5.set_title('JPG Sliding Window',fontsize=10)
    
    ax6.cla()
    ax6.plot(SPG, 'g',linewidth=1)
    ax6.plot(spg_merged_q1_q3_peaks,SPG[spg_merged_q1_q3_peaks],'ko')
    ax6.plot(spg_q3_peaks,SPG[spg_q3_peaks],'ko')
    ax6.plot(spg_q4_peaks,SPG[spg_q4_peaks],'ko')
    for point in spg_merged_q1_q3_peaks:
        ax6.text(point, SPG[point] - 0.00003, 'q3', fontsize=12, color='black', ha='center', va='bottom')        
    for point in spg_q3_peaks:
        ax6.text(point, SPG[point] - 0.00003, 'q3', fontsize=12, color='black', ha='center', va='bottom')
    for point in spg_q4_peaks:
        ax6.text(point, SPG[point] + 0.00003, 'q4', fontsize=12, color='black', ha='center', va='bottom')
        
    ax6.grid()
    ax6.set_title('SPG Sliding Window',fontsize=10)

    
    if event.key == 'right':
        i = i + winhop
        fig.canvas.draw()
    elif event.key == 'left':
        i = i - winhop
        fig.canvas.draw()


    



fig = plt.figure(figsize=(20,30))
fig.subplots_adjust(top=1,bottom=0.03,left=0.045,right=0.99,hspace=0.5)  

ax1 = fig.add_subplot(611)
ax1.grid()

ax2 = fig.add_subplot(612)
ax2.grid()

ax3 = fig.add_subplot(613)
ax3.grid()

ax4 = fig.add_subplot(614)
ax4.grid()

ax5 = fig.add_subplot(615)
ax5.grid()

ax6 = fig.add_subplot(616)
ax6.grid()

fig.canvas.mpl_connect('key_press_event', on_press)


plt.show()

