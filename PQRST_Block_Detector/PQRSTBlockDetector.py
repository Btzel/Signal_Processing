# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:11:15 2023

@author: BurakT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks

def derivative_filter(signal):
    return np.diff(signal)

ecg_signal = electrocardiogram()
fs = 360  #Frequency

#Lower and Upper bounds
lower = 2150
upper = 3050
#Signal time
time = np.arange(ecg_signal.size) / fs

from scipy.signal import windows

def gaussian_smooth(data, sigma=2):
    window = windows.gaussian(5*sigma+5, sigma)
    smoothed_data = np.convolve(data, window, mode='same') / window.sum()
    return smoothed_data
#Smoothing the signal
smoothed_ecg_signal = gaussian_smooth(ecg_signal)

def baseline_correction(signal, window_size):
    rolling_mean = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    corrected_signal = signal - rolling_mean
    return corrected_signal
#Baseline correction
smoothed_ecg_signal = baseline_correction(smoothed_ecg_signal, window_size=100)
#Smoothed signal time
time_secg = np.arange(len(smoothed_ecg_signal)) / fs

#Function to detect R peaks in a signal
def detect_r_peaks(signal, threshold, neighborhood=5):
    peaks, _ = find_peaks(signal, height=threshold)
    filtered_peaks = []
    for peak in peaks:
        if len(filtered_peaks) == 0 or peak - filtered_peaks[-1] > neighborhood:
            filtered_peaks.append(peak)
        elif signal[peak] > signal[filtered_peaks[-1]]:
            filtered_peaks[-1] = peak
    return np.array(filtered_peaks)


threshold_filtered = 0.15 * max(smoothed_ecg_signal)
r_peaks_secg = detect_r_peaks(smoothed_ecg_signal, threshold_filtered, neighborhood=120)

#Lower and Upper Bound in time (limiting with it)
time_lower = lower / fs
time_upper = upper / fs

#Detecting other keypoints
def detect_p_q_s_t_points(signal, r_peaks, fs):
    p_points = []
    q_points = []
    s_points = []
    t_points = []

    for r_peak in r_peaks:
        
        search_window_p_start = max(0, r_peak - int(0.2 * fs))  
        search_window_p_end = min(r_peak, r_peak - int(0.07 * fs))
        
        p_point = np.argmax(signal[search_window_p_start:search_window_p_end]) + search_window_p_start

        p_points.append(p_point)
        
        search_window_start = max(0, r_peak - int(0.04* fs))
        search_window_end = min(len(signal), r_peak + int(0.04 * fs))

        q_point = np.argmin(signal[search_window_start:r_peak]) + search_window_start
        s_point = np.argmin(signal[r_peak:search_window_end]) + r_peak

        search_window_t_start = s_point
        search_window_t_end = min(len(signal), s_point + int(0.2 * fs))

        t_point = np.argmax(signal[search_window_t_start:search_window_t_end]) + search_window_t_start

        q_points.append(q_point)
        s_points.append(s_point)
        t_points.append(t_point)

    return np.array(p_points), np.array(q_points), np.array(s_points), np.array(t_points)

#Finding Keypoint times between lower and upper bound times
def find_keypoint_times(p_point,q_point,r_point,s_point,t_point,fs,lower,upper):
    p_point_times = []
    q_point_times = []
    r_point_times = []
    s_point_times = []
    t_point_times = []
    
    for point in p_point:
        point_time = time_lower + point/fs
        if((point >= lower) & (point <= upper)):
            p_point_times.append(point_time)
            
        
    for point in q_point:
        point_time = time_lower + point/fs
        if((point >= lower) & (point <= upper)):
            q_point_times.append(point_time)
        
    for point in r_point:
        point_time = time_lower + point/fs
        if((point >= lower) & (point <= upper)):
            r_point_times.append(point_time)
        
    for point in s_point:
        point_time = time_lower + point/fs
        if((point >= lower) & (point <= upper)):
            s_point_times.append(point_time)
        
    for point in t_point:
        point_time = time_lower + point/fs
        if((point >= lower) & (point <= upper)):
            t_point_times.append(point_time)
    point_iter = len(p_point_times)
    if((len(p_point_times) == point_iter) & (len(q_point_times) == point_iter) & (len(r_point_times) == point_iter) & (len(s_point_times) == point_iter) & (len(t_point_times) == point_iter)):
        return p_point_times,q_point_times,r_point_times,s_point_times,t_point_times,point_iter
#Calculating intervals depending on keypoint times
def calculate_intervals():
    p_point_times,q_point_times,r_point_times,s_point_times,t_point_times,point_iter = find_keypoint_times(p_points, q_points, r_peaks_secg, s_points, t_points, fs, lower, upper)
    for i in range(0, point_iter):
        
        print("********************************")
        print("** BLOCK {0} PQ INTERVAL: ".format(i+1) + f"{q_point_times[i]-p_point_times[i]:0.3f} **")
        print("** BLOCK {0} QR INTERVAL: ".format(i+1) + f"{r_point_times[i]-q_point_times[i]:0.3f} **")
        print("** BLOCK {0} RS INTERVAL: ".format(i+1) + f"{s_point_times[i]-r_point_times[i]:0.3f} **")
        print("** BLOCK {0} ST INTERVAL: ".format(i+1) + f"{t_point_times[i]-s_point_times[i]:0.3f} **")
        print("** BLOCK {0} PT INTERVAL: ".format(i+1) + f"{t_point_times[i]-p_point_times[i]:0.3f} **")
        print("********************************")
        
    for i in range(0, point_iter-1):
        print("*******************************************")
        print("** BLOCK {0} TO BLOCK {1} PP INTERVAL: ".format(i+1, i+2) + f"{p_point_times[i+1]-p_point_times[i]:0.3f} **")
        print("** BLOCK {0} TO BLOCK {1} QQ INTERVAL: ".format(i+1, i+2) + f"{q_point_times[i+1]-q_point_times[i]:0.3f} **")
        print("** BLOCK {0} TO BLOCK {1} TP INTERVAL: ".format(i+1, i+2) + f"{p_point_times[i+1]-t_point_times[i]:0.3f} **")
        print("*******************************************")
#using detecting keypoints function
p_points,q_points,s_points, t_points = detect_p_q_s_t_points(smoothed_ecg_signal , r_peaks_secg, fs)

#using calcule intervals
calculate_intervals()


# Plotting the Raw ECG-Signal
plt.figure(figsize=(20, 10))

# Plot the ECG Signal
plt.subplot(2, 1, 1)
plt.plot(time, ecg_signal)
plt.title('ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(time_lower, time_upper)

# Ploting the Smoothed ECG-Signal
plt.subplot(2, 1, 2)
plt.plot(time_secg, smoothed_ecg_signal)
plt.plot(r_peaks_secg/fs, smoothed_ecg_signal[r_peaks_secg], 'bx')  # Mark R peaks in blue x
plt.title('Smoothed ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(time_lower, time_upper)

plt.tight_layout()
plt.show()


# Plot the Smoothed ECG Signal with Q,R,S Keypoints
plt.figure(figsize=(20, 5))
plt.plot(time_secg, smoothed_ecg_signal)
plt.plot(r_peaks_secg/fs, smoothed_ecg_signal[r_peaks_secg], 'ro')  # Mark R peaks in red
plt.plot(q_points/fs, smoothed_ecg_signal[q_points], 'go')  # Mark Q points in green
plt.plot(s_points/fs, smoothed_ecg_signal[s_points], 'bo')  # Mark S points in blue
plt.title(r'Smoothed ECG Signal With'+ 
          r'\textcolor{green}{Q} '+ 
          r'\textcolor{red}{R} '+r'and '+r'\textcolor{blue}{S} ' )
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(time_lower, time_upper)

plt.tight_layout()
plt.show()


# Plot the Smoothed ECG Signal with P,Q,R,S and T points
plt.figure(figsize=(15, 4))
plt.plot(time_secg, smoothed_ecg_signal)
plt.plot(r_peaks_secg/fs, smoothed_ecg_signal[r_peaks_secg], 'ro')  # Mark R peaks in red
plt.plot(q_points/fs, smoothed_ecg_signal[q_points], 'go')  # Mark Q points in green
plt.plot(s_points/fs, smoothed_ecg_signal[s_points], 'bo')  # Mark S points in blue
plt.plot(p_points/fs, smoothed_ecg_signal[p_points], 'mo')  # Mark P points in magenta
plt.plot(t_points/fs, smoothed_ecg_signal[t_points], 'yo')  # Mark P points in magenta
plt.title('Smoothed ECG Signal With P, Q, R, S, T Points')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(time_lower, time_upper)

plt.tight_layout()
plt.show()