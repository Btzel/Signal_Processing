# 191805060 Semih Utku Polat
# 191805057 Burak Tüzel
# Libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import matplotlib.patches as patches

# Paths
dataset_path = 'C:/Users/SUP/Desktop/bio/EMG_data_for_gestures-master/'
best_model_path = 'C:/Users/SUP/Desktop/bio/best_model.pkl'


# Get data from dataset
folders = [folder.name for folder in os.scandir(dataset_path) if folder.is_dir()]

all_data = pd.DataFrame()

for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    files = [file.name for file in os.scandir(folder_path) if file.is_file()]
    print(folder, files)
    for file in files:
        file_path = os.path.join(folder_path, file)
        current_data = pd.read_csv(file_path, sep = '\t')
        all_data = pd.concat([all_data, current_data])

rawsignals = all_data.dropna()
rawsignals = rawsignals.to_numpy()

with open(best_model_path, 'rb') as f:
    best_model = pickle.load(f)
    
features_of_interest = ["kurtosis_energy_operator", "zero_crossing", "norm_entropy", "modified_mean_absolute_value_2", 
                        "modified_mean_absolute_value_1", "ch5mean", "weighted_ssr_absolute", "kurtosis", "ch8mean", "mean_deviation"]


emgch0 = rawsignals[:, 1]
emgch1 = rawsignals[:, 2]
emgch2 = rawsignals[:, 3]
emgch3 = rawsignals[:, 4]
emgch4 = rawsignals[:, 5]
emgch5 = rawsignals[:, 6]
emgch6 = rawsignals[:, 7]
emgch7 = rawsignals[:, 8]

outclass = rawsignals[:, 9]
lensignal = len(emgch0)
fs = 1000
time = np.arange(lensignal) / fs

winsize = 1000
winhop = 500
i = 0

def on_press(event):
    global i
    sys.stdout.flush()

    lower = i
    upper = i + winsize

    ax1.cla()
    ax1.plot(time, emgch0, 'g')
    ax1.plot(time, outclass, 'r:', linewidth=2, alpha=0.5)
    ax1.grid()
    ax1.set_title(f'Raw Signals - Actual Label: {int(outclass[lower])}')

    ax1.add_patch(patches.Rectangle((time[lower], ax1.get_ylim()[0]), winsize/fs, ax1.get_ylim()[1] - ax1.get_ylim()[0], linewidth=2, edgecolor='g', facecolor='none'))



    selmat = all_data.iloc[lower:upper, 1:-1].to_numpy().flatten()
    features_dict  = extractFeatures(selmat)
    ch_means = all_data.iloc[lower:upper, 1:9].mean().to_numpy()
    features_dict.update({
        f'ch{i}mean': ch_means[i-1] for i in range(1, 9)
    })
    
    selected_features = [features_dict[feature] for feature in features_of_interest]
    selected_features_df = pd.DataFrame([selected_features], columns=features_of_interest)
    

    y = outclass[lower:upper]

    ax2.cla()
    for x_slice in [emgch0[lower:upper], emgch1[lower:upper], emgch2[lower:upper], emgch3[lower:upper], emgch4[lower:upper], emgch5[lower:upper], emgch6[lower:upper], emgch7[lower:upper]]:
        ax2.plot(x_slice)

    # Predict using the best model
    predicted_label = best_model.predict(selected_features_df)

    ax2.plot(y, 'r:', linewidth=2, alpha=0.5)
    ax2.grid()
    ax2.set_title(f'Sliding Window - Predicted Label: {int(predicted_label)}')
    
    if event.key == 'right':
        i += winhop
        fig.canvas.draw()
    elif event.key == 'left':
        i -= winhop
        fig.canvas.draw()

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(time, emgch0, 'g')
ax1.plot(time, outclass, 'r:', linewidth=2, alpha=0.5)
ax1.grid()
ax1.set_title('Raw EMG Signal')

ax2 = fig.add_subplot(212)
ax2.grid()

fig.canvas.mpl_connect('key_press_event', on_press)

plt.show()


# Feature operations
def extractFeatures(x):
    N = len(x)
    
    mean = np.mean(x)
    variance = np.var(x)
    std_deviation = np.std(x)
    root_mean_square = np.sqrt(np.mean(x**2))
    max_value = np.max(x)
    kurtosis = np.sum((x - mean)**4) / (N * std_deviation**4)
    skewness = np.sum((x - mean)**3) / (N * std_deviation**3)
    kurtosis_energy_operator = np.sum(np.diff(x)**4) / (N**2 * np.mean(np.diff(x)**2)**2)
    absolute_media = np.mean(np.abs(x))
    cpt1 = np.max(np.abs(x)) / np.sqrt(root_mean_square)
    cpt2 = np.max(np.abs(x)) / np.sqrt(np.mean(x**2))
    cpt3 = np.max(np.abs(x)) / np.mean(np.abs(x))
    cpt4 = np.sum(np.log(np.abs(x) + 1)) / (N * np.log(std_deviation + 1))
    cpt5 = np.sum(np.exp(x)) / (N * np.exp(std_deviation))
    cpt6 = np.sum(np.sqrt(np.abs(x))) / (N * np.sqrt(variance))
    fifth_statistic_moment = np.sum((x - mean)**5)
    shape_factor = root_mean_square / np.mean(np.abs(x))
    impulse_factor = np.max(np.abs(x)) / (1 / N * np.sum(np.abs(x)))
    clearance_factor = np.max(np.abs(x)) / (1 / N * np.sum(x**2))
    delta_rms = np.sqrt(np.mean(np.diff(x)**2))
    root_sum_of_squares = np.sqrt(np.sum(x**2))
    energy = np.sum(x**2)
    latitude_factor = np.max(np.abs(x)) / (1 / N * np.sum(np.sqrt(np.abs(x))))
    weighted_ssr_absolute = 1 / N * np.sum(np.sqrt(np.abs(x)))**2
    # pulse_index = np.max(x) / np.mean(x)
    mean_square_error = np.mean((x - mean)**2)
    normalized_normal_negative_likelihood = np.log(std_deviation / root_mean_square)
    mean_deviation = np.sum(np.abs(x - mean)) / (N * np.sqrt(variance))
    std_deviation_impulse_factor = std_deviation / np.mean(np.abs(x))
    log_log_ratio = 1 / np.log(std_deviation) * np.sum(np.log(np.abs(x) + 1))
    kth_central_moment = np.mean((x - np.mean(x))**3) # K is set to 3
    histogram_lower_bound = np.min(x) - 0.5 * (np.max(x) - np.min(x)) / (N - 1)
    histogram_upper_bound = np.max(x) + 0.5 * (np.max(x) - np.min(x)) / (N - 1)
    normalized_moment = np.sum((x - np.mean(x))**5) / np.sqrt(np.sum((x - np.mean(x))**2)**5)
    # shannon_entropy = -np.sum(np.log(x**2))
    # log_energy_entropy = np.sum(np.log(x**2))
    # threshold_entropy = np.where(np.abs(x) > 0.2, 1, 0)
    sure_entropy = len(x) - np.count_nonzero(np.abs(x) <= 0.2) + np.sum(np.minimum(x**2, 0.2**2))
    norm_entropy = np.sum(np.abs(x)**0.2)
    peak_to_peak = np.max(x) - np.min(x)
    minimum_value = np.min(x)
    peak_value = 0.5 * (np.max(x) - np.min(x))
    sixth_statistical_moment = np.sum((x - mean)**6)
    crest_factor = np.max(np.abs(x)) / root_mean_square
    integrated_signal = np.sum(np.abs(x))
    square_root_amplitude_value = (np.sum(np.sqrt(np.abs(x))) / N)**2
    simple_square_integral = np.sum(np.abs(x)**2)
    zero_crossing = np.sum(np.sign(-x[:-1] * x[1:]))
    wavelength = np.sum(np.abs(np.diff(x)))
    wilson_amplitude = np.sum(np.where(np.abs(np.diff(x)) - 0.2 > 0, 1, 0))
    slope_sign_change = np.sum(np.where((x[1:] - x[:-1]) * (x[1:] - np.roll(x, 1)[1:]) >= 0, 1, 0))
    # log_detector = np.exp(np.mean(np.log(np.abs(x))))
    modified_mean_absolute_value_1 = np.mean(np.where((0.25 * N <= np.arange(N)) & (np.arange(N) <= 0.75 * N), 1, 0) * np.abs(x))
    modified_mean_absolute_value_2 = np.mean(np.where((0.25 * N <= np.arange(N)) & (np.arange(N) <= 0.75 * N), 1, 0) * np.abs(x))
    mean_absolute_value_slope = np.mean(np.diff(x))
    mean_of_amplitude = np.sum(np.abs(np.diff(x)))
    log_rms = np.log(np.sqrt(np.sum(x**2)))
    conduction_velocity_signal = 1 / (N - 1) * np.sum(x**2)
    average_amplitude_change = 1 / N * np.sum(np.diff(x)**2)
    v_order_2 = np.sqrt(1 / N * np.sum(x**2))
    v_order_3 = np.cbrt(1 / N * np.sum(x**3))
    maximum_fractal_length = np.log10(np.sum(np.abs(np.diff(x))))
    difference_absolute_standard_deviation = np.sqrt(1 / (N - 1) * np.sum(np.diff(x)**2))
    myopulse_percentage_rate = np.mean(np.where(x >= 0.2, 1, 0))
    higher_order_temporal_moments = np.mean(np.abs(x)**3)
    difference_absolute_variance_value = 1 / (N - 2) * np.sum(np.diff(x)**2)
    # margin_index = (np.max(x) / np.sqrt(1 / N * np.sum(x)))**2
    waveform_indicators = np.sum(x) / N
    # weibull_negative_log_likelihood = -np.sum(np.log((0.2 * 1)**5 - np.sign(x) * x))
    pulse_indicators = np.max(x) / (1 / N * np.sum(np.abs(x)))


    features  = {
        'mean': mean,
        'variance': variance,
        'std': std_deviation,
        'root_mean_square': root_mean_square,
        'max': max_value,
        'kurtosis': kurtosis,
        'skewness': skewness,
        'kurtosis_energy_operator': kurtosis_energy_operator,
        'absolute_media': absolute_media,
        'cpt1': cpt1,
        'cpt2': cpt2,
        'cpt3': cpt3,
        'cpt4': cpt4,
        'cpt5': cpt5,
        'cpt6': cpt6,
        'fifth_statistic_moment': fifth_statistic_moment,
        'shape_factor': shape_factor,
        'impulse_factor': impulse_factor,
        'clearance_factor': clearance_factor,
        'delta_rms': delta_rms,
        'root_sum_of_squares': root_sum_of_squares,
        'energy': energy,
        'latitude_factor': latitude_factor,
        'weighted_ssr_absolute': weighted_ssr_absolute,
        # 'pulse_index': pulse_index,
        'mean_square_error': mean_square_error,
        'normalized_normal_negative_likelihood': normalized_normal_negative_likelihood,
        'mean_deviation': mean_deviation,
        'std_deviation_impulse_factor': std_deviation_impulse_factor,
        'log_log_ratio': log_log_ratio,
        'kth_central_moment': kth_central_moment,
        'histogram_lower_bound': histogram_lower_bound,
        'histogram_upper_bound': histogram_upper_bound,
        'normalized_moment': normalized_moment,
        # 'shannon_entropy': shannon_entropy,
        # 'log_energy_entropy': log_energy_entropy,
        # 'threshold_entropy': threshold_entropy,
        'sure_entropy': sure_entropy,
        'norm_entropy': norm_entropy,
        'peak_to_peak': peak_to_peak,
        'minimum_value': minimum_value,
        'peak_value': peak_value,
        'sixth_statistical_moment': sixth_statistical_moment,
        'crest_factor': crest_factor,
        'integrated_signal': integrated_signal,
        'square_root_amplitude_value': square_root_amplitude_value,
        'simple_square_integral': simple_square_integral,
        'zero_crossing': zero_crossing,
        'wavelength': wavelength,
        'wilson_amplitude': wilson_amplitude,
        'slope_sign_change': slope_sign_change,
        # 'log_detector': log_detector,
        'modified_mean_absolute_value_1': modified_mean_absolute_value_1,
        'modified_mean_absolute_value_2': modified_mean_absolute_value_2,
        'mean_absolute_value_slope': mean_absolute_value_slope,
        'mean_of_amplitude': mean_of_amplitude,
        'log_rms': log_rms,
        'conduction_velocity_signal': conduction_velocity_signal,
        'average_amplitude_change': average_amplitude_change,
        'v_order_2': v_order_2,
        'v_order_3': v_order_3,
        'maximum_fractal_length': maximum_fractal_length,
        'difference_absolute_standard_deviation': difference_absolute_standard_deviation,
        'myopulse_percentage_rate': myopulse_percentage_rate,
        'higher_order_temporal_moments': higher_order_temporal_moments,
        'difference_absolute_variance_value': difference_absolute_variance_value,
        # 'margin_index': margin_index,
        'waveform_indicators': waveform_indicators,
        # 'weibull_negative_log_likelihood': weibull_negative_log_likelihood,
        'pulse_indicators': pulse_indicators
    }
    
    return features 