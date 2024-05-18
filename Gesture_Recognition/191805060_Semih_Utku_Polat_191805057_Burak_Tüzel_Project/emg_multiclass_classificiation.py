# 191805060 Semih Utku Polat
# 191805057 Burak Tüzel
# Libraries
import matplotlib.pylab as plt
import os
import pandas as pd
import numpy as np
import time
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.inspection import permutation_importance


# Paths
main_path = 'C:/Users/SUP/Desktop/bio'
dataset_path = 'C:/Users/SUP/Desktop/bio/EMG_data_for_gestures-master/'
window_size_features_path = 'C:/Users/SUP/Desktop/bio/window_size_features'
window_hop_features_path = 'C:/Users/SUP/Desktop/bio/window_hop_features'
best_model_path = 'C:/Users/SUP/Desktop/bio/best_model.pkl'
predicted_dataset_path = 'C:/Users/SUP/Desktop/bio/predicted_dataset.csv'


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

all_data = all_data.dropna()

print('\nCount of total rows:', len(all_data))

'''
Classes
0 - unmarked data,
1 - hand at rest, 
2 - hand clenched in a fist, 
3 - wrist flexion,
4 – wrist extension,
5 – radial deviations,
6 - ulnar deviations,
7 - extended palm (the gesture was not performed by all subjects).
'''
'''
# Removing rows with class 0 and 7. Using only classes with 1 to 6
labels_to_remove = [0, 7]
all_data = all_data[~all_data['class'].isin(labels_to_remove)]

print('\nCount of rows without class 0 and 7:', len(all_data), '\n')

# 4.237.907 total rows
# 2.725.157 rows with class 0
# 13.696 rows with class 7

# 1.499.054 rows without class 0 and 7
'''

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


    return mean, variance, std_deviation, root_mean_square, max_value, kurtosis, skewness, kurtosis_energy_operator, absolute_media, cpt1, cpt2, cpt3, cpt4, cpt5, cpt6, fifth_statistic_moment, shape_factor, impulse_factor, clearance_factor, delta_rms, root_sum_of_squares, energy, latitude_factor, weighted_ssr_absolute, mean_square_error, normalized_normal_negative_likelihood, mean_deviation, std_deviation_impulse_factor, log_log_ratio, kth_central_moment, histogram_lower_bound, histogram_upper_bound, normalized_moment, sure_entropy, norm_entropy, peak_to_peak, minimum_value, peak_value, sixth_statistical_moment, crest_factor, integrated_signal, square_root_amplitude_value, simple_square_integral, zero_crossing, wavelength, wilson_amplitude, slope_sign_change, modified_mean_absolute_value_1, modified_mean_absolute_value_2, mean_absolute_value_slope, mean_of_amplitude, log_rms, conduction_velocity_signal, average_amplitude_change, v_order_2, v_order_3, maximum_fractal_length, difference_absolute_standard_deviation, myopulse_percentage_rate, higher_order_temporal_moments, difference_absolute_variance_value, waveform_indicators, pulse_indicators

def getFeatures(all_data, window_size, window_hop, window_features_path):
    
    features_list = []
    
    for i in range(0, len(all_data), window_hop):
        
        selmat = all_data.iloc[i:i+window_size, 1:-1].to_numpy().flatten()
        mean, variance, std_deviation, root_mean_square, max_value, kurtosis, skewness, kurtosis_energy_operator, absolute_media, cpt1, cpt2, cpt3, cpt4, cpt5, cpt6, fifth_statistic_moment, shape_factor, impulse_factor, clearance_factor, delta_rms, root_sum_of_squares, energy, latitude_factor, weighted_ssr_absolute, mean_square_error, normalized_normal_negative_likelihood, mean_deviation, std_deviation_impulse_factor, log_log_ratio, kth_central_moment, histogram_lower_bound, histogram_upper_bound, normalized_moment, sure_entropy, norm_entropy, peak_to_peak, minimum_value, peak_value, sixth_statistical_moment, crest_factor, integrated_signal, square_root_amplitude_value, simple_square_integral, zero_crossing, wavelength, wilson_amplitude, slope_sign_change, modified_mean_absolute_value_1, modified_mean_absolute_value_2, mean_absolute_value_slope, mean_of_amplitude, log_rms, conduction_velocity_signal, average_amplitude_change, v_order_2, v_order_3, maximum_fractal_length, difference_absolute_standard_deviation, myopulse_percentage_rate, higher_order_temporal_moments, difference_absolute_variance_value, waveform_indicators, pulse_indicators = extractFeatures(selmat)
        
        
        ch1mean = all_data.iloc[i:i+window_size,1].mean()
        ch2mean = all_data.iloc[i:i+window_size,2].mean()
        ch3mean = all_data.iloc[i:i+window_size,3].mean()
        ch4mean = all_data.iloc[i:i+window_size,4].mean()
        ch5mean = all_data.iloc[i:i+window_size,5].mean()
        ch6mean = all_data.iloc[i:i+window_size,6].mean()
        ch7mean = all_data.iloc[i:i+window_size,7].mean()
        ch8mean = all_data.iloc[i:i+window_size,8].mean()
        
        
        # flabel: the most frequent class
        bincountlist = np.bincount(all_data.iloc[i:i+window_size, -1].to_numpy(dtype='int64'))
        most_frequent_class = bincountlist.argmax()
        label = most_frequent_class
        
        # fpercent: the percentage of the most frequent class
        percentage_most_frequent=bincountlist[most_frequent_class] / len(all_data.iloc[i:i+window_size, -1].to_numpy(dtype='int64'))
        percent = percentage_most_frequent
        
        # flabel2: the second most frequent class 
        if percentage_most_frequent == 1.0:
            most_frequent_class2 = most_frequent_class
        else:
            bincountlist[most_frequent_class] = 0
            most_frequent_class2=bincountlist.argmax()
            
        label2 = most_frequent_class2
        
        
        features_list.append({
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
            'pulse_indicators': pulse_indicators,
            
            'ch1mean': ch1mean,
            'ch2mean': ch2mean,
            'ch3mean': ch3mean,
            'ch4mean': ch4mean,
            'ch5mean': ch5mean,
            'ch6mean': ch6mean,
            'ch7mean': ch7mean,
            'ch8mean': ch8mean,
            
            'label': label,
            'percent': percent,
            '2ndlabel': label2
        })
        

    # Save the features
    rdf = pd.DataFrame(features_list)
    rdf.to_csv(f'{window_features_path}/emg_gesture_ws{window_size}_hop{window_hop}.csv', index = None, header = True)
    print(f'Created: emg_gesture_ws{window_size}_hop{window_hop}.csv')
    # rdf.info()
    
    return rdf


def plotBar(title, data_results, data, xlabel, ylabel, save_path, figsize = (10, 6), bar_color = 'blue', bar_width = 0.8):
    plt.figure(figsize = figsize)
    plt.title(title)
    plt.bar(range(len(data_results)), data_results, tick_label = data, color = bar_color, width = bar_width)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(save_path)
    plt.show()



# Optimization of best window size
window_size_times = []
window_size_results = []
window_size_performance = []
window_sizes = [50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

for window_size in window_sizes:
    start_time = time.time()
    window_hop = 50
    
    
    rdf = getFeatures(all_data, window_size, window_hop, window_size_features_path)
    
    X = rdf.drop(columns = ['label', 'percent', '2ndlabel']) # Features
    y = rdf['label'] # Class
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 50605057)
    
    rf_classifier = RandomForestClassifier()
    scores = cross_val_score(rf_classifier, X_train, y_train, cv = 10)
    mean_accuracy_score = np.mean(scores)


    total_time = time.time() - start_time
    window_size_times.append(total_time)
    window_size_results.append(mean_accuracy_score)
    window_size_performance.append(mean_accuracy_score / total_time)
    
    print('Window Size:', window_size, 'Window Hop:', window_hop, 'Mean Accuracy Score:', mean_accuracy_score)
    print('')

# Visualize the window size results
plotBar("Window sizes' mean accuracy score", window_size_results, window_sizes, 'Window Sizes', 'Mean Accuracy Score', f'{window_size_features_path}/mean_accuracy_score.png')
plotBar("Window sizes' training time", window_size_times, window_sizes, 'Window Sizes', 'Training Time', f'{window_size_features_path}/time.png')
plotBar("Window sizes' mean accuracy score / training time", window_size_performance, window_sizes, 'Window Sizes', 'Performance (Mean Accuracy Score / Training Time)', f'{window_size_features_path}/mean_accuracy_score_time.png')

# Find the best window size
best_window_size = window_sizes[np.argmax(window_size_results)]
print('Best window size:', best_window_size)
print('')


# best_window_size = 1000
# Optimization of best hop/stride size for the best window size
window_hop_times = []
window_hop_results = []
window_hop_performance = []
window_hops = [50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

for window_hop in window_hops:
    start_time = time.time()


    rdf = getFeatures(all_data, best_window_size, window_hop, window_hop_features_path)
    
    X = rdf.drop(columns = ['label', 'percent', '2ndlabel']) # Features
    y = rdf['label'] # Class
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 50605057)
    
    rf_classifier = RandomForestClassifier()
    scores = cross_val_score(rf_classifier, X_train, y_train, cv = 10)
    mean_accuracy_score = np.mean(scores)


    total_time = time.time() - start_time
    window_hop_times.append(total_time)
    
    window_hop_results.append(mean_accuracy_score)
    window_hop_performance.append(mean_accuracy_score / total_time)
    
    print('Window Size:', best_window_size, 'Window Hop:', window_hop, 'Mean Accuracy Score:', mean_accuracy_score)
    print('')

# Visualize the window hop results
plotBar("Stride sizes' mean accuracy score", window_hop_results, window_hops, 'Window Strides', 'Mean Accuracy Score', f'{window_hop_features_path}/mean_accuracy_score.png')
plotBar("Stride sizes' training time", window_hop_times, window_hops, 'Window Strides', 'Training Time', f'{window_hop_features_path}/time.png')
plotBar("Stride sizes' mean accuracy score / training time", window_hop_performance, window_hops, 'Window Strides', 'Performance (Mean Accuracy Score / Training Time)', f'{window_hop_features_path}/mean_accuracy_score_time.png')

# Find the best window hop for the best window size
best_window_hop = window_hops[np.argmax(window_hop_results)]
print('Best window size:', best_window_size, 'Best window hop:', best_window_hop)
print('')


best_window_size = 1000
best_window_hop = 50
# Get the best window size and its hop
best_rdf = pd.read_csv(f'{window_hop_features_path}/emg_gesture_ws{best_window_size}_hop{best_window_hop}.csv')

X = best_rdf.drop(columns = ['label', 'percent', '2ndlabel']) # Features
y = best_rdf['label'] # Class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 50605057)

rf_classifier = RandomForestClassifier(n_estimators = 100)
rf_classifier.fit(X_train, y_train)


# Get feature importances
importances = rf_classifier.feature_importances_
importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importances_df = importances_df.sort_values(by = 'Importance', ascending = False)

print("Feature importances with a forest of trees:")
for index, row in importances_df.iterrows():
    print(row['Feature'], row['Importance'])
    

top_n = 15
top_features = importances_df.head(top_n)
other_features = importances_df.iloc[top_n:]

plt.figure(figsize=(12, 6))
plt.bar(range(len(top_features)), top_features['Importance'], tick_label=top_features['Feature'], label='Top Features')
plt.bar(len(top_features), other_features['Importance'].sum(), label='Other Features', color='gray')
plt.title(f"Top {top_n} Feature Importances with a Forest of Trees")
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(f'{main_path}/feature_importances.png')
plt.show()


# Feature importances with a permutation
result = permutation_importance(rf_classifier, X_train, y_train, n_repeats = 5, scoring = "accuracy", random_state = 50605057, n_jobs = 5)

# Sort permutation importances by mean in descending order
sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(result.importances[sorted_importances_idx].T, columns = X.columns[sorted_importances_idx])

top_n = 15
top_features_permutation = importances.head(top_n)

plt.figure(figsize=(12, 6))
plt.barh(top_features_permutation.columns, top_features_permutation.mean(), xerr=top_features_permutation.std(), align='center')
plt.title(f"Top {top_n} Feature Importances (Permutation)")
plt.xlabel("Mean Decrease in Accuracy Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig(f'{main_path}/feature_importances_permutation.png')

# Select the best 10 features and their importance values from permutation importances
best_features_permutation = importances.mean().nlargest(10).index
best_importances_permutation = importances[best_features_permutation].mean()

# Print each feature along with its importance value
print('')
print("Feature importances from permutation importances:")
for feature, importance in best_importances_permutation.items():
    print(f"{feature}: {importance}")
print('')

# Extract the names of the best features
best_feature_names = best_features_permutation.tolist()

# Create X and split data again, keeping only the best features
X = X[best_feature_names]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 50605057)


def initializeModels():
    return {
        'Random Forest': RandomForestClassifier(n_estimators = 100),
        'AdaBoost': AdaBoostClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'k-Nearest Neighbors': KNeighborsClassifier(),
        'Multi-layer Perceptron': MLPClassifier()
    }

def crossValidationAndSaveBestModel(models, X_train, y_train):
    results = {}
    acc_scores = []
    model_names = []

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv = 10)
        results[model] = scores
        acc_scores.append(np.mean(scores))
        model_names.append(name)
        
        print('Model:', name, 'Mean Accuracy Scores:', np.mean(scores))
    
    plotBar('Mean Accuracy Scores', acc_scores, model_names, 'Models', 'Mean Accuracy Score', f'{main_path}/models.png')
    
    best_model = max(results, key = lambda x: np.mean(results[x]))
    best_model.fit(X_train, y_train)
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model, f)

    return best_model, model_names[np.argmax(acc_scores)]


models = initializeModels()
best_model, best_model_name = crossValidationAndSaveBestModel(models, X_train, y_train)

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('')
print(f'Best Model ({best_model_name}) Accuracy:', accuracy)



# with open(best_model_path, 'rb') as f:
#     best_model = pickle.load(f)
# X = all_data.iloc[:, 1:-1]
# predicted_class = best_model.predict(X)
# all_data['predicted_class'] = predicted_class
# all_data.to_csv(predicted_dataset_path, index = None, header = True)


# ranrow=np.random.randint(0,len(y_test))
# pckl_input=X_test.iloc[ranrow,:]
# pckl_label=y_test.iloc[ranrow]
# prediction = best_model.predict([pckl_input])
# print("Actual:",pckl_label,"Predicted:",prediction[0])  