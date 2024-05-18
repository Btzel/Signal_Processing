# Libraries
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import pickle


# Paths
path_spec_db = 'sounds/spec_db.csv'
path_labels = 'sounds/labels.csv'
path_scores = 'scores/'
path_confusion_matrix = 'confusion_matrix/'
path_models = 'models/'
path_plots = 'plots/'
path_best_model = 'best_model.sav'


seed = 50605057


# Get data
audio_signal = np.loadtxt(path_spec_db, delimiter=',').transpose()
audio_labels = np.loadtxt(path_labels, ndmin = 2).reshape(-1)


# Splitting
X_train, X_test, y_train, y_test = train_test_split(audio_signal, audio_labels, test_size = 0.25, random_state = seed)


# Models
models = []
models.append(('RF', RandomForestClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('SVC', SVC()))


# Confusion matrix labels
labels_y = ['wait', 'computer', 'engineering', 'name', 'surname', 'cough', 'clap_hands', 'snap_finger']


# To store outputs
model_names = []

mean_cv_scores = []
accuracy_scores = []

precision_scores_micro = []
precision_scores_macro = []
precision_scores_weighted = []

recall_scores_micro = []
recall_scores_macro = []
recall_scores_weighted = []

f1_scores_micro = []
f1_scores_macro = []
f1_scores_weighted = []

mean_squared_errors = []

times = []
performances = []


for model_name, model in models:
    # Start time
    start_time = time.time()
    model_names.append(model_name)
    # Model training
    print(model_name + ' - Training...')
    model.fit(X_train, y_train)
    
    # Prediction
    print(model_name + ' - Predicting...')
    y_pred = model.predict(X_test)

    
    # Calculate cross validation, accuracy, precision, recall, f1, mse scores
    print(model_name + ' - Calculating Scores...')
    cv_scores = cross_val_score(model, X_train, y_train, cv = 10)
    mean_cv_sc = cv_scores.mean()
    
    accuracy_sc = accuracy_score(y_test, y_pred) 
    
    precision_score_micro = precision_score(y_test, y_pred, average = 'micro')
    precision_score_macro = precision_score(y_test, y_pred, average = 'macro')
    precision_score_weighted = precision_score(y_test, y_pred, average = 'weighted')
    
    recall_score_micro = recall_score(y_test, y_pred, average = 'micro')
    recall_score_macro = recall_score(y_test, y_pred, average = 'macro')
    recall_score_weighted = recall_score(y_test, y_pred, average = 'weighted')
    
    f1_score_micro = f1_score(y_test, y_pred, average = 'micro')
    f1_score_macro = f1_score(y_test, y_pred, average = 'macro')
    f1_score_weighted = f1_score(y_test, y_pred, average = 'weighted')
    
    mean_squared_err = mean_squared_error(y_test, y_pred)
    
    
    # Save scores
    print(model_name + ' - Saving Scores...')
    mean_cv_scores.append(mean_cv_sc)
    accuracy_scores.append(accuracy_sc)
    
    precision_scores_micro.append(precision_score_micro)
    precision_scores_macro.append(precision_score_macro)
    precision_scores_weighted.append(precision_score_weighted)
    
    recall_scores_micro.append(recall_score_micro)
    recall_scores_macro.append(recall_score_macro)
    recall_scores_weighted.append(recall_score_weighted)
    
    f1_scores_micro.append(f1_score_micro)
    f1_scores_macro.append(f1_score_macro)
    f1_scores_weighted.append(f1_score_weighted)
    
    mean_squared_errors.append(mean_squared_err)
    
    
    with open(path_scores + model_name + '.txt', 'w') as file:
        file.write(model_name + ' - Results:\n')
        
        file.write(f'Mean Cross Validation Score = {mean_cv_sc}\n')
        file.write(f'Accuracy Score = {accuracy_sc}\n')
        
        file.write(f'Precision Score (micro) = {precision_score_micro}\n')
        file.write(f'Precision Score (macro) = {precision_score_macro}\n')
        file.write(f'Precision Score (weighted) = {precision_score_weighted}\n')
        
        file.write(f'Recall Score (micro) = {recall_score_micro}\n')
        file.write(f'Recall Score (macro) = {recall_score_macro}\n')
        file.write(f'Recall Score (weighted) = {recall_score_weighted}\n')
        
        file.write(f'F1 Score (micro) = {f1_score_micro}\n')
        file.write(f'F1 Score (macro) = {f1_score_macro}\n')
        file.write(f'F1 Score (weighted) = {f1_score_weighted}\n')
        
        file.write(f'Mean Squared Error = {mean_squared_err}\n')

    
    # Confusion matrix
    print(model_name + ' - Confusion Matrix...')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize = (10, 8))
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = labels_y, yticklabels = labels_y)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    plt.savefig(path_confusion_matrix + model_name + '.png')
        
    
    # Save model
    print(model_name + ' - Saving Model...\n')
    filename = path_models + model_name + '_model.sav'
    pickle.dump(model , open(filename, 'wb'))
    
    
    # Calculate elapsed time
    total_time = time.time() - start_time
    times.append(total_time)
    
    
    # Calculate performance
    performances.append(accuracy_sc / total_time)
    
    
def plotBar(title, data_results, data, xlabel, ylabel, save_path, figsize = (10, 6), bar_color = 'blue', bar_width = 0.8):
    plt.figure(figsize = figsize)
    plt.title(title)
    plt.bar(range(len(data_results)), data_results, tick_label = data, color = bar_color, width = bar_width)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(save_path)
    plt.tight_layout()
    plt.show()
    
    
# Visualize the models
plotBar("Models' Accuracy Score", accuracy_scores, model_names, 'Models', 'Accuracy Score', f'{path_plots}/accuracy_score.png')
plotBar("Models' Time Taken", times, model_names, 'Models', 'Training Time', f'{path_plots}/time.png')
plotBar("Models' Performance", performances, model_names, 'Models', 'Performance (Accuracy Score / Time Taken)', f'{path_plots}/performance.png')


# Find the best model
best_model_idx = np.argmax(accuracy_scores)
best_model_name = model_names[best_model_idx]
best_model_accuracy = accuracy_scores[best_model_idx]

# Print the best model
print(f"The best model is '{best_model_name}' with an accuracy score of {best_model_accuracy}")