# Signal Processing Projects

A comprehensive implementation of biomedical signal processing techniques focusing on ECG, PPG, EMG, and audio signal analysis.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![NumPy](https://img.shields.io/badge/NumPy-latest-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-latest-red)

## 🎯 Overview

This repository contains four main signal processing projects developed for biomedical signal analysis:
1. Sound Signal Analysis: Voice and acoustic event recognition
2. ECG PQRST Block Detector: Detection and analysis of ECG signal components
3. PPG Signal Analysis: Analysis of photoplethysmogram signals
4. EMG Gesture Recognition: Hand gesture recognition using MYO Thalmic bracelet

## 🔧 Technical Components

### Project 1: Sound Signal Analysis

#### Audio Processing Pipeline
- Record Audio Files (with/without background noise)
- Short-Time Fourier Transform (STFT) analysis
- Model Training with multiple classifiers
- Interactive spectrogram labeling

#### Core Components
1. **Signal Processing**
   - STFT implementation
   - Background noise handling
   - Feature extraction

2. **Model Implementation**
   - RandomForestClassifier
   - KNeighborsClassifier
   - DecisionTreeClassifier
   - SVC

### Project 2: ECG PQRST Block Detector

#### Signal Processing Pipeline
- Gaussian smoothing
- Baseline correction
- Peak detection
- PQRST complex analysis

#### Key Features
1. **Wave Detection**
   - P wave detection
   - QRS complex identification
   - T wave analysis
   - Time interval calculations

2. **Signal Analysis**
   - Derivative-based detection
   - Zero-crossing analysis
   - Amplitude threshold detection

### Project 3: PPG Signal Analysis

#### Signal Processing Components
- Peak detection
- Time difference analysis
- PWD calculations
- Interactive visualization

### Project 4: EMG Gesture Recognition

#### Processing Pipeline
- 8-channel EMG data processing
- Feature extraction
- Window size optimization
- Real-time gesture prediction

## 🛠️ Dependencies

- numpy: Numerical operations
- scipy: Signal processing
- scikit-learn: Machine learning models
- matplotlib: Visualization
- librosa: Audio processing
- pandas: Data manipulation

## 🚀 Usage

1. Install dependencies:
```bash
pip install numpy scipy scikit-learn matplotlib librosa pandas
```

2. Run sound signal analysis:
```bash
python sound_signal_model_training.py
```

3. Run ECG analysis:
```bash
python pqrst_detector.py
```

## 📊 Results

### EMG Gesture Recognition Performance
- Random Forest: 84.83% accuracy
- Best Window Size: 1000
- Optimal Stride: 50

### Sound Signal Analysis
- Successfully implemented voice recognition
- Achieved accurate background noise filtering
- Generated comprehensive spectrograms

## 🔄 Future Improvements

1. **Model Enhancement**
   - Deep learning implementation
   - Real-time processing optimization
   - Enhanced feature extraction

2. **Performance Optimization**
   - Improved window size analysis
   - Better noise reduction
   - Faster inference times

## 🤝 Contributing

Contributions are welcome! Key areas:
1. Model optimization
2. Signal preprocessing improvements
3. Documentation enhancement
4. New feature implementation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Burak TÜZEL (191805057)
- Semih Utku POLAT (191805060)

## 🏫 Academic Context

Biomedical Signal Analysis and Machine Learning (2023/2024)  
Aydin Adnan Menderes University  
Computer Engineering Department
