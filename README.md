Signal Processing Projects

A collection of advanced biomedical and audio signal processing projects developed by Semih Utku Polat and Burak Tüzel, demonstrating sophisticated machine learning and signal analysis techniques.

Python
Scikit-learn
🎯 Overview

This repository features four signal processing projects:

    Sound Signal Analysis - Comprehensive audio classification system
    PQRST Block Detector - ECG signal processing for cardiac analysis
    PPG Keypoint Marking - PPG signal analysis with advanced metrics
    EMG Gesture Recognition - EMG-based gesture classification

Each project combines advanced signal processing and machine learning techniques.
🔧 Projects Overview
1. Sound Signal Analysis 🔊

An interactive audio classification system capable of:

    Spectrogram generation and labeling
    Recognizing sounds using machine learning
    Handling audio with background noise

Key Techniques

    Short-Time Fourier Transform (STFT)
    Manual signal segment labeling
    Random Forest classification with multi-model comparison

Sound Categories

    Computer, Engineering, Names, Cough, Clap, Snap, Ambient noise

Performance

    Best Accuracy: 83.33% (10/12 predictions correct)

2. PQRST Block Detector ❤️

A robust ECG signal processing project for detecting and analyzing cardiac features:

    Preprocessing for smoothing and baseline correction
    Automated detection of P, Q, R, S, T points
    Calculating cardiac intervals and visualizing characteristics

Key Techniques

    Gaussian smoothing
    Baseline correction
    Peak detection algorithms

3. PPG Keypoint Marking 💓

A photoplethysmogram (PPG) analysis system that identifies key physiological points and computes advanced metrics:

    Pulse Wave Duration (PWD)
    Heart Rate (HR)
    Systolic/Diastolic Phases

Key Techniques

    Derivative signals (VPG, APG, JPG, SPG)
    Zero-crossing detection
    Time difference and amplitude calculations

4. EMG Gesture Recognition 🖐️

An EMG signal-based gesture recognition project with data recorded using the MYO Thalmic bracelet.
Dataset Characteristics

    8 EMG channels, 36 subjects, 6-7 gestures, 4,237,907 data points

Gesture Classes

    Rest
    Fist
    Wrist flexion/extension
    Radial/ulnar deviations

Performance

    Best Model: Random Forest Classifier
    Accuracy: 84.83%

🛠️ Dependencies

    NumPy: Numerical operations
    Pandas: Data manipulation
    Scikit-learn: Machine learning models
    Librosa: Audio analysis
    Matplotlib: Visualizations

🚀 Usage

    Clone the repository:

git clone https://github.com/yourusername/signal-processing-projects.git
cd signal-processing-projects

    Install dependencies:

pip install -r requirements.txt

    Run a project:
        Sound Signal Analysis:

python sound_signal_analysis.py

PQRST Block Detector:

        python pqrst_detector.py

📊 Results
Sound Signal Analysis

    Model: Random Forest
    Accuracy: 83.33%

EMG Gesture Recognition

    Model: Random Forest
    Accuracy: 84.83%

🤝 Contributing

Contributions are welcome!
Focus areas:

    Feature engineering improvements
    Adding advanced ML models
    Optimizing code for large-scale datasets

📝 License

This project is licensed under the MIT License.
👥 Authors

    Semih Utku Polat - Contact
    Burak Tüzel - Contact

Acknowledgment: Special thanks to the Computer Engineering Department for supporting this research.
