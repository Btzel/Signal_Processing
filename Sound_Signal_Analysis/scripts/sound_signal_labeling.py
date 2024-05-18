import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import pandas as pd
# Define the file paths for your audio recordings
audio_file = "sounds/merged_sound.wav"
# Load and concatenate all audio files
audio, sample_rate = librosa.load(audio_file)

def apply_stft_and_plot(audio, sample_rate, enable_labeling=True):
    hop_length = 512
    n_fft = 2048
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spec_db = librosa.amplitude_to_db(np.abs(spec))
    arr = np.zeros( (len(spec_db[1]), ) , dtype=np.int64)
    arr_ratio = arr.shape[0] / audio.shape[0]
    # Create the figure and axes objects
    fig, ax = plt.subplots(figsize=(12, 8))
    # Display the spectrogram using imshow with the correct time axis
    im = ax.imshow(spec_db, aspect='auto', origin='lower', extent=[0, len(audio) / sample_rate, 0, sample_rate / 2],
                   cmap='viridis')  # You can choose your desired colormap
    plt.colorbar(im, ax=ax, format='%+2.0f dB')  # Display the colorbar
    plt.title('Spectrogram')
    current_label = 1  # Initialize the current label
    # If enable_labeling is True, enable interactive labeling
    if enable_labeling:

        # Callback function for SpanSelector
        def onselect(xmin, xmax):
            nonlocal current_label  # Use the current_label from the outer function
            label_regions(arr, xmin*arr_ratio, xmax*arr_ratio, current_label, sample_rate)
            print(f'Labeled Segment: {xmin} to {xmax} seconds with label {current_label}')
        # Create a SpanSelector to interactively select segments
        span_selector = SpanSelector(ax, onselect, 'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))
        # Set x-axis limits explicitly after the SpanSelector is created
        ax.set_xlim([0, len(audio) / sample_rate])
        # Key event handler to change the current label with keys 1 to 7
        def on_key(event):
            nonlocal current_label
            if event.key.isdigit() and 1 <= int(event.key) <= 7:
                current_label = int(event.key)
                print(f'Current label set to {current_label}')
            elif event.key == 'enter':
                save_labels(spec_db, "sounds/spec_db.csv",enable_labeling=False)
                save_labels(arr,"sounds/labels.csv")
            else:
                None
        fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=True)

    return span_selector,arr,spec_db

def save_labels(data, file_path, enable_labeling=True):
    if enable_labeling:
        # Convert NumPy array to a Pandas DataFrame with a single column
        df = pd.DataFrame({'Value': data})
    else:
        # Convert NumPy array to a Pandas DataFrame without specifying columns
        df = pd.DataFrame(data)

    # Write to CSV using Pandas
    df.to_csv(file_path, index=False, header=False)  # Set header to False to omit column names

    if enable_labeling:
        print(f"Labels have been saved into '{file_path}'.")
    else:
        print(f"Data has been saved into '{file_path}'.")

# Function to label regions based on criteria
def label_regions(arr, start_time, end_time, label, sample_rate):
    start_idx = int(start_time * sample_rate)
    end_idx = int(end_time * sample_rate)
    arr[start_idx:end_idx] = label

# Assume arr is the array to store labels
span_selector,arr,spec_db = apply_stft_and_plot(audio, sample_rate, enable_labeling=True)
