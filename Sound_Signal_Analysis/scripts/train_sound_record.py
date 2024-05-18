import sounddevice as sd
import soundfile as sf
import time

train_sound_path = 'sounds/'
background_sound_path = 'background_sounds/train/'

samplerate = 44100
duration = 15
delay_between_records = 5

background_sounds = [sf.read(f"{background_sound_path}{i}.wav", dtype='float32')[0] for i in range(1, 4)]

for i in range(1, 7):
    print(f'Ready in {delay_between_records} seconds...\n')
    time.sleep(delay_between_records)
    
    background_type = 'with' if i >= 4 else 'without'
    
    print(f'Start Record {background_type.capitalize()} Background {i}...')
    recorded_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
    
    if i >= 4:
        background_slice = background_sounds[i - 4][:len(recorded_data), 0]
        
        background_slice *= 0.15
        
        recorded_data = recorded_data.flatten()
        background_slice = background_slice.flatten()
        
        recorded_data += background_slice
    
    print(f'End Record {background_type.capitalize()} Background {i}.\n')
    
    sd.wait()
    
    filename = f'{background_type}_background_{i}.wav'
    sf.write(train_sound_path + filename, recorded_data, samplerate)




