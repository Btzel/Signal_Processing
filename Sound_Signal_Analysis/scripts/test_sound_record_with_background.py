import sounddevice as sd
import soundfile as sf
import time

test_sound_path = 'test_sounds/withBG/'
background_sound_path = 'background_sounds/test/test_1.wav'

samplerate = 44100
duration = 2
delay_between_records = 5

background_sound, _ = sf.read(background_sound_path, dtype='float32')
background_sound = background_sound[:int(samplerate * duration), 0]

for i in range(1, 8):
    print(f'Ready in {delay_between_records} seconds...\n')
    time.sleep(delay_between_records)
    
    print(f'Start Record {i}...')
    recorded_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
    
    background_slice = background_sound * 0.15
    
    recorded_data = recorded_data.flatten()
    background_slice = background_slice.flatten()
    
    recorded_data += background_slice
    
    print(f'End Record {i}.\n')
    
    sd.wait()
    
    filename = f'with_background_{i}.wav'
    sf.write(test_sound_path + filename, recorded_data, samplerate)




