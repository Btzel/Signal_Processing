import sounddevice as sd
import soundfile as sf
import time

test_sound_path = 'test_sounds/withoutBG/'

samplerate = 44100
duration = 2
delay_between_records = 5

for i in range(7, 8):
    print(f'Ready in {delay_between_records} seconds...\n')
    time.sleep(delay_between_records)
    
    print(f'Start Record {i}...')
    recorded_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
    
    print(f'End Record {i}.\n')
    
    sd.wait()
    
    filename = f'without_background_{i}.wav'
    sf.write(test_sound_path + filename, recorded_data, samplerate)




