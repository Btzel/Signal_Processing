import pickle
import numpy as np
import librosa, librosa.display
import sounddevice as sd

# liste içinde en çok tekrar eden sayıyı buluyoruz.
def most_frequent(List):
    if(List):
        return max(set(List), key = List.count)

# en iyi sonuca ait modeli yüklüyoruz.
loaded_model = pickle.load(open('models/KNN_model.sav', 'rb'))

##################### KAYDEDİLMİŞ VERİ İLE TEST ###########################

audio_signal_1,sample_rate_bg = librosa.load("test_sounds/withoutBG/test_1.wav")
audio_signal_2,sample_rate_bg = librosa.load("test_sounds/withoutBG/test_2.wav")
audio_signal_3,sample_rate_bg = librosa.load("test_sounds/withoutBG/test_3.wav")
audio_signal_4,sample_rate_bg = librosa.load("test_sounds/withoutBG/test_4.wav")
audio_signal_5,sample_rate_bg = librosa.load("test_sounds/withoutBG/test_5.wav")
audio_signal_6,sample_rate_bg = librosa.load("test_sounds/withoutBG/test_6.wav")
audio_signal_7,sample_rate_bg = librosa.load("test_sounds/withoutBG/test_7.wav")

audio_signal_bg_1,sample_rate_bg = librosa.load("test_sounds/withBG/test_1.wav")
audio_signal_bg_2,sample_rate_bg = librosa.load("test_sounds/withBG/test_2.wav")
audio_signal_bg_3,sample_rate_bg = librosa.load("test_sounds/withBG/test_3.wav")
audio_signal_bg_4,sample_rate_bg = librosa.load("test_sounds/withBG/test_4.wav")
audio_signal_bg_5,sample_rate_bg = librosa.load("test_sounds/withBG/test_5.wav")
audio_signal_bg_6,sample_rate_bg = librosa.load("test_sounds/withBG/test_6.wav")
audio_signal_bg_7,sample_rate_bg = librosa.load("test_sounds/withBG/test_7.wav")

test_datas = []
test_datas.append(('computer',audio_signal_1))
test_datas.append(('engineering',audio_signal_2))
test_datas.append(('semih utku',audio_signal_3))
test_datas.append(('polat',audio_signal_4))
test_datas.append(('cough',audio_signal_5))
test_datas.append(('clap',audio_signal_6))
test_datas.append(('snap',audio_signal_7))
test_datas.append(('computer',audio_signal_bg_1))
test_datas.append(('engineering',audio_signal_bg_2))
test_datas.append(('semih utku',audio_signal_bg_3))
test_datas.append(('polat',audio_signal_bg_4))
test_datas.append(('cough',audio_signal_bg_5))
test_datas.append(('clap',audio_signal_bg_6))
test_datas.append(('snap',audio_signal_bg_7))

results = []
for dataName, data in test_datas:

    hop_length = 512
    n_fft = 2048
    X = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
    S = librosa.amplitude_to_db(abs(X))

    s_transpose = np.transpose(S)
    
    y_predict = loaded_model.predict(s_transpose)

    none_zero = []
    for i in y_predict:
        if i !=0:
            none_zero.append(i)   
    none_zero.append(0)
    command = most_frequent(none_zero)
    
    if command == 1:
        results.append("computer")
    elif command == 2:
        results.append("engineering")
    elif command == 3:
        results.append("semih utku")
    elif command == 4:
        results.append("polat")
    elif command == 5:
        results.append("cough")
    elif command == 6:
        results.append("clap")
    elif command == 7:
        results.append("snap")
    else:
        results.append("wait")

###################### CANLI KAYIT TEST ###########################     
samplerate = 22050
duration = 2
# 2 saniyelik kayıt başlatıyoruz bu süre zarfında konuşmanız gerekir.(ileri,geri,sağ,sol gibi)
print("start")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
print("end")
sd.wait()
print("--------------")
audio_signal_rt = np.reshape(mydata, (len(mydata),))


# STFT uyguluyoruz.
hop_length = 512
n_fft = 2048
X = librosa.stft(audio_signal_rt, n_fft=n_fft, hop_length=hop_length)
S = librosa.amplitude_to_db(abs(X))
# ML algoritması için transposunu alıyoruz
s_transpose = np.transpose(S)

# model tahminleme yapıyor
y_predict = loaded_model.predict(s_transpose)
# 0 dışındaki değerleri listeye atıyoruz.
none_zero = []
for i in y_predict: #  and i!=6
    if i !=0:
        none_zero.append(i)   
none_zero.append(0)
# predictionlar içinde en çok tekrar eden değerimiz bizim sonucumuz oluyor. Yani komut'umuz.
command = most_frequent(none_zero)
# komut değeri 1,2,3,4,5,6 ise aşşağıdaki değerleri alıyor. Değil ise tekrar deneyin yazdırıyoruz.

if command == 1:
    print("computer")
elif command == 2:
    print("engineering")
elif command == 3:
    print("burak")
elif command == 4:
    print("tüzel")
elif command == 5:
    print("cough")
elif command == 6:
    print("clap")
elif command == 7:
   print("snap")
else:
    print("wait")

 
print(np.mean(none_zero))
        
