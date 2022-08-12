import os
from time import sleep
from django.shortcuts import render
import youtube_dl
from tensorflow.keras.models import Model,load_model
import librosa
import numpy as np
import pandas as pd

# Create your views here.

def read_csv_file():
    df = pd.read_csv('fileToRaga2.csv', index_col=False)
    ragaNames=set(df['raga'])
    ragaNames = list(ragaNames)
    ragaNames.sort()
    raga_to_ragaId = pd.DataFrame(data=ragaNames,columns=['raga'])
    return (df,raga_to_ragaId)

def index(request):
    context = {'data':"None",'embed':"", 'error':""}
    if (request.POST):
        print(request.POST)
        # sleep(10)
        # return render(request,'index.html',{'data':"None",'error':"Sorry Invalid Video"})
        # Enter model code here
        df,raga_sto_ragaId = read_csv_file()
        model = load_model('500epochsAdam.h5')
        link = request.POST['link']
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'audio_' + link.split('/')[-1] + '.wav',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        context['embed'] = link + "?controls=0"
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([link])
        except:
            print("Invalid")
            return render(request,'index.html',{'data':"None",'error':"Sorry Invalid Video"})
        filename = 'audio.wav'
        y,sr = librosa.load(filename)
        dur = librosa.get_duration(filename=filename)
        tstart = 465
        tend = 525

        # Predictor

        n_fft = 2048
        hop_length = 512
        tstart_t = tstart*sr
        tend_t=tend*sr
        mel_sgram_array =[]
        MFCC_array =[]
        y_cut =y[tstart_t:tend_t]
        stft = librosa.core.stft(y_cut, hop_length=hop_length, n_fft=n_fft)
        sgram_mag, _ = librosa.magphase(stft)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sr)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram)
        mel_sgram_array.append(mel_sgram)
        MFCCs = librosa.feature.mfcc(y_cut, n_fft=n_fft,hop_length=hop_length,n_mfcc=8)
        MFCC_array.append(MFCCs)
        mel_sgram_array = np.array(mel_sgram_array)
        MFCC_array =np.array(MFCC_array)
        y_out  = model.predict([mel_sgram_array,MFCC_array])
        print(y_out)
        y_pred = np.argmax(y_out)
        print(y_pred)
        print("Predicted Raga",raga_sto_ragaId['raga'].iloc[y_pred])
        context['data'] = raga_sto_ragaId['raga'].iloc[y_pred]
        #os.remove('audio.wav')
    return render(request,'index.html',context)

