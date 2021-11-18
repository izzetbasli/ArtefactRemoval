from scipy.io import arff
import numpy as np
from scipy.signal import butter, lfilter,filtfilt
class EyeState():
    def __init__(self):
        data, meta=arff.loadarff('Dataset/Eyestate/EEG Eye State.arff')
        for i in meta:
            print(i)
        print(data['AF3'].shape)

        newdatas=np.reshape(data['AF3'],(14980,1)).T
        ch_names=['AF3', 'F7', 'F3' ,'FC5','T7', 'P7', 'O1', 'O2', 'P8', 'T8','FC6','F4','F8','AF4']

        for i in ch_names:
            new = np.reshape(data[str(i)], (14980, 1)).T
            newdatas = np.append(newdatas, new, axis=0)
        print(newdatas.shape)
        self.labels = data['eyeDetection']
        self.N_ch=15
        self.noisy=newdatas
        self.fs=250
        
    def filt(self,low,high):
        datas=self.noisy
        s=0
        for ch in datas:
            datas[s]=(butter_bandpass_filter(ch, lowcut=low, highcut=high, fs=self.fs, order=5))
            s+=1
        return datas
def butter_bandpass(lowcut, highcut, fs, order, axis=-1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5,axis=-1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order, axis=axis)
    y = filtfilt(b, a, data, axis=axis)
    return y        
