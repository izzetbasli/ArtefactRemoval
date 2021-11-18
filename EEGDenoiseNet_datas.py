import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter,filtfilt

class EEGDenoisingNet():
    def __init__(self,desired_SNR,N_tr):
        data_path="C:/Users/izzet/Documents/Github/ArtefactRemoval/Dataset/EEGDenoisingNet"
        eeg=np.load(data_path+"/EEG_all_epochs.npy")
        eog=np.load(data_path+"/EOG_all_epochs.npy")
        self.fs=256
        self.N_ch=64
        self.N_tr=N_tr
        
        eeg=eeg.astype(np.float32)
        eog=eog.astype(np.float32)

        self.pure=np.reshape(eeg[0:4514//64*64], (4514//64,64,512))
        self.N_pure=self.pure.shape[0]

        self.noisy=add_noise(desired_SNR,self.pure, eog[0:N_tr])
    
def filt(datas,low,high,fs):
    for i in range (0,datas.shape[0]):
        for s in range(0,datas.shape[1]):
            datas[i][s]=(butter_bandpass_filter(datas[i][s], lowcut=low, highcut=high, fs=fs, order=5))
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

def SNR(signal,noise):
  psignal=np.sum((signal*signal)/signal.shape[0])
  pnoise=np.sum((noise*noise)/noise.shape[0])
  return 10*np.log10(psignal/pnoise)

# desired SNR için gürültülü data oluşturma
def add_noise(desired_SNR,signals, noises):
    new_signal=np.zeros((70*noises.shape[0],64,512))
    i=0
    for noise in noises:
        for signal in signals:
            for ch in range(0,64):
                new_desired=desired_SNR+np.random.rand()*4-2
                psignal=np.sum((signal[ch]*signal[ch])/signal[ch].shape[0])
                pnoise=np.sum((noise*noise)/noise.shape[0])
                K=np.sqrt((psignal/pnoise)*pow(10,-new_desired/10))
                new_signal[i][ch]= signal[ch]+K*noise
            i=i+1
    return new_signal
        

