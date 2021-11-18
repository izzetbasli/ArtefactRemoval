
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter,filtfilt
from scipy.io import loadmat
from sklearn import preprocessing as skp
from scipy.io import arff


class EEGDenoisingNet():
    def __init__(self, desired_SNR, N_tr):
        data_path = "Dataset/EEGDenoisingNet"
        eeg = np.load(data_path + "/EEG_all_epochs.npy")
        eog = np.load(data_path + "/EOG_all_epochs.npy")
        self.fs = 256
        self.N_ch = 64
        self.N_tr = N_tr

        eeg = eeg.astype(np.float32)
        eog = eog.astype(np.float32)

        self.pure = np.reshape(eeg[0:4514 // 64 * 64], (4514 // 64, 64, 512))
        self.N_pure = self.pure.shape[0]

        self.noisy = add_noise(desired_SNR, self.pure, eog[0:N_tr])


def filt(datas, low, high, fs):
    for i in range(0, datas.shape[0]):
        for s in range(0, datas.shape[1]):
            datas[i][s] = (butter_bandpass_filter(datas[i][s], lowcut=low, highcut=high, fs=fs, order=5))
    return datas


def butter_bandpass(lowcut, highcut, fs, order, axis=-1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=-1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order, axis=axis)
    y = filtfilt(b, a, data, axis=axis)
    return y


def SNR(signal, noise):
    psignal = np.sum((signal * signal) / signal.shape[0])
    pnoise = np.sum((noise * noise) / noise.shape[0])
    return 10 * np.log10(psignal / pnoise)


# desired SNR için gürültülü data oluşturma
def add_noise(desired_SNR, signals, noises):
    new_signal = np.zeros((70 * noises.shape[0], 64, 512))
    i = 0
    for noise in noises:
        for signal in signals:
            for ch in range(0, 64):
                new_desired = desired_SNR + np.random.rand() * 4 - 2
                psignal = np.sum((signal[ch] * signal[ch]) / signal[ch].shape[0])
                pnoise = np.sum((noise * noise) / noise.shape[0])
                K = np.sqrt((psignal / pnoise) * pow(10, -new_desired / 10))
                new_signal[i][ch] = signal[ch] + K * noise
            i = i + 1
    return new_signal

class Semisim():
    def __init__(self,sample=1):
        data_path="Dataset/SemiSimulated"

        noisy=loadmat(data_path+"/noisy_data.mat",squeeze_me=True)
        pure=loadmat(data_path+"/pure_data.mat",squeeze_me=True)
        self.pure=pure.get('sim'+str(sample)+'_resampled')
        self.noisy = noisy.get('sim'+str(sample)+'_con')
        self.fs=200
        self.N_ch=19


def Snp(data,sample):
    eeg = data[sample]
    norm = skp.Normalizer()
    norm.fit(eeg)
    eeg2 = norm.transform(eeg)
    return eeg2


class EyeState():
    def __init__(self):
        data, meta=arff.loadarff('Dataset/EyeState/EEG Eye State.arff')
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