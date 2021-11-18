
from scipy.io import loadmat
import mne
from EyeBlink_data import *
from Eye_State import *
import sklearn.preprocessing as skp
import sklearn.metrics as skm
from datas import *

datas=EEGDenoisingNet(-5, 100)
datas_filted=filt(datas.pure,0.5,125,datas.fs)

N_ch=datas.N_ch
fs=float(datas.fs)
eeg = datas.pure[0]
noisy=datas.noisy[0]
norm=skp.Normalizer()
norm.fit(noisy)
noisy2=norm.transform(noisy)
eeg2=norm.transform(eeg)

eeg=(eeg-eeg.min())/(eeg.max()-eeg.min())
noisy2=(noisy-noisy.min())/(noisy.max()-noisy.min())




def spec_cos_sim(x,y):
    x=abs(np.fft.fft(x)[:,0:x.shape[1]//2])
    y=abs(np.fft.fft(y)[:,0:y.shape[1]//2])
    return cos_sim(x,y)
def cos_sim(x,y):
    cos_sim=0
    for ch in range(0,x.shape[0]):
        cos_sim+=skm.pairwise.cosine_similarity([x[ch]], [y[ch]])[0]/x.shape[0]
    return cos_sim
def spec_mse(x,y):
    x=abs(np.fft.fft(x)[:,0:x.shape[1]//2])
    y=abs(np.fft.fft(y)[:,0:y.shape[1]//2])
    mse_spec=skm.mean_squared_error(x, y,multioutput='uniform_average')
    return mse_spec

def spec_cor(x,y):
    x=abs(np.fft.fft(x)[:,0:x.shape[1]//2])
    y=abs(np.fft.fft(y)[:,0:y.shape[1]//2])
    corr=0
    for ch in range(0,x.shape[0]):
        corr+=np.corrcoef(x[ch],y[ch])[0,1]/x.shape[0]
    return corr
    
def correlation(x,y):
    corr=0
    for ch in range(0,x.shape[0]):
        corr+=np.corrcoef(x[ch],y[ch])[0,1]/x.shape[0]
    return corr
        
        

def apply_mne_ica(noisy, fs, N_ch, max_iter, N_comp,ex=[0]):
    info = mne.create_info(N_ch, sfreq=fs,ch_types=["eeg"] * N_ch )
    raw = mne.io.RawArray(noisy, info)
    raw.set_montage("standard_1020",on_missing ="ignore")
    ica = mne.preprocessing.ICA(method="infomax", n_pca_components=N_comp, max_iter=max_iter,
                                fit_params={"extended": True}, random_state=1)
    ica.fit(raw)
    ica.exclude = ex
    raw_corrected = raw.copy()
    ica.apply(raw_corrected)
    corrected=raw_corrected.get_data()
    return corrected

corrected1=apply_mne_ica(noisy2,fs,N_ch,500,64,[0])
corrected2=apply_mne_ica(noisy2,fs,N_ch,500,64,[0,1])
corrected3=apply_mne_ica(noisy2,fs,N_ch,500,32,[0])

plt.figure(figsize=(25,15),dpi=50)
for ch in range(0,6):
    plt.plot(eeg2[ch]+ch*0.2,"r")
    plt.plot(corrected1[ch]+ch*0.2,"b")
    #plt.plot(noisy2[ch]+ch*0.5,"black")
plt.show()
'''
mse1=skm.mean_squared_error(eeg2, corrected1,multioutput='uniform_average')
mse2=skm.mean_squared_error(eeg2, corrected2,multioutput='uniform_average')
mse3=skm.mean_squared_error(eeg2, corrected3,multioutput='uniform_average')

corr1=correlation(eeg2,corrected1)
corr2=correlation(eeg2,corrected2)
corr3=correlation(eeg2,corrected3)

spec_mse1=spec_mse(eeg2,corrected1)
spec_mse2=spec_mse(eeg2,corrected2)
spec_mse3=spec_mse(eeg2,corrected3)

spec_cor1 = spec_cor(eeg2,corrected1)
spec_cor2 = spec_cor(eeg2,corrected2)
spec_cor3 = spec_cor(eeg2,corrected3)

cossim1=cos_sim(eeg2,corrected1)
cossim2=cos_sim(eeg2,corrected2)
cossim3=cos_sim(eeg2,corrected3)

spec_cossim1=spec_cos_sim(eeg2,corrected1)
spec_cossim2=spec_cos_sim(eeg2,corrected2)
spec_cossim3=spec_cos_sim(eeg2,corrected3)
'''