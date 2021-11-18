import numpy as np
from datas import *
from ICA_MNE import apply_mne_ica
from sklearn import preprocessing as skp
from sklearn import metrics as skm
from ICA_MNE import correlation,spec_mse,spec_cor,spec_cos_sim,cos_sim
import pandas as pd

#datas= EyeState()                     ############EyeState

datas=Semisim(1)               ##############################   Semisim için

#datas=EEGDenoisingNet(-7,100)
#pure=Snp(pure,0)
#noisy=Snp(noisy,0)  ##########################Denoise için

pure=datas.pure                             ###########Denoise ve Semisim için
noisy=datas.noisy
fs=datas.fs
Nch=datas.N_ch

corrected=apply_mne_ica(noisy,fs,Nch,500,Nch,[0])

# Görselleştirme

plt.figure(figsize=(60,25))
for ch in range(0,Nch):
    plt.plot(noisy[ch,:]+ch*500,"r")
    plt.plot(corrected[ch,:]+ch*500,"b")
plt.show()


mse1=skm.mean_squared_error(pure, corrected,multioutput='uniform_average')
corr1=correlation(pure,corrected)
spec_mse1=spec_mse(pure,corrected)
spec_cor1 = spec_cor(pure,corrected)
cossim1=cos_sim(pure,corrected)
spec_cossim1=spec_cos_sim(pure,corrected)


def metrik_hesaplama(iteration,component):
    corrected = apply_mne_ica(noisy, fs, Nch, iteration , component , [0])
    mse1 = skm.mean_squared_error(pure, corrected, multioutput='uniform_average')
    corr1 = correlation(pure, corrected)
    spec_mse1 = spec_mse(pure, corrected)
    spec_cor1 = spec_cor(pure, corrected)
    cossim1 = cos_sim(pure, corrected)
    spec_cossim1 = spec_cos_sim(pure, corrected)
    return mse1,corr1,spec_mse1,spec_cor1,cossim1[0],spec_cossim1[0]

iterasyonlar=[500,400]
komponentler=[3,5,Nch]
metrik_names=['mse1','corr1','spec_mse1','spec_cor1','cossim1','spec_cossim1']
metriks=[]
satir_basliklar=[]
for i in iterasyonlar:
    for j in komponentler:
        metrikler= metrik_hesaplama(i,j)
        metriks.append(metrikler)
for b in iterasyonlar:
    for c in komponentler:
        satir_baslik='iterasyon sayisi= '+ str(b) +'    komponent sayisi = '+ str(c)
        satir_basliklar.append(satir_baslik)
basliklar=[str(metrik_names[i])for i in range(len(metrik_names))]
df=pd.DataFrame(metriks,index=satir_basliklar,columns=basliklar)
print(df)