

from EyeBlink_data import EyeBlink

eyeblink=EyeBlink()
data=eyeblink.datas_vr
datas_filted=filt(data,0.5,124,eyeblink.fs)
corrected=apply_mne_ica(datas_filted[0],eyeblink.fs,eyeblink.N_ch,500)

plt.figure(figsize=(25,15),dpi=50)
for ch in range(0,eyeblink.N_ch):
    plt.plot(datas_filted[0][ch][0:500]+ch*100,"r")
    plt.plot(corrected[ch][0:500]+ch*100,"b")
    #plt.plot(noisy[ch]+ch*1000,"black")
plt.show()


eyestate=EyeState()
eyestate.filt(0.5,124)
datas=eyestate.noisy[:,2000:10000]
corrected=apply_mne_ica(datas,eyestate.fs,eyestate.N_ch,1500)

plt.figure(figsize=(25,15),dpi=50)
for ch in range(3,6):
    plt.plot(datas[ch][4500:5500]+ch*100,"r")
    plt.plot(corrected[ch][4500:5500]+ch*100,"b")
    #plt.plot(noisy[ch]+ch*1000,"black")
plt.show()