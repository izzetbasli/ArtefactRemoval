
import numpy as np
from matplotlib import pyplot as plt
import glob, os
class EyeBlink():
    def __init__(self):
        os.chdir("C:/Users/izzet/Documents/GitHub/ArtefactRemoval/Dataset/EyeBlink/EEG-IO/")
        datas_io=[]
        for file in glob.glob("*_data.csv"):
            file="C:/Users/izzet/Documents/GitHub/ArtefactRemoval/Dataset/EyeBlink/EEG-IO/"+file
            data_sig = np.loadtxt(file, delimiter=";", skiprows=1, usecols=(1,2)).T
            datas_io.append(data_sig)

        os.chdir("C:/Users/izzet/Documents/GitHub/ArtefactRemoval/Dataset/EyeBlink/EEG-VR/")
        datas_vr=[]
        for file in glob.glob("*_data.csv"):
            file="C:/Users/izzet/Documents/GitHub/ArtefactRemoval/Dataset/EyeBlink/EEG-VR/"+file
            data_sig = np.loadtxt(file, delimiter=",", skiprows=5, usecols=(1,2)).T
            datas_vr.append(data_sig)

        os.chdir("C:/Users/izzet/Documents/GitHub/ArtefactRemoval/Dataset/EyeBlink/EEG-VV/")
        datas_vv=[]
        for file in glob.glob("*_data.csv"):
            file="C:/Users/izzet/Documents/GitHub/ArtefactRemoval/Dataset/EyeBlink/EEG-VV/"+file
            data_sig = np.loadtxt(file, delimiter=",", skiprows=5, usecols=(1,2)).T
            datas_vv.append(data_sig)
        self.datas_io=datas_io
        self.datas_vr=datas_vr
        self.datas_vv=datas_vv
        self.fs=250
        self.N_ch=2
        self.data1=datas_io[0]
        self.data2=datas_io[1]