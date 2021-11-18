
from scipy.io import loadmat

class Semisim():
    def __init__(self):
        data_path="Dataset/SemiSimulated"

        noisy=loadmat(data_path+"/noisy_data.mat",squeeze_me=True)
        pure=loadmat(data_path+"/pure_data.mat",squeeze_me=True)
        self.pure1=pure.get('sim1_resampled')
        self.noisy1 = noisy.get('sim1_con')
        self.fs=200
        self.N_ch=19
        print(self.pure1.shape)