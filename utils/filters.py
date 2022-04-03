import numpy as np
import paddle

class LowPassFilter():
    isFistTime=True
    hatxprev=None
    def __init__(self):
        pass

    def filter(self,x,alpha):
        if self.isFistTime:
            self.isFistTime=False
            self.hatxprev=x
        hatx=alpha * x + (1-alpha) * self.hatxprev
        self.hatxprev=hatx
        return hatx

PI=3.14159


"""
Data update rate in HZ: rate
Cutoff frequency in HZ: cutoff

"""
def alpha(rate,cutoff):
    eps = 1e-6
    tau=1.0 / (2*np.pi*cutoff + eps)
    te=1.0/ (rate + eps)
    return 1.0 / (1.0 + tau / (te + eps) )



class OneEuroFilter():
    isFistTime = True
    rate = 30 #Hz
    minCutoff = 3 #Hz
    dcutoff=0.6
    beta = 0

    def __init__(self):
        self.xLowPassFilter=LowPassFilter()
        self.dxLowPassFilter=LowPassFilter()

    def filter(self,x):
        dx=np.zeros(x.shape,dtype=np.float32)
        if self.isFistTime:
            self.isFistTime=False
        else:
            pass
            if self.dxLowPassFilter.hatxprev.max() > 1e30:
                # self.dxLowPassFilter.hatxprev*=0.001
                self.dxLowPassFilter.hatxprev = x

            dx=(x-self.dxLowPassFilter.hatxprev) *self.rate #位移*速度
        # self.dcutoff belong to [0,1]
        edx=self.dxLowPassFilter.filter(dx,alpha(self.rate,self.dcutoff))
        cutoff = self.minCutoff + self.beta * np.abs(edx)
        # cutoff = self.minCutoff

        result = self.xLowPassFilter.filter(x,alpha(self.rate,cutoff))
        return result


if __name__=="__main__":
    pass
