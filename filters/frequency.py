import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Frequency:

    def __init__(self):
        pass

    def getFrequencies(self,data,maxLine):
        ans = np.zeros((maxLine,2))
        for x in range(maxLine):
            ans[x][0] = x+1
        for x in data:
            ans[int(x[0])-1][1] += 1
        return ans

    def plot_freq(self,data,saveplace=None):
        plt.figure(figsize=(20,12))
        plt.bar(data[:,0],data[:,1])
        if saveplace is not None:
            plt.savefig(saveplace)
            plt.close()
        else:
            plt.show()
