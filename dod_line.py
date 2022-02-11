from dataclasses import dataclass
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pdb

def compute_pos(xticks, width, i, models):
    index = np.arange(len(xticks))
    n = len(models)
    correction = i-0.5*(n-1)
    return index + width*correction

class Dod_data():
    def __init__(self,file_path):
        self.x_axis_start = 0.0
        self.x_axis_end = 1.0
        self.y_axis_start = 0.0
        self.y_axis_end = 100.0
        self.raw_data = self.read_data(file_path)
        self.X = None
        self.Y = None
        self.get_X_and_Y(self.raw_data)

    def read_data(self,file_path):
        '''
        Currently support .txt file
        '''
        data = {}
        with open(file_path) as f:
            lines = f.readlines()
            lines = list(filter(('\n').__ne__,lines))
            for line in lines:
                key, value = line.split('=')
                data[key.strip()] = value.strip()
        
        return data

    def get_X_and_Y(self,data):
        Y = []
        for key, value in data.items():
            if '[' in value or ']' in value:
                value = np.fromstring(value.strip('[,]'), dtype = float, sep = ',')
                if 'x' in key :
                    start,end,number = value
                    X = np.arange(start,end+start,(end-start)/(number-1))
                else : 
                    Y.append(value)
        self.X = X
        self.Y = np.array(Y)

    


data = Dod_data("./test.txt")


# # x = [10,20,30,40,50,60,70,80,90,100]
# x = np.array([10,20,30,40,50,60,70,80,90,100]) * 0.01
# y1 = [ 71.7, 29.5, 7.1, 1.1, 0, 0, 0, 0,0,0]
# y2 = [84.09, 45.4, 6.5, 0.4, 0.1, 0, 0, 0, 0, 0]
# y3 = [44.2, 0.04, 0, 0, 0, 0, 0, 0, 0, 0]
# y4 = [48.2, 2.7, 0, 0, 0, 0, 0, 0, 0, 0]

 
 
# # Plot lines with different marker sizes
# plt.ylim(-5,100)
# plt.xlim(-0.05,1.05)
# plt.xlabel("r",fontsize=13)
# plt.ylabel("Accuracy (%)",fontsize=13)
# plt.plot(x, y1, label = 'Inc-v3', lw=2, marker='o', ms=10) # square
# plt.plot(x, y2, label = 'Inc-v4', lw=2, marker='D', ms=10) # triangle
# plt.plot(x, y3, label = 'Res-18', lw=2, marker='s', ms=10) # circle
# plt.plot(x, y4, label = 'IncRes-v2', lw=2, marker='p', ms=10) # diamond

 
# plt.legend()
# plt.show()
# plt.close()
# ##############################################################

# pos = np.arange(5)
# xticks = ['conv2d_2b','conv2d_4b','mixed_5c','mixed_6a','mixed_6c']
# # xticks = ['bn1','layer1','layer2','layer3','layer4']
# plt.xticks(pos,xticks)

# # x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]


# # Res18 data
# y1 = [ 22.7455,54.2,75.4,75.5,43.4]
# y2 = [28.2565,59.5,77,76,43.2]
# y3 = [100,100,100,100,100]
# y4 = [18.62,39.24,56.06,50.6,28.6]
# y5 = [72.173,93.294,97.2,97.3,83.6]
# y6 = [44.1,84,95.5,97.4,78]
# y7 = [23.22,24.02,27.60,28.23,22.5]
# y8 = [8.1,11.31,12.32,13.21,10.3]

