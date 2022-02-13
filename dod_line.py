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
        self.x_axis_name = "X"
        self.y_axis_name = "Y"
        self.x_axis_font_size = 13
        self.y_axis_font_size = 13
        self.raw_data = self.read_data(file_path)
        self.X = None
        self.Y = None
        self.label_names = []
        self.markers = []
        self.line_width = 2
        self.marker_size = 10
        self.preprocess(self.raw_data)

    def read_data(self,file_path):
        '''
        Currently support .txt file
        '''
        data = {}
        with open(file_path) as f:
            lines = f.readlines()
            lines = list(filter((' ').__ne__,lines))
            lines = list(filter(('\n').__ne__,lines))
            for line in lines:
                key, value = line.split('=')
                data[key.strip()] = value.strip()
        return data

    def preprocess(self,data):
        Y = []
        label_names =[]
        markers = []
        for key, value in data.items():
            if '[' in value or ']' in value:
                value = np.fromstring(value.strip('[,]'), dtype = float, sep = ',')
                if 'x' in key :
                    start,end,number = value
                    X = np.arange(start,end+start,(end-start)/(number-1))
                else : 
                    Y.append(value)
                    
            elif 'label_name' in key:
                self.label_names.append(value)
            elif 'marker' in key:
                self.markers.append(value)
            elif key == 'x_axis_start':
                self.x_axis_start = float(value)
            elif key == 'x_axis_end':
                self.x_axis_end = float(value)
            elif key == 'y_axis_start':
                self.y_axis_start = float(value)
            elif key == 'y_axis_end':
                self.y_axis_end = float(value)
            elif key == 'x_axis_name':
                self.x_axis_name = value
            elif key == 'y_axis_name':
                self.y_axis_name = value
            elif key == 'x_axis_font_size':
                self.x_axis_font_size = float(value)
            elif key == 'y_axis_font_size':
                self.y_axis_font_size = float(value)
            elif key == 'line_width':
                self.line_width = float(value)
            elif key == 'marker_size':
                self.marker_size = float(value)

        self.X = X
        self.Y = np.array(Y)



data = Dod_data("./test.txt")

plt.xlim(data.x_axis_start,data.x_axis_end)
plt.ylim(data.y_axis_start,data.y_axis_end)
plt.xlabel(data.x_axis_name, fontsize = data.x_axis_font_size)
plt.ylabel(data.y_axis_name, fontsize = data.y_axis_font_size)

for idx,y in enumerate(data.Y):
    # plt.plot(data.X,y, label = 'Inc-v3', lw = data.line_width, marker = data.markers[idx], ms = data.marker_size)
    # break
    plt.plot(data.X,y,label = data.label_names[idx], lw = data.line_width, marker = data.markers[idx], ms = data.marker_size)


 
plt.legend()
plt.show()
plt.close()
# ##############################################################
pdb.set_trace()
