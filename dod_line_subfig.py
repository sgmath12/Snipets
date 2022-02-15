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
            elif 'marker_shape' in key:
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



data_1 = Dod_data("./test.txt")

fig, (a1,a2,a3,a4) = plt.subplots(1,4,figsize=(15,2.5))


plt.subplots_adjust(bottom = 0.275, wspace = 0.4)
# plt.tight_layout()
# plt.constraint_layout()
a1.set_xlim(data_1.x_axis_start,data_1.x_axis_end)
a1.set_ylim(data_1.y_axis_start,data_1.y_axis_end)
a1.set_xlabel(data_1.x_axis_name, fontsize = data_1.x_axis_font_size)
a1.set_ylabel(data_1.y_axis_name, fontsize = data_1.y_axis_font_size)

for idx,y in enumerate(data_1.Y):
    a1.plot(data_1.X,y, lw = data_1.line_width, marker = data_1.markers[idx], ms = data_1.marker_size)

data_2 = Dod_data("./test_2.txt")

a2.set_xlim(data_2.x_axis_start,data_2.x_axis_end)
a2.set_ylim(data_2.y_axis_start,data_2.y_axis_end)
a2.set_xlabel(data_2.x_axis_name, fontsize = data_2.x_axis_font_size)
a2.set_ylabel(data_2.y_axis_name, fontsize = data_2.y_axis_font_size)

for idx,y in enumerate(data_2.Y):
    a2.plot(data_2.X,y, lw = data_2.line_width, marker = data_2.markers[idx], ms = data_2.marker_size)


data_3 = Dod_data("./test_2.txt")

a3.set_xlim(data_3.x_axis_start,data_3.x_axis_end)
a3.set_ylim(data_3.y_axis_start,data_3.y_axis_end)
a3.set_xlabel(data_3.x_axis_name, fontsize = data_3.x_axis_font_size)
a3.set_ylabel(data_3.y_axis_name, fontsize = data_3.y_axis_font_size)

for idx,y in enumerate(data_3.Y):
    a3.plot(data_3.X,y, lw = data_3.line_width, marker = data_3.markers[idx], ms = data_3.marker_size)


data_4 = Dod_data("./test_2.txt")

a4.set_xlim(data_4.x_axis_start,data_4.x_axis_end)
a4.set_ylim(data_4.y_axis_start,data_4.y_axis_end)
a4.set_xlabel(data_4.x_axis_name, fontsize = data_4.x_axis_font_size)
a4.set_ylabel(data_4.y_axis_name, fontsize = data_4.y_axis_font_size)

for idx,y in enumerate(data_4.Y):
    a4.plot(data_4.X,y, lw = data_4.line_width, marker = data_4.markers[idx], ms = data_4.marker_size)



fig.legend(labels = data_1.label_names,loc = (0.25, -0.0), ncol = 4)

# plt.legend()
# plt.show()
plt.savefig("test.png",bbox_inches='tight')
plt.close()
