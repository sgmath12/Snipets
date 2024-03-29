from dataclasses import dataclass
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pdb
import argparse

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
        self.title = None
        self.xticks = None
        self.yticks = None
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
            elif key == 'title':
                self.title = value
            elif key == 'x_ticks':
                value = np.fromstring(value.strip('(,)'), dtype = float, sep = ',')
                # value = value.split(',')
                self.xticks = np.arange(value[0],value[1],value[2])
            elif key == 'y_ticks':
                value = np.fromstring(value.strip('(,)'), dtype = float, sep = ',')
                self.yticks = np.arange(value[0],value[1],value[2])

        self.X = X
        self.Y = np.array(Y)

def main(args):

    plt.style.use(['seaborn-darkgrid'])

    data_names = ["./data/curvature.txt","./data/curvature.txt"]
    # data_names = ["./data/data_1.txt","./data/data_2.txt","./data/data_3.txt","./data/data_4.txt"]
   

    fig_size_x = int(args.fig_size_x)
    # fig_size_y = int(args.fig_size_y)
    fig_size_y = fig_size_x * len(data_names)
    fig, axes = plt.subplots(1,len(data_names),figsize=(fig_size_y,fig_size_x))

    plt.subplots_adjust(bottom = 0.3, wspace = 0.4)

    for idx, ax in enumerate(fig.axes):
        data = Dod_data(data_names[idx])
        ax.set_xticks(data.xticks)
        ax.set_yticks(data.yticks)
        ax.set_xlim(data.x_axis_start,data.x_axis_end)
        ax.set_ylim(data.y_axis_start,data.y_axis_end)
        ax.set_xlabel(data.x_axis_name, fontsize = data.x_axis_font_size)
        ax.set_ylabel(data.y_axis_name, fontsize = data.y_axis_font_size) 
        ax.title.set_text(data.title)

        for idx,y in enumerate(data.Y):
            if args.common_legend == 'True':
                ax.plot(data.X,y, lw = data.line_width, marker = data.markers[idx], ms = data.marker_size)
            else : 
                ax.plot(data.X,y,label = data.label_names[idx], lw = data.line_width, marker = data.markers[idx], ms = data.marker_size)
                ax.legend(loc = "upper right")

    if args.common_legend == 'True':
        fig.legend(labels = data.label_names,loc = (0.25, 0.02), ncol = 5)

    # plt.legend()
    # plt.show()
    plt.savefig("./result_figure/" + args.fig_name,bbox_inches='tight')
    plt.close()




def argument_parsing():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fig_name", default="result.png")
    parser.add_argument("--common_legend", default="False")
    parser.add_argument("--fig_size_x", default="3")
    # parser.add_argument("--fig_size_y", default="3")


    return parser

if __name__ == "__main__":
    args = argument_parsing().parse_args()
    main(args)