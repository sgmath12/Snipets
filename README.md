# Pytorch Code Snipes

## Dataset


 - This code made it easy to retrieve the data set before entering the **dataloader** of the pytorch model.

## Feature Extractor


 - It is a code that extracts the intermediate feature of the pytorch model. 
 - You can select the type of layer you want. For instance, If you want to extract the intermediate feature after the **relu** activation, modify as follows.
 
 ```
 if  isinstance(layer,nn.ReLU)  :
     layer.register_forward_hook(self.get_activation(idx))
     self.names[idx] = layer
     idx += 1
 ```

## Dod_line_subfigure

- It's a code for drawing figures easily using matplotlib.
- You will draw a dot-line as shown in the following picture.
- You can enter data in the text file, and for more information on how to operate, see the code in `Dod_data class` and /data, then you can understand how to use.
- Enter the data and run it as follows.
```
python dod_line_subgraph.py
```
- If `--common_legend True`, then it will creates subgraphs with common legned as follows. 
![prob_test](https://user-images.githubusercontent.com/21999383/154065375-8ad9af56-bdec-428c-b633-f78228bf8844.png)
- If `--common_legend False`, then it will creates subgraph wihh normal legend as follows.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="https://user-images.githubusercontent.com/21999383/154065384-91022f29-be45-47bc-b067-e8d1056ee2d7.png" width="400" height="300"/>
