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
