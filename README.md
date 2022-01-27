# Pytorch Code Snipes

## Feature Extractor

### Dates
22/01/27

### Intro

 - You can select the type of layer you want. For instance, If you want to extract the intermediate feature after the **relu** activation, modify as follows.
 
 ```
 if  isinstance(layer,nn.ReLU)  :
     layer.register_forward_hook(self.get_activation(idx))
     self.names[idx] = layer
     idx += 1
 ```
