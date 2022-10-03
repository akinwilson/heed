# Audio classification pipeline 

Date: 3rd October 2022 
<br>
Currently, feature extraction baked into data loading pipeline. But I havent
made the changes to this repository to reflect this
<br>
<br>
_Why not part of model architecture?_
<br>
To export the model via onnx, the onnx operation set, _opset_, needs to include
fourier-related transfors, i.e. FFT, IFFT etc, opset 17, the newest operation
set, includes these and it is just a matter of time before they become
avaialble in the stable version of torch. 
_Consequence_
<br>
When serving the model, the same transformations that are applied during the
data loading stage need to be applied whilst endpoint of the model is queried 
<br>
To do:
- [ ] Move trainer into train.py from notebook 
- [ ] Parameterise trainer whilst in the pipeline
- [ ] Find a way to inlucde programmatically in the data loader, what
  transformation to apply. 
- [ ] Include functional augmentation in the dataloader 

