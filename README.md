# Notes for devs 






### Building deployment container
Run: 
`docker build . -f torch-deploy/Dockerfile -t serve:latest`
From the directory level of this README.md file. The build context needs access to the model definition at the moment and install the python package locally. 
This will build the docker images

Run:
`docker build . -f torch-deploy/Dockerfile.tensorrt -t serve:latest` 
To deploy the TensorRT serving image

To do:
- [x] Get one architecture working, as in fitting, and saving 
- [ ] Normalisation approach: apply PCEN https://github.com/daemon/pytorch-pcen
- [x] change the output of networks to be logits let Routine function handle 
- [x] use nn.crossEntropy with raw logits -> not needed 
- [x] Make sure to adjust the Exporting of the model to onnx to include the correct input signature 
- [ ] Bench mark and stress test the TensorRT backend GPU (cuda) and CPU (MLAS) 
- [ ] Check back for when OpSet 17 is in stable torch, want to try to export
  feature extraction as part of model architecture. torch.ftt ops should be
supported in opset 17. 
- [ ] Create a presentation as to why 
- [ ] From onnx to TFlite -> implement basic converted and test against chris
  butcher
