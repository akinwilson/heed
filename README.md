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
- Get one architecture working, as in fitting, and saving 
- Normalisation approach: apply PCEN https://github.com/daemon/pytorch-pcen
- change the output of networks to be logits
- use nn.crossEntropy with raw logits 
- Make sure to adjust the Exporting of the model to onnx to include the correct input signature 
- Stress test the TensorRT backend version