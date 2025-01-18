# Model serving

Using onnxruntime to serve predictions wrapped within a web-framework ( [fastAPI](https://fastapi.tiangolo.com/) ) to create an API.

## Containerised serving

### Building serving container


### Building serving container

To Build the image locally run: THIS IS CURRENTLY FAILING: [SEE HERE](https://github.com/onnx/onnx-tensorrt/issues/1009) FOR TensorRT runtime. 

```
docker build . -f Dockerfile.tensorrt -t serve:latest
```

### Running serving container

To deploy the image and test the endpoint, run:

```
docker run --gpus all -p 8080:80 -e "WORKERS=1" -e "EXECUTION_PROVIDER=TensorrtExecutionProvider"  --name="rt_test" -it serve:latest
```
 
**Note** You will need to move a `model.onnx` into the `/deploy/model` directory if you wish to deploy a model. 

#### Running the image iteractively 

`docker run --gpus all -p 8080:8080 -it serve:latest /bin/bash`

**Note** This will require Nvidia docker. You will enter the container at `/workspace`. 

From within the container, enter the command: `../app && python main.py` 
To start the server. Obviously you can start the server with single command rather than entering into the container, but the `entrypoint` of the dockerfile needs to be configured. To test out the enpoint via the swagger UI go to: 
```
http://0.0.0.0:8080/docs
```

To do:
18 Jan 2025

- [ ] Make the execution provider an environment variable ['TensorrtExecutionProvider','CPUExecutionProvider','CUDAExecutionProvider']
- [ ] Test the endpoint and create report using different providers and record latency and max QPS
