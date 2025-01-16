## Model library installation

To install the `kws` package, you can run the following command:

```
pip install -e "git+https://github.com/akinwilson/heed.git#egg=kws&subdirectory=models"
```

**NOTE**: `egg=PACKAGE_NAME` where `PACKAGE_NAME` needs to be replaced with our python library name, and `subdirectory=SUB_DIRECTORY_NAME` needs to be replaced with the subdirectory containing the python library.

## Containerised fitting

### Building container

Want to run the fitting routine containerised. First, we will build the fitting container. Run the following command from within this folder:

```
docker build . -f Dockerfile.train -t fit:latest
```

### Running container

Next, we want to mount the volumes for inputs (training data) and outputs (model artefacts) that the fitting routine will use throughout its process. **NOTE**: we call `$(pwd)` in the below command which means the `volume` will be dependent on where this command is executed. So move up one level to the root of this repository and run the following command. This is such that there is alignment between where the serving container will look to find model artefacts, i.e. `/output` directory, which will be mounted as a volume to the serving container. See the `docker-compose.yaml` file for the collective setup.

```
docker run --name fit --rm -it -v $(pwd)/output:/usr/src/app/output -v /media/$(whoami)/Samsung_T5/data/audio/keyword-spotting:/usr/src/app/data fit:latest /bin/bash
```

Then you will have an `output/` directory generated at the root of this repostory which will be populated by model artefacts produced during the fitting routine.
