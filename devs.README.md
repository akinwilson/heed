# Project Layout
```
.
├── app
├── deploy
├── devs.README.md
├── fit.py
├── notebooks
├── output
├── README.md
└── requirements.txt
```

# Fit a model

```
python fit.py -m {HSTAT|ResNet|DeepSpeech|LeeNet|MobileNet} -s {dev|test|prod}
```
defaults to: 
```
python.py fit.py -m ResNet -s dev
```
This will run the training routine for the ResNet model with the development environment variables


# HPO model
from inside `./notebooks `
```
python hpo.py
```