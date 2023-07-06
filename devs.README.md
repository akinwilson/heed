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
python.py -m ResNet -s dev
```