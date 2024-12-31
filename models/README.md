# Audio classification pipeline 

## Date: 5rd November 2022 updates

- Now have means of converting model from onnx to TFline via `TfliteConverter` in `app/src/wwv/util.py`. But it is a bit dodgy. 

- `OnnxExporter` in `app/src/wwv/util.py` now accepts _opset_version_ and _input_shape_ as arguments
- Feature extraction atm in data loader pipeline. But tested with Torchlibrosa and can now export part of model with onnx opset 17
- `../fit.py` now allows for multi-architecture training via passing command line args `--model_name MODEL_STRING` to script.
where `MODEL_STRING` can be: 
  1) ResNet
  2) HSTAT
  3) DeepSpeech
  4) LeeNet
  5) MobileNet

- `app/src/wwv/routine.py` contains pytorch lightning training, validating testing loops packaging a model architecture for fitting; works for both binary classification and localisation tasks 



| Model name       |               Task |                       Size |                              Caveats | Type | TTR | FTR | Acc| 
| ----------- |        ----------- | -----------                |                          ----------- |----------- | - | -| - |
| ResNet               | Binary classification                  |    ~37M         |  None              | CNN | 93% | 3% | 95%  |
| HTSWin Transformer   | Binary classification, Localisation    |    ~40M         | Frame-level localisation to be implemented     | Transformer| 90% | 2%|  91% |
| DeepSpeech   | Binary classification     |    ~90M         | issues with exporting     | RNN|  | |  |
| LeeNet   | Binary classification     |    ~5M         |   Not complete  |CNN|  | |  |
| MobileNet  | Binary classification     |    ~1M         | Not complete  |CNN|  | |  |




## **To Dos 5rd November 2022**:

- [ ] Include functional augmentation in the dataloader 

