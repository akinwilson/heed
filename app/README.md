# Audio classification pipeline 

## Date: 3rd November 2022 updates

- Now have means of converting model from onnx to TFline via `TfliteConverter` in `app/src/wwv/util.py`
- `OnnxExporter` in `app/src/wwv/util.py` now accepts _opset_version_ and _input_shape_ as arguments
- Currently, feature extraction baked into data loading pipeline. But tested with Torchlibrosa and can now export part of model with onnx opset 17
- Expanded model zoo 
- `app/src/wwv/routine.py` contains pytorch lightning training, validating testing loops packaging a model architecture for fitting

<br>

| Model name       |               Task |                       Size |                              Caveats | Type | TTR | FTR | Acc| 
| ----------- |        ----------- | -----------                |                          ----------- |----------- | - | -| - |
| ResNet               | Binary classification                  |    ~37M         |  None              | CNN | 93% | 3% | 95%  |
| HTSWin Transformer   | Binary classification, Localisation    |    ~40M         | Frame-level localisation to be implemented     | Transformer| 90% | 2%|  91% |
| DeepSpeech   | Binary classification     |    ~90M         | issues with exporting     | RNN|  | |  |
| LeeNet   | Binary classification     |    ~5M         |   Not complete  |CNN|  | |  |
| MobileNet  | Binary classification     |    ~1M         | Not complete  |CNN|  | |  |




<br>
To do:

- [ ] Parameterise trainer whilst in the pipeline
- [ ] Find a way to inlucde programmatically in the data loader, what
  transformation to apply. 
- [ ] Include functional augmentation in the dataloader 

