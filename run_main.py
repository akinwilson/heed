from wwv.config import Config 
MODEL_DIR = "/home/akinwilson/Code/pytorch/output/model"
DATA_DIR = "/home/akinwilson/Code/pytorch/dataset/keywords"
LR_RANGE = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5][-2]
BATCH_SIZE_RANGE = [1,8,32, 64, 128, 256][1]
EPOCH_RANGE = [1, 10, 30, 50, 100, 1000][1]
ES_PATIENCE_RANGE = [1, 10, 20, 100, 200][2]
MODELS = ["VecM5", "Resnet2vec1D","SpecResnet2D", "HSTAT", "DeepSpeech"][0]
AUDIO_FEATURE_OPT = ["spectrogram", "mfcc", "pcm"][-1]
PRETRAINED_MODEL_NAME_OR_PATH = "facebook/wav2vec2-base-960h"
AUGS = False 


params = {
    "audio_duration":3,
    "sample_rate":16000,
    "model_name": MODELS,
    "verbose": False,
    "path": {
        "model_dir": MODEL_DIR,
        "data_dir": DATA_DIR,
        "pretrained_name_or_path": PRETRAINED_MODEL_NAME_OR_PATH
        },
    "fit_param": {"init_lr":LR_RANGE, "weight_decay":0.0001, "max_epochs":EPOCH_RANGE, "gamma": 0.1,"es_patience":ES_PATIENCE_RANGE}, 
    "data_param":{"train_batch_size": BATCH_SIZE_RANGE, "val_batch_size": BATCH_SIZE_RANGE,"test_batch_size": BATCH_SIZE_RANGE}, 
    "audio_feature": AUDIO_FEATURE_OPT,
    "audio_feature_param": { "mfcc":{"sr":16000,"n_mfcc":20,"norm": 'ortho',"verbose":True,"ref":1.0,"amin":1e-10,"top_db":80.0,"hop_length":512,},
                            "spectrogram":{"sr":16000, "n_fft":2048, "win_length":None,"n_mels":128,"hop_length":512,"window":'hann',"center":True,"pad_mode":'reflect',"power":2.0,"htk":False,"fmin":0.0,"fmax":None,"norm":1,"trainable_mel":False,"trainable_STFT":False,"verbose": True },
                            "pcm": {}},
    "augmentation":{'Gain': AUGS, 'PitchShift': AUGS, 'Shift': AUGS},
    "augmentation_param":{"Gain": {  "min_gain_in_db":-18.0,"max_gain_in_db":  6.0,"mode":'per_example',"p":1,"p_mode":'per_example'},
                        "PitchShift": {"min_transpose_semitones": -4.0, "max_transpose_semitones": 4.0,"mode":'per_example',"p":1,"p_mode":'per_example',"sample_rate":16000,"target_rate": None,"output_type": None,},
                        "Shift":{ "min_shift":-0.5,"max_shift": 0.5,"shift_unit":'fraction',"rollover": True,"mode":'per_example',"p":1,"p_mode": 'per_example',"sample_rate": 16000,"target_rate":None,"output_type":None}},
    }


if __name__ == "__main__":
    cfg = Config(params)
    from wwv.train import main
    main(cfg)