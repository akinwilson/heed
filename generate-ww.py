from pydub import AudioSegment
from gtts import gTTS
import shutil
from pathlib import Path 
import pandas as pd 
import numpy as np 
import json 


def get_record(wav_path, label):
    return {
        "wav_path":wav_path, 
        "label":label
            }


cls_label =[
    1,
    0,
    0,
    0,
    1,
    0,
    0,
    1,
    0,
    0,
]

text_list = [
    "Hello Sky",
    "Oscar",
    "OK Sky Q",
    "TV Guide",
    "Hey Sky",
    "OK Sky",
    "Why Sky",
    "Hi Sky",
    "Test 1",
    "Oscar Sky",
]

txt_and_labels = list(zip(text_list, cls_label))
data_records = []



class StatisticGatherer:

    def __init__(self):
        self.means = []
        self.vars = []

    def get_stats(self, wav_array):
        self.means.append(np.mean(wav_array))
        self.vars.append(np.var(wav_array))


    def get_avg_stats(self):
        return {
            "mean": sum(self.means)/len(self.means),
            "var":  sum(self.vars)/len(self.vars)

        }




sg = StatisticGatherer()


def text_to_speech(language, txt_and_labels, dst_folder, slow_):

    for (mytext, label) in txt_and_labels:
        # Passing the text and language to the engine,
        # here we have marked slow=False. Which tells
        # the module that the converted audio should
        # have a high speed
        myobj = gTTS(text=mytext, lang=language, slow=slow_)

        # Saving the converted audio in a mp3 file named
        file_save_path = dst_folder / f"{mytext.replace(' ', '_')}_{language}_{'slow' if slow_ else 'fast'}.mp3"

        myobj.save(file_save_path)

        sound = AudioSegment.from_file(file_save_path)
        sound = sound.set_frame_rate(16000)
        wav_array = np.array(sound.get_array_of_samples())

        # We devide by max dtype value to ensure the values in
        # the array are bound to the range [-1, 1]
        dtypes_to_max_val = {"int16": np.iinfo(np.int16).max}
        dtype_max_val = dtypes_to_max_val[str(wav_array.dtype)]
        sg.get_stats(wav_array / dtype_max_val)        


        wav_path = Path().cwd() / "data/wav"
        wav_path.mkdir(parents=True, exist_ok=True)
        wav_file_path = wav_path / f"{mytext.replace(' ', '_')}_{language}.wav" 

        sound.export(wav_file_path, format="wav")
        data_records.append(get_record(wav_file_path, label))

    return pd.DataFrame.from_records(data_records)









if __name__ == "__main__":

    language_list = ["en", "fr", "hi", "it", "es"][:1]
    speed_list = [True, False]
    dest_folder = Path().cwd() / "data/mp3"
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    df_folder = Path().cwd() / "data"
    dfs = []
    for language in language_list:
        for speed in speed_list:
            print(f"Create {language} {speed}")
            dfs.append(text_to_speech(language, txt_and_labels, dest_folder, speed))
            
    # saving fd
    pd.concat(dfs).to_csv(df_folder / "data.csv", index=False)
    # removing old mp3s
    shutil.rmtree(dest_folder)
    # saving stats 
    with open(df_folder / 'stats.json', 'w', encoding='utf-8') as f:
            json.dump(sg.get_avg_stats(), f, ensure_ascii=False, indent=4)
    print(f"Saving  stats ...")