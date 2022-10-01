from pydub import AudioSegment
from gtts import gTTS
import os

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


def text_to_speech(language, text_list, dst_folder, slow_):

    for mytext in text_list:
        # Passing the text and language to the engine,
        # here we have marked slow=False. Which tells
        # the module that the converted audio should
        # have a high speed
        myobj = gTTS(text=mytext, lang=language, slow=slow_)

        # Saving the converted audio in a mp3 file named
        file_save_path = (
            dst_folder
            + "/"
            + mytext.replace(" ", "_")
            + "_"
            + language
            + "_"
            + str("slow" if slow_ else "fast")
            + ".mp3"
        )
        myobj.save(file_save_path)

        sound = AudioSegment.from_file(file_save_path)
        sound = sound.set_frame_rate(16000)
        wav_file_path = (
            "./wav/"
            + mytext.replace(" ", "_")
            + "_"
            + language
            + "_"
            + str("slow" if slow_ else "fast")
            + ".wav"
        )
        sound.export(wav_file_path, format="wav")


language_list = ["en", "fr", "hi", "it", "es"]
speed_list = [True, False]

for language in language_list:
    for speed in speed_list:
        print("Create {} {}".format(language, speed))
        text_to_speech(language, text_list, "mp3", speed)
