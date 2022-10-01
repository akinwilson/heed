import os
import base64
import json
import time
import requests
import streamlit as st

SAGEMAKER_TIMEOUT = 2
SAGEMAKER_TIMEOUT_ERR = -1


class VerificationClient:
    def __init__(self):
        self.endpoint = "http://localhost:8080/invocations"

    def encode_wav_base64_request(self, wav_content):
        """
        The data when posting a request to the sagemaker endpoint has to be encoded in base64
        This methos take raw audio as input and converts it to base64 which can then be used
        in the POST request.
        :param wav_content: WW audio
        :return: base64 encoded audio
        """
        base64_encoded = base64.b64encode(wav_content).decode("UTF-8")
        print("---> ", len(base64_encoded))
        print("Wav --> ", len(wav_content))
        return json.dumps({"content": base64_encoded})

    def do_ww_verification(self, audio_payload):
        json_data = {}
        try:
            start = time.time()
            response = requests.post(
                self.endpoint,
                data=audio_payload,
                timeout=SAGEMAKER_TIMEOUT,
                headers={"Content-Type": "application/json"},
            )
            print(response.text)
            json_data = response.text
            end = time.time()
            print("Response got in " + str(end - start) + " seconds")
        except Exception as e:
            print("Connection Error: " + type(e).__name__)
            json_data["prediction"] = SAGEMAKER_TIMEOUT_ERR
            print(
                "Timed out after "
                + str(SAGEMAKER_TIMEOUT)
                + " seconds "
                + str(json_data)
            )

        return json_data


st.title("WakeWord Verification Client")

# Assumes the app is run from root of the project
st.image(os.getcwd() + "/apps/streamlit-ww-client/Llama.jpg")
path = st.text_input(
    "Path to ww wave file:",
    value=os.getcwd()
    + "/apps/streamlit-ww-client/0000_2021_03_02_13_50_23_WakeWord.wav",
)

if st.button("Send to Cloud Model"):
    st.markdown("____________________________________")
    st.text("Sending " + path)
    client = VerificationClient()
    wav_path = path
    with open(wav_path, "rb") as wav_binary:
        wav_content = wav_binary.read()
        result = client.do_ww_verification(
            client.encode_wav_base64_request(wav_content)
        )

    st.markdown("____________________________________")
    st.text_input("Result", value=str(result))
