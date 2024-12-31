from channels.generic.websocket import AsyncWebsocketConsumer
import requests


# can reference container via service name, notice: it is the port internally to the serving container
# which we are sending requests to. Shouldnt really be a post request, since we dont change the state of the application at all.
# but current application server is configured to be a post request
URL_KWS_SERIVCE = "http://serving:80/api/v1/predict"


class KWSConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        await self.accept()

    async def receive(self, bytes):
        print(f"Recieved bytes: {bytes}")
        self.socket.send(bytes)

    async def get_spot_result(self, data):
        # data better be base64 encoded
        print(f"get_spot_result(data) ->  data: {data}")
        result = requests.post(URL_KWS_SERIVCE, data=data)
        print("result.status_code: ", result.status_code)
        print("result.status_code: ", result.status_code)
        return result

    # {
    #         "keyword_probability": ww_prob,
    #         "prediction": 1.0 if ww_prob > CONFIG["DECISION_THRESHOLD"] else 0.0,
    #         "false_alarm_probability": 1 - ww_prob,
    #         "decision_threshold": CONFIG["DECISION_THRESHOLD"],
    #         "wwvm_version": CONFIG["MODEL_VERSION"],
    #         "inference_time": f - s,
    #     }

    async def spot(self, data):
        print(f"spot data: {data}")
        result = await self.get_spot_result(data)

        if "channel" in data:
            spot_result = data["channel"]["alternatives"][0]["spot"]

            if spot_result:
                await self.send(spot_result)
