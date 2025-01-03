#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, List
from pydantic import BaseModel, Field


with open("base64audio_label_1", "rb") as file:
    BASE64 = file.read()


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """

    content: Optional[bytes] = Field(
        ..., example=BASE64, title="Base64 encoded PCM waveform"
    )


class InferenceResult(BaseModel):
    """
    Inference result from the model: As previously defied
    """

    wake_word_probability: float = Field(
        ..., example=0.987526, title="Probablity of keyword"
    )
    prediction: float = Field(..., example=1.0, title="Predicted label")
    false_alarm_probability: float = Field(
        ..., example=0.001, title="False alarm probability"
    )
    decision_threshold: float = Field(..., example=0.55, title="Decision threshold")
    wwvm_version: str = Field(..., example="model-v0.0.1", title="Model version")
    inference_time: float = Field(..., example=0.2560, title="Inference time")


class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """

    error: bool = Field(..., example=False, title="Whether there is error")
    result: InferenceResult = ...


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """

    error: bool = Field(..., example=True, title="Whether there is error")
    message: str = Field(..., example="", title="Error message")
    traceback: str = Field(None, example="", title="Detailed traceback of the error")
