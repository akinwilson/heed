#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from fastapi.testclient import TestClient

from ..main import app


@pytest.fixture
def client():
    # use "with" statement to run "startup" event of FastAPI
    with TestClient(app) as c:
        yield c


def test_main_predict(client):
    """
    Test predction response
    """
    with open("../base64audio_label_1", "rb") as f:
        example = f.read()

    headers = {}
    body = {
        "content": example,
    }

    response = client.post("/api/v1/predict",
                           headers=headers,
                           json=body)

    try:
        assert response.status_code == 200
        reponse_json = response.json()
        assert reponse_json['error'] == False
        assert isinstance(reponse_json['results']['setosa'], float)
        assert isinstance(reponse_json['results']['pred'], str)

    except AssertionError:
        print(response.status_code)
        print(response.json())
        raise
