# Python
from email.mime import audio
from importlib.resources import path
import io
import os
from typing import Optional
import base64
from urllib import response
import pandas as pd
import numpy as np

# --
import collections
import contextlib
import wave
import webrtcvad
from deepspeech import Model
from timeit import default_timer as timer
import glob
from typing import List

# import fastapi_server.preprocessing_f as pr
from fastapi_server.preprocessing_poo import Preprocessing as pr
from fastapi_server.preprocessing_poo import DeepSpeechModel as dps

# Pydantic
from pydantic import BaseModel
from pydantic import Field

# FastAPI
from fastapi import FastAPI, Header, Response, UploadFile, File
from fastapi import status
from fastapi import Body, Query, Path, Form
from starlette.types import Message
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi import HTTPException

app = FastAPI(
    title="Audio Transcript Service",
    description="This is an audio transcript service based on a FastAPI server and Deepspeech for transcripting feature.",
    version="1.0.1",
)


@app.post(
    path="/upload-audios/",
    status_code=status.HTTP_200_OK,
    description="Here you can upload .wav audios you want to transcript.",
    tags=["Upload .wav audios"],
)
async def upload_audios(audios: list[UploadFile] = File(...)):
    for j in range(len(audios)):
        if audios[j].content_type != "audio/wav":
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Files uploaded are not .wav audios.",
            )
    """
    Desde aqui -------------
    """
    model: str = "~/audio_transcript_fastapi_server/fastapi_server/models"
    dir_name = os.path.expanduser(model)
    ds_instance = dps(dir_name)
    output_graph, scorer = ds_instance.resolve_models_paths()
    model_retval: list[str] = ds_instance.load_models(output_graph, scorer)

    aggresive = 1
    filenames = [audio.filename for audio in audios]
    transcriptions: list[str] = []
    new_data: list[str] = []
    final_data: list[str] = []
    header: list[str] = ["audio_name", "transcriptions"]

    for i in range(len(audios)):
        audio_data = audios[i].file
        wave_file = audio_data

        # print("INFO")
        # print(audio_data)
        # print(type(audio_data))

        # AQUI CONTINUA LO CORRECTO
        wave_instance = pr(wave_file, 30, 300, aggresive)
        segments, sample_rate, audio_length, vad = wave_instance.vad_segment_generator(
            wave_file, aggresive
        )
        print(segments)
        # print(sample_rate)
        # print(audio_length)
        # print(vad)

        for k, segment in enumerate(segments):
            audio = np.frombuffer(segment, dtype=np.int16)
            output = ds_instance.transcript_audio_segments(model_retval[0], audio)
            transcript = output[0]
        transcriptions.append(transcript)
        new_data = [filenames[i], transcriptions[i]]
        final_data.append(new_data)

    # print("SEGMENTS")
    # print(segments)
    # print("NEW DATA")
    # print(new_data)
    # print("FINAL DATA")
    # print(final_data)
    # print(len(audios))
    # print("TRANCRIPTIONS")
    # print(transcriptions)

    new_df = pd.DataFrame(final_data, columns=header)
    stream = io.StringIO()
    new_df.to_csv(stream, index=False)
    response: Response = StreamingResponse(
        iter([stream.getvalue()]), media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=my-file.csv"
    return response
    # return "ok"
