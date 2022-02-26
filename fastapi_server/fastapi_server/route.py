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
from fastapi_server.preprocessing import Preprocessing, DeepSpeechModel

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


class AudioBase64(BaseModel):
    audio_name: str = Field(..., min_length=1, example="my-audio")
    base64_audio: str = Field(
        ...,
        min_length=1,
        # example=""
    )


@app.post(
    path="/upload-base64-audios/",
    status_code=status.HTTP_200_OK,
    description="Here you can upload audios in base64 format for transcripting them.",
    tags=["Upload base64-format audios"],
)
async def upload_base64_audios(audios: list[AudioBase64] = Body(...)):
    # all_names: list[str] = []
    # for i in range(len(audios)):
    #     names = audios[i].audio_name
    #     all_names.append(names)
    # return f"You have uploaded {len(all_names)} audios which names are: {all_names}"

    model: str = "~/audio_transcript_fastapi_server/fastapi_server/models"
    dir_name = os.path.expanduser(model)
    """Resolves all the paths of model files by instantiating DeepSpeechModel class
    """
    ds_instance = DeepSpeechModel(dir_name)
    output_graph, scorer = ds_instance.resolve_models_paths()
    """Loads output_graph and scorer into load_models method from DeepSpeechModel class
    """
    model_retval: list[str] = ds_instance.load_models(output_graph, scorer)

    transcriptions: list[str] = []
    new_data: list[str] = []
    final_data: list[str] = []
    header: list[str] = ["audio_name", "transcriptions"]

    aggresive = 1
    filenames = [audio.filename for audio in audios]
    # audio_data = [audio.file for audio in audios]

    for k in range(len(audios)):
        # audio_data = audios[k].file
        # wave_file = audio_data
        # wave_instance = Preprocessing(wave_file, 30, 300, aggresive)
        # segments, sample_rate, audio_length, vad = wave_instance.vad_segment_generator(
        #     wave_file, aggresive
        # )
        # return sample_rate, audio_length
        audstr = audios[k]
