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

import fastapi_server.preprocessing_f as pr

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
    data_base64: str = Field(
        ...,
        min_length=1,
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
    output_graph, scorer = pr.resolve_models(dir_name)
    model_retval: List[str] = pr.load_model(output_graph, scorer)

    aggresive = 1
    filenames = [audio.filename for audio in audios]
    transcriptions: list[str] = []
    new_data: list[str] = []
    final_data: list[str] = []
    header: list[str] = ["audio_name", "transcriptions"]

    for i in range(len(audios)):
        audio_data = audios[i].file
        wave_file = audio_data
        # print("OJO")
        # print(audios[i].filename)

        segments, sample_rate, audio_length = pr.vad_segment_generator(
            wave_file, aggresive
        )
        for k, segment in enumerate(segments):
            audio = np.frombuffer(segment, dtype=np.int16)
            output = pr.stt(model_retval[0], audio)
            transcript = output[0]
            # print("AQUI")
            # print(transcript)
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


@app.post(
    path="/upload-base64-audios/",
    status_code=status.HTTP_200_OK,
    description="Here you can upload audios in base64 format for transcripting them.",
    tags=["Upload base64-format audios"],
)
async def upload_base64_audios(audios: list[AudioBase64] = Body(...)):
    all_names: list[str] = []
    all_datas: list[str] = []
    all_decode: list[str] = []
    for i in range(len(audios)):
        name = audios[i].audio_name
        data = audios[i].data_base64
        decode = base64.b64decode(data)
        all_names.append(name)
        all_datas.append(data)
        all_decode.append(decode)
        filename = "%s.wav" % name
        print(all_decode)
        with open(filename, "wb") as f:
            f.write(decode)

        # prcm_data, sample_rate, audio_length = pr.read_wave(f.write(decode))
    print("ALL DATAS")
    # print(all_datas)
    print("MY FILE")
    # print(my_file)
    # print(type(my_file))
    print("READ WAVE INFO")
    # print(sample_rate)
    # print(decodes)
    return f"You have uploaded {len(all_names)} audios which names are: {all_names}, which info is: {all_datas}"


# imgstr = base64_image
# imgdata = base64.b64decode(imgstr)
# name = "au1"
# filename = "%s.wav" % name
# with open(filename, "wb") as f:
#     f.write(imgdata)
