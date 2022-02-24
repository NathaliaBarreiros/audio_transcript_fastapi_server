# Python
from email.mime import audio
from importlib.resources import path
from os import stat
from typing import Optional
import base64

# Pydantic
from pydantic import BaseModel
from pydantic import Field

# FastAPI
from fastapi import FastAPI, Header, UploadFile, File
from fastapi import status
from fastapi import Body, Query, Path, Form
from starlette.types import Message


class AudioBase64(BaseModel):
    audio_name: str = Field(..., min_length=1, example="my-audio")
    base64_audio: str = Field(
        ...,
        min_length=1,
        # example=""
    )


app = FastAPI(
    title="Audio Transcript Service",
    description="This is an audio transcript service based on a FastAPI server and Deepspeech for transcripting feature.",
    version="1.0.1",
)


@app.get(path="/", tags=["Audio Uploader & Transcripter"])
async def home() -> str:
    return (
        "Hey, you're using an audio transcript service based on FastAPI and Deepspeech!"
    )


@app.post(
    path="/upload-audios/",
    status_code=status.HTTP_200_OK,
    description="Here you can upload .wav audios you want to transcript.",
    tags=["Upload .wav audios"],
)
async def upload_audios(audios: list[UploadFile] = File(...)):
    filenames = [audio.filename for audio in audios]

    return f"You have uploaded {len(audios)} audios which names are: {filenames}"


@app.post(
    path="/upload-base64-audios/",
    status_code=status.HTTP_200_OK,
    description="Here you can upload audios in base64 format for transcripting them.",
    tags=["Upload base64-format audios"],
)
async def upload_base64_audios(audios: list[AudioBase64] = Body(...)):
    names_final: list[str] = []
    for i in range(len(audios)):
        names = audios[i].audio_name
        names_final.append(names)
    return f"You have uploaded {len(names_final)} audios which names are: {names_final}"


# EX:


@app.post(path="/post-audio/")
async def post_audio(audio: UploadFile = File(...)):

    return {
        "Filename": audio.filename,
        "Format": audio.content_type,
        "Size(kb)": round(len(audio.file.read()) / 1024, ndigits=2),
        "content": audio.file.read(),
    }


@app.post("/files/")
async def create_files(files: list[bytes] = File(...)):
    return {"file_sizes": [len(file) for file in files]}
