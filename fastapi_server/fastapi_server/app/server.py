import os
import io
from black import out
import jwt
import base64
import mimetypes
import magic
import numpy as np
import pandas as pd
from fastapi import (
    FastAPI,
    status,
    UploadFile,
    File,
    HTTPException,
    Response,
    Depends,
    Body,
)
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from tortoise.contrib.fastapi import register_tortoise
from passlib.hash import bcrypt
from fastapi_server.app.preprocessing import Preprocessing as pr
from fastapi_server.app.preprocessing import DeepSpeechModel as ds
from fastapi_server.models.user_model import (
    User,
    User_Pydantic,
    UserIn_Pydantic,
    UserShow,
)
from fastapi_server.models.audiobase64_model import AudioBase64
from fastapi_server.app.config import settings

app = FastAPI(
    title="Audio Transcript Service",
    description="This is an audio transcript service based on a FastAPI web framework and Deepspeech for transcripting feature. This application accept either .wav and base64-format audios.",
    version="1.0.1",
)


oauth2_schema = OAuth2PasswordBearer(tokenUrl="token")


async def authenticate_user(
    username: str,
    password: str,
):
    user = await User.get(username=username)
    if not user:
        return False
    if not user.verify_password(password):
        return False
    return user


async def get_current_user(token: str = Depends(oauth2_schema)):
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        user = await User.get(id=payload.get("id"))
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    return await User_Pydantic.from_tortoise_orm(user)


def get_extension_base64(audios: list[AudioBase64]):
    extensions: list[str] = []
    bool_value = True
    for j in range(len(audios)):
        data = audios[j].base64_str
        decode = base64.b64decode(data)
        mime_type = magic.from_buffer(decode, mime=True)
        file_ext = mimetypes.guess_extension(mime_type)
        extensions.append(file_ext)
    for extension in extensions:
        if extension != ".wav":
            bool_value = False

    return bool_value


@app.post(
    path="/token/",
    status_code=status.HTTP_200_OK,
    description="In order to authenticate you, a JSON web token is generated entering your correct username and password.",
    tags=["Generate a JWT"],
)
async def generate_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    user_obj = await User_Pydantic.from_tortoise_orm(user)
    my_user: dict = user_obj.dict()
    del my_user["password_hash"]
    token = jwt.encode(
        my_user,
        settings.jwt_secret,
    )

    return {"access_token": token, "token_type": "bearer"}


@app.post(
    path="/users/",
    response_model=UserShow,
    status_code=status.HTTP_200_OK,
    description="Enter user information, specify username, email, password.",
    tags=["Create a user"],
)
async def create_user(user: UserIn_Pydantic):
    user_obj = User(
        username=user.username,
        email=user.email,
        password_hash=bcrypt.hash(user.password_hash),
    )
    await user_obj.save()
    return await User_Pydantic.from_tortoise_orm(user_obj)


@app.post(
    path="/upload-audios/",
    status_code=status.HTTP_200_OK,
    description="Upload .wav audios you want to transcript.",
    tags=["Upload .wav audios"],
)
async def upload_audios(
    audios: list[UploadFile] = File(...),
    user: User_Pydantic = Depends(get_current_user),
):
    for j in range(len(audios)):
        if audios[j].content_type != "audio/wav":
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Uploaded files are not .wav-format audios.",
            )

    model: str = "~/audio_transcript_fastapi_server/fastapi_server/deepspeech_models"
    dir_name = os.path.expanduser(model)
    ds_instance = ds(dir_name)
    output_graph, scorer = ds_instance.resolve_models_paths()
    model_retval: list[str] = ds_instance.load_models(output_graph, scorer)

    aggresive = 1
    filenames: list[str] = [audio.filename for audio in audios]
    transcriptions: list[str] = []
    new_data: list[str] = []
    final_data: list[str] = []
    headers: list[str] = ["audio_name", "transcriptions"]

    for i in range(len(audios)):
        audio_data = audios[i].file
        wave_file = audio_data
        wave_instance = pr(wave_file, 30, 300, aggresive)
        (
            segments,
            sample_rate,
            audio_length,
            vad,
        ) = await wave_instance.vad_segment_generator(wave_file, aggresive)

        for k, segment in enumerate(segments):
            audio = np.frombuffer(segment, dtype=np.int16)
            output = await ds_instance.transcript_audio_segments(model_retval[0], audio)
            transcript = output[0]
        transcriptions.append(transcript)
        new_data = [filenames[i], transcriptions[i]]
        final_data.append(new_data)

    new_df = pd.DataFrame(final_data, columns=headers)
    stream = io.StringIO()
    new_df.to_csv(stream, index=False)
    response: Response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        status_code=200,
    )
    response.headers["Content-Disposition"] = "attachment; filename=transcriptions.csv"

    return response


@app.post(
    path="/upload-base64-audios/",
    status_code=status.HTTP_200_OK,
    description="Upload audios in base64 format for transcripting them. Enter an audio name and its Base64 encoding string. For entering more audio data, just copy the dictionary of the example value given separated by a comma. When you finish adding your data you should have a list of key-value dictionaries corresponding to audio_name and base64_str information.",
    tags=["Upload base64-format audios"],
)
async def upload_base64_audios(
    audios: list[AudioBase64] = Body(...),
    user: User_Pydantic = Depends(get_current_user),
):
    if get_extension_base64(audios) == False:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Data uploaded do not correspond to Base64 encoding from .wav audio files.",
        )

    model: str = "~/audio_transcript_fastapi_server/fastapi_server/deepspeech_models"
    dir_name = os.path.expanduser(model)
    ds_instance = ds(dir_name)
    output_graph, scorer = ds_instance.resolve_models_paths()
    model_retval: list[str] = ds_instance.load_models(output_graph, scorer)

    aggresive = 1
    all_names: list[str] = []
    all_datas: list[str] = []
    all_decode: list[str] = []
    transcriptions: list[str] = []
    new_data: list[str] = []
    final_data: list[str] = []
    headers: list[str] = ["audio_name", "transcriptions"]

    for i in range(len(audios)):
        name = audios[i].audio_name
        data = audios[i].base64_str
        decode = base64.b64decode(data)
        all_names.append(name)
        all_datas.append(data)
        all_decode.append(decode)

        with io.BytesIO() as buffer:
            buffer.write(decode)
            buffer.seek(0)

            wave_instance = pr(buffer, 30, 300, aggresive)
            (
                segments,
                sample_rate,
                audio_length,
                vad,
            ) = await wave_instance.vad_segment_generator(buffer, aggresive)

            for k, segment in enumerate(segments):
                audio = np.frombuffer(segment, dtype=np.int16)
                output = await ds_instance.transcript_audio_segments(
                    model_retval[0], audio
                )
                transcript = output[0]
            transcriptions.append(transcript)
            new_data = [all_names[i], transcriptions[i]]
            final_data.append(new_data)

    new_df = pd.DataFrame(final_data, columns=headers)
    stream = io.StringIO()
    new_df.to_csv(stream, index=False)
    response: Response = StreamingResponse(
        iter([stream.getvalue()]), media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=my-file.csv"
    return response


register_tortoise(
    app,
    db_url="sqlite://users.sqlite3",
    modules={"models": ["server"]},
    generate_schemas=True,
    add_exception_handlers=True,
)
