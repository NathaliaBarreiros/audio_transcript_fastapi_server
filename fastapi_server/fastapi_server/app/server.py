import os
import io
from urllib import response
import numpy as np
import pandas as pd
from fastapi import FastAPI, status, UploadFile, File, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi_server.app.preprocessing import Preprocessing as pr
from fastapi_server.app.preprocessing import DeepSpeechModel as dps

app = FastAPI(
    title="Audio Transcript Service",
    description="This is an audio transcript service based on a FastAPI web framework and Deepspeech for transcripting feature. This application accept either .wav and base64-format audios.",
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
                detail="Uploaded files are not .wav-format audios.",
            )

    model: str = "~/audio_transcript_fastapi_server/fastapi_server/deepspeech_models"
    dir_name = os.path.expanduser(model)
    ds_instance = dps(dir_name)
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
        segments, sample_rate, audio_length, vad = wave_instance.vad_segment_generator(
            wave_file, aggresive
        )

        for k, segment in enumerate(segments):
            audio = np.frombuffer(segment, dtype=np.int16)
            output = ds_instance.transcript_audio_segments(model_retval[0], audio)
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
