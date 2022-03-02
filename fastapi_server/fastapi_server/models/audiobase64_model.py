from pydantic import BaseModel, Field


class AudioBase64(BaseModel):
    audio_name: str = Field(
        ...,
        min_length=1,
        example="my-audio",
    )
    base64_str: str = Field(
        ...,
        min_length=1,
    )
