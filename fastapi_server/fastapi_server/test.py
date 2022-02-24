# Python
from email import message
from os import stat
from typing import Optional
from enum import Enum
from attr import field

# Pydantic
from pydantic import BaseModel
from pydantic import Field
from pydantic import EmailStr

# FastAPI
from fastapi import FastAPI, Header, Cookie, UploadFile, File
from fastapi import status
from fastapi import Body, Query, Path, Form
from starlette.types import Message

app = FastAPI(title="Upload file tester")


class LoginOut(BaseModel):
    username: str = Field(..., max_length=20, example="Migue2022")
    message: str = Field(default="Login Succesfully!")


@app.post(path="/login/", response_model=LoginOut, status_code=status.HTTP_200_OK)
def login(
    username: str = Form(...),
    password: str = Form(...),
):
    return LoginOut(username=username)


# Cookies and Headers Parameters
@app.post(path="/contact/", status_code=status.HTTP_200_OK)
def contact(
    first_name: str = Form(
        ...,
        max_length=20,
        min_length=1,
    ),
    last_name: str = Form(
        ...,
        max_length=20,
        min_length=1,
    ),
    email: EmailStr = Form(...),
    message: str = Form(..., min_length=20),
    user_agent: Optional[str] = Header(default=None),
    ads: Optional[str] = Cookie(default=None),
):
    return user_agent


# Fields
@app.post(path="/post-image/")
def post_image(image: UploadFile = File(...)):
    return {
        "Filename": image.filename,
        "Format": image.content_type,
        "Size(kb)": round(len(image.file.read()) / 1024, ndigits=2),
    }
