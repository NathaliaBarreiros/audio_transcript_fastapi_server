# Python
from email import message
from os import stat
from typing import Optional
from enum import Enum
from attr import field

# Pydantic
from pydantic import BaseModel
from pydantic import Field

# FastAPI
from fastapi import FastAPI
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
