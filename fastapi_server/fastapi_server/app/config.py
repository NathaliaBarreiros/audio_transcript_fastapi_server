from pydantic import BaseSettings


class Settings(BaseSettings):
    jwt_secret: str

    class Config:
        env_file = ".env"


settings = Settings()
