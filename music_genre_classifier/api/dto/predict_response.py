from pydantic import BaseModel


class PredictResponse(BaseModel):
    genre: str
