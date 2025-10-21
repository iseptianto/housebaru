from pydantic import BaseModel, Field
from typing import Optional

class OLXPredictionRequest(BaseModel):
    LB: float = Field(..., gt=0)
    LT: float = Field(..., gt=0)
    KM: int   = Field(..., ge=0)
    KT: int   = Field(..., ge=0)

    kota_kab: str = Field(..., alias="Kota/Kab")
    provinsi: str = Field(..., alias="Provensi")
    type_: str    = Field(..., alias="tyype")

    harga_per_m2: Optional[float] = Field(None, alias="harga_per_m2")
    ratio_bangunan_ruma: Optional[float] = Field(None, alias="ratio_bangunan ruma")

    class Config:
        allow_population_by_alias = True
        anystr_strip_whitespace = True

class PredictionResponse(BaseModel):
    prediction: float
    prediction_time: str
