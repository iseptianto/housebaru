from pydantic import BaseModel, Field
from typing import Optional

class OLXPredictionRequest(BaseModel):
    # numerik wajib
    LB: float = Field(..., gt=0, description="Luas Bangunan (m²)")
    LT: float = Field(..., gt=0, description="Luas Tanah (m²)")
    KM: int   = Field(..., ge=0, description="Kamar Mandi")
    KT: int   = Field(..., ge=0, description="Kamar Tidur")

    # kategorikal wajib (alias = nama kolom CSV persis)
    kota_kab: str = Field(..., alias="Kota/Kab")
    provensi: str = Field(..., alias="Provensi")
    typpe: str    = Field(..., alias="tyype")

    # fitur opsional (kalau ada di pipeline)
    harga_per_m2: Optional[float] = Field(None, alias="harga_per_m2")
    ratio_bangunan_ruma: Optional[float] = Field(None, alias="ratio_bangunan ruma")

    class Config:
        allow_population_by_alias = True
        anystr_strip_whitespace = True

class PredictionResponse(BaseModel):
    prediction: float
    prediction_time: str
