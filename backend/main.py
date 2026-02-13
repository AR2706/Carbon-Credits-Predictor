import pickle
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SolarRequest(BaseModel):
    lat: float
    lon: float
    area: float
    eff: float
    days: int

# Load the ML Model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Model Error: {e}")
    model = None

def get_irradiance(lat, lon):
    try:
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN&community=RE&longitude={lon}&latitude={lat}&start=20240101&end=20240101&format=JSON"
        response = requests.get(url, timeout=5)
        return list(response.json()['properties']['parameter']['ALLSKY_SFC_SW_DWN'].values())[0]
    except:
        return 5.25 # Fallback average

@app.post("/api/calculate")
async def calculate_credits(data: SolarRequest):
    # 1. Fetch Irradiance
    irradiance = get_irradiance(data.lat, data.lon)
    
    # 2. Predict Grid Emission Factor (Z) in kg/kWh
    if model:
        features = pd.DataFrame([{
            'Latitude': data.lat, 'Longitude': data.lon,
            'State Name': 'Default', 'Region': 'Default',
            'Tech_Clean': 'Subcritical', 'Age': 15,
            'Coal_Intensity': 0.8, 'Import_Ratio': 0.1, 'Load_Factor': 0.7
        }])
        z_factor = model.predict(features)[0]
    else:
        z_factor = 0.82 # Average grid intensity fallback

    # 3. Solar Generation (kWh)
    # Formula: Irradiance * Area * Efficiency * Performance Ratio (0.75)
    daily_kwh = irradiance * data.area * data.eff * 0.75
    total_kwh = daily_kwh * data.days
    
    # 4. Carbon Credit Conversion
    # Total kg avoided = kWh * Z. 
    # Credits = kg / 1000 (Metric Tonnes)
    co2_avoided_kg = total_kwh * z_factor
    carbon_credits = co2_avoided_kg / 1000

    return {
        "total_yield_kwh": round(total_kwh, 2),
        "co2_avoided_kg": round(co2_avoided_kg, 2),
        "carbon_credits": round(carbon_credits, 6), # Official Metric Tonne Value
        "z_factor": round(z_factor, 3),
        "days": data.days
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)