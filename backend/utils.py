import requests

def get_nasa_irradiance(lat, lon):
    """Fetches daily solar irradiance (kWh/m²/day) from NASA."""
    try:
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN&community=RE&longitude={lon}&latitude={lat}&start=20240101&end=20240101&format=JSON"
        response = requests.get(url, timeout=5)
        data = response.json()
        return list(data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].values())[0]
    except Exception:
        return 5.2 # Default average if API is throttled