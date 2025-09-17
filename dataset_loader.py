import pandas as pd
import requests
import io
import re

def load_local(path: str) -> pd.DataFrame:
    """Carga un dataset local en CSV o Excel."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        return pd.read_excel(path)

def load_from_drive(url: str) -> pd.DataFrame:
    """Carga un dataset desde un enlace de Google Drive."""
    # Convertir enlace de visualización a enlace de descarga directa
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError("URL de Drive inválida")
    file_id = match.group(1)
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    resp = requests.get(download_url)
    resp.raise_for_status()
    data = io.StringIO(resp.text)
    return pd.read_csv(data)
