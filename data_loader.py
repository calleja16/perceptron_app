import pandas as pd
import requests
from io import StringIO
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

class DataLoader:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
    
    def load_local_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.data = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    self.data = pd.read_excel(file_path)
                elif file_path.endswith('.json'):
                    self.data = pd.read_json(file_path)
                else:
                    messagebox.showerror("Error", "Formato de archivo no soportado")
                    return False
                
                return self._process_data()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar el archivo: {str(e)}")
                return False
        
        return False
    
    def load_from_url(self):
        url = simpledialog.askstring("Cargar desde URL", "Ingrese la URL del dataset:")
        if url:
            try:
                # Manejar enlaces de Google Drive
                if 'drive.google.com' in url:
                    # Extraer el ID del archivo
                    if '/file/d/' in url:
                        file_id = url.split('/file/d/')[1].split('/')[0]
                    elif 'id=' in url:
                        file_id = url.split('id=')[1].split('&')[0]
                    else:
                        file_id = url.split('/')[-1]
                    
                    # Crear URL de descarga directa
                    download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
                    response = requests.get(download_url)
                else:
                    response = requests.get(url)
                
                response.raise_for_status()
                
                # Intentar determinar el tipo de archivo
                content_type = response.headers.get('content-type', '')
                
                if 'csv' in content_type or url.endswith('.csv'):
                    self.data = pd.read_csv(StringIO(response.text))
                elif 'json' in content_type or url.endswith('.json'):
                    self.data = pd.read_json(StringIO(response.text))
                else:
                    # Intentar auto-detectar
                    try:
                        self.data = pd.read_csv(StringIO(response.text))
                    except:
                        # Si falla, intentar con diferentes codificaciones
                        self.data = pd.read_csv(StringIO(response.text), encoding='latin-1')
                
                return self._process_data()
            
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar desde URL: {str(e)}")
                return False
        
        return False

    #Separa los datos en X e y
    def _process_data(self):
        if self.data is not None:
            # Asumimos que la última columna es la salida
            self.X = self.data.iloc[:, :-1].values
            self.y = self.data.iloc[:, -1].values
            
            info_text = f"Dataset cargado exitosamente:\n"
            info_text += f"Número de patrones: {len(self.data)}\n"
            info_text += f"Número de entradas: {self.X.shape[1]}\n"
            info_text += f"Número de salidas: 1\n"
            info_text += f"Primeras filas:\n{self.data.head().to_string()}"
            
            return True, info_text
        
        return False, "No hay datos cargados"
    
    def get_data(self):
        return self.X, self.y
    
    def get_data_info(self):
        if self.data is not None:
            return f"Dataset: {len(self.data)} patrones, {self.X.shape[1]} entradas"
        return "No hay datos cargados"