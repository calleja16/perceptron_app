import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading, requests, io

# ------------------ Clase Perceptrón ------------------
class PerceptronSimple:
    def __init__(self):
        self.weights = None
        self.threshold = 0
        self.lr = 0.1
        self.max_iter = 100
        self.max_error = 0.01
        self.error_history = []
        self.trained = False

    def step(self, x): return 1 if x >= 0 else 0

    def initialize(self, n_features):
        self.weights = np.random.uniform(-1, 1, n_features)
        self.threshold = np.random.uniform(-1, 1)
        self.error_history = []
        self.trained = False

    def train(self, X, y, callback=None):
        self.error_history.clear()
        for epoch in range(self.max_iter):
            total_err = 0
            for xi, target in zip(X, y):
                net = np.dot(xi, self.weights) - self.threshold
                out = self.step(net)
                error = target - out
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.threshold -= self.lr * error
                total_err += abs(error)
            avg = total_err / len(y)
            self.error_history.append(avg)
            if callback: callback(epoch+1, avg)
            if avg <= self.max_error:
                self.trained = True
                return f"✅ Entrenado en {epoch+1} iteraciones"
        self.trained = True
        return f"⚠️ Alcanzado máx iteraciones"

    def predict(self, X):
        if X.ndim == 1: X = X.reshape(1, -1)
        return np.array([self.step(np.dot(xi, self.weights)-self.threshold) for xi in X])

# ------------------ Interfaz gráfica ------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptrón Simple")
        self.root.geometry("1100x600")

        self.model = PerceptronSimple()
        self.X = self.y = None

        self.build_ui()

    def build_ui(self):
        main = tk.Frame(self.root, bg="#e8e8e8")
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, width=350, bg="#f0f0f0", padx=10, pady=10)
        left.pack(side="left", fill="y")
        right = tk.Frame(main, bg="white", padx=10, pady=10)
        right.pack(side="right", fill="both", expand=True)

        # -------- Dataset --------
        tk.Label(left, text="📁 Dataset", font=("Segoe UI",12,"bold"), bg="#f0f0f0").pack(anchor="w")
        ttk.Button(left, text="Cargar Dataset Local", command=self.load_dataset_local).pack(fill="x", pady=3)

        tk.Label(left, text="o pegar URL de Drive:", bg="#f0f0f0").pack(anchor="w", pady=(8,0))
        self.url_entry = tk.Entry(left)
        self.url_entry.pack(fill="x", pady=2)
        ttk.Button(left, text="Cargar desde URL", command=self.load_dataset_url).pack(fill="x", pady=3)

        self.dataset_info = tk.Label(left, text="No se ha cargado ningún dataset", bg="#f0f0f0", wraplength=300)
        self.dataset_info.pack(anchor="w", pady=2)

        # -------- Configuración --------
        tk.Label(left, text="\n⚙️ Configuración del Modelo", font=("Segoe UI",12,"bold"), bg="#f0f0f0").pack(anchor="w")

        form = tk.Frame(left, bg="#f0f0f0")
        form.pack(fill="x")
        tk.Label(form,text="Tasa de aprendizaje (η):",bg="#f0f0f0").grid(row=0,column=0,sticky="w")
        self.lr = tk.DoubleVar(value=0.1)
        tk.Entry(form,textvariable=self.lr).grid(row=0,column=1,sticky="we")

        tk.Label(form,text="Máx Iteraciones:",bg="#f0f0f0").grid(row=1,column=0,sticky="w")
        self.max_iter = tk.IntVar(value=100)
        tk.Entry(form,textvariable=self.max_iter).grid(row=1,column=1,sticky="we")

        tk.Label(form,text="Error máximo (ε):",bg="#f0f0f0").grid(row=2,column=0,sticky="w")
        self.max_error = tk.DoubleVar(value=0.01)
        tk.Entry(form,textvariable=self.max_error).grid(row=2,column=1,sticky="we")

        # -------- Entrenamiento --------
        tk.Label(left, text="\n🚀 Entrenamiento", font=("Segoe UI",12,"bold"), bg="#f0f0f0").pack(anchor="w")
        self.train_btn = tk.Button(left, text="Entrenar Modelo", bg="#4CAF50", fg="white", command=self.start_training)
        self.train_btn.pack(fill="x", pady=5)

        # -------- Pruebas --------
        tk.Label(left, text="\n🧪 Pruebas del Modelo", font=("Segoe UI",12,"bold"), bg="#f0f0f0").pack(anchor="w")

        tk.Label(left,text="Patrón manual:",bg="#f0f0f0").pack(anchor="w")
        self.manual_entry = tk.Entry(left)
        self.manual_entry.pack(fill="x", pady=2)
        ttk.Button(left, text="Probar Patrón Manual", command=self.test_manual).pack(fill="x")

        tk.Label(left,text="Probar desde dataset:",bg="#f0f0f0").pack(anchor="w", pady=(10,0))
        self.row_var = tk.StringVar()
        self.row_combo = ttk.Combobox(left, textvariable=self.row_var, state="readonly")
        self.row_combo.pack(fill="x")
        ttk.Button(left, text="Probar Fila del Dataset", command=self.test_row).pack(fill="x", pady=2)

        # -------- Resultado --------
        tk.Label(left, text="\n📋 Resultado:", font=("Segoe UI",12,"bold"), bg="#f0f0f0").pack(anchor="w")
        self.result_lbl = tk.Label(left, text="Aquí aparecerá el resultado de la predicción", bg="#dff0d8", wraplength=300, justify="left")
        self.result_lbl.pack(fill="x", pady=3)

        # -------- Gráfica --------
        self.fig, self.ax = plt.subplots(figsize=(6,4))
        self.ax.set_title("Evolución del Error durante el Entrenamiento")
        self.ax.set_xlabel("Iteraciones")
        self.ax.set_ylabel("Error")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # -------- Funciones --------
    def load_dataset_local(self):
        path = filedialog.askopenfilename(filetypes=[("CSV","*.csv"),("Excel","*.xlsx")])
        if not path: return
        df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
        self.X = df.iloc[:,:-1].astype(float).values
        self.y = df.iloc[:,-1].astype(int).values
        self.row_combo['values'] = [f"Fila {i}: {list(row)} → {self.y[i]}" for i,row in enumerate(self.X)]
        self.dataset_info.config(text=f"{path.split('/')[-1]} | {len(self.X)} filas, {self.X.shape[1]} entradas")

    def load_dataset_url(self):
        url = self.url_entry.get().strip()
        if "drive.google.com" in url:
            # Convertir link compartido de Drive a link de descarga directa
            file_id = url.split("/d/")[1].split("/")[0]
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
        try:
            content = requests.get(url).content
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            self.X = df.iloc[:,:-1].astype(float).values
            self.y = df.iloc[:,-1].astype(int).values
            self.row_combo['values'] = [f"Fila {i}: {list(row)} → {self.y[i]}" for i,row in enumerate(self.X)]
            self.dataset_info.config(text=f"(URL) {len(self.X)} filas, {self.X.shape[1]} entradas")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar: {e}")

    def start_training(self):
        if self.X is None:
            messagebox.showerror("Error","Primero carga un dataset"); return
        self.model.lr = self.lr.get()
        self.model.max_iter = self.max_iter.get()
        self.model.max_error = self.max_error.get()
        self.model.initialize(self.X.shape[1])
        def train():
            result = self.model.train(self.X, self.y, self.update_graph)
            self.result_lbl.config(text=result)
        threading.Thread(target=train, daemon=True).start()

    def update_graph(self, ep, err):
        self.ax.clear()
        self.ax.set_title("Evolución del Error durante el Entrenamiento")
        self.ax.set_xlabel("Iteraciones"); self.ax.set_ylabel("Error")
        self.ax.plot(range(1,len(self.model.error_history)+1), self.model.error_history, 'bo-')
        self.canvas.draw()

    def test_manual(self):
        try:
            vals = np.array([float(x) for x in self.manual_entry.get().split(",")])
            pred = self.model.predict(vals)[0]
            self.result_lbl.config(text=f"Patrón {vals} → Predicho: {pred}")
        except:
            messagebox.showerror("Error","Entrada inválida")

    def test_row(self):
        if self.row_combo.current() == -1:
            messagebox.showwarning("Error","Selecciona una fila"); return
        i = self.row_combo.current()
        pred = self.model.predict(self.X[i])[0]
        self.result_lbl.config(text=f"Fila {i} → Esperado: {self.y[i]} | Predicho: {pred}")

# ------------------ Main ------------------
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
