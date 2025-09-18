import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from perceptron import PerceptronSimple
from data_loader import DataLoader

class PerceptronApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptrón Simple - Implementación desde Cero")
        self.root.geometry("1000x700")
        
        self.perceptron = None
        self.data_loader = DataLoader()
        
        self.setup_ui()
    
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Sección de carga de datos
        ttk.Label(main_frame, text="Carga de Dataset", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        
        ttk.Button(main_frame, text="Cargar desde Archivo Local", 
                  command=self.load_local_file).grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(main_frame, text="Cargar desde URL", 
                  command=self.load_from_url).grid(row=1, column=1, padx=5, pady=5)
        
        # Información del dataset
        self.dataset_info = tk.Text(main_frame, height=5, width=50)
        self.dataset_info.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Sección de configuración
        ttk.Label(main_frame, text="Configuración del Perceptrón", font=('Arial', 12, 'bold')).grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Label(main_frame, text="Pesos iniciales (separados por coma):").grid(row=4, column=0, sticky=tk.W, padx=5)
        self.weights_entry = ttk.Entry(main_frame, width=30)
        self.weights_entry.grid(row=4, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Label(main_frame, text="Umbral inicial:").grid(row=5, column=0, sticky=tk.W, padx=5)
        self.threshold_entry = ttk.Entry(main_frame, width=30)
        self.threshold_entry.grid(row=5, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Label(main_frame, text="Tasa de aprendizaje:").grid(row=6, column=0, sticky=tk.W, padx=5)
        self.learning_rate_entry = ttk.Entry(main_frame, width=30)
        self.learning_rate_entry.insert(0, "0.1")
        self.learning_rate_entry.grid(row=6, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Label(main_frame, text="Máx iteraciones:").grid(row=7, column=0, sticky=tk.W, padx=5)
        self.max_iter_entry = ttk.Entry(main_frame, width=30)
        self.max_iter_entry.insert(0, "100")
        self.max_iter_entry.grid(row=7, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Label(main_frame, text="Error máximo permitido:").grid(row=8, column=0, sticky=tk.W, padx=5)
        self.max_error_entry = ttk.Entry(main_frame, width=30)
        self.max_error_entry.insert(0, "0.01")
        self.max_error_entry.grid(row=8, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
        
        # Botones de control
        ttk.Button(main_frame, text="Inicializar Perceptrón", 
                  command=self.initialize_perceptron).grid(row=9, column=0, padx=5, pady=10)
        
        ttk.Button(main_frame, text="Iniciar Entrenamiento", 
                  command=self.start_training).grid(row=9, column=1, padx=5, pady=10)
        
        # Gráfica
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=10, column=0, columnspan=2, padx=5, pady=10, sticky=(tk.W, tk.E))
        
        # Sección de prueba
        ttk.Label(main_frame, text="Prueba del Modelo", font=('Arial', 12, 'bold')).grid(row=11, column=0, columnspan=2, pady=10)
        
        ttk.Label(main_frame, text="Patrón de entrada (separado por coma):").grid(row=12, column=0, sticky=tk.W, padx=5)
        self.test_entry = ttk.Entry(main_frame, width=30)
        self.test_entry.grid(row=12, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Button(main_frame, text="Probar Patrón", 
                  command=self.test_pattern).grid(row=13, column=0, columnspan=2, padx=5, pady=10)
        
        self.result_label = ttk.Label(main_frame, text="Resultado: ")
        self.result_label.grid(row=14, column=0, columnspan=2, padx=5, pady=5)
    
    def load_local_file(self):
        success = self.data_loader.load_local_file()
        if success:
            success, info_text = self.data_loader._process_data()
            self.dataset_info.delete(1.0, tk.END)
            self.dataset_info.insert(tk.END, info_text)
    
    def load_from_url(self):
        success = self.data_loader.load_from_url()
        if success:
            success, info_text = self.data_loader._process_data()
            self.dataset_info.delete(1.0, tk.END)
            self.dataset_info.insert(tk.END, info_text)
    
    def initialize_perceptron(self):
        X, y = self.data_loader.get_data()
        if X is None:
            messagebox.showerror("Error", "Primero cargue un dataset")
            return
        
        n_inputs = X.shape[1]
        self.perceptron = PerceptronSimple(n_inputs)
        
        # Obtener parámetros de la interfaz
        try:
            weights = None
            if self.weights_entry.get():
                weights = list(map(float, self.weights_entry.get().split(',')))
                if len(weights) != n_inputs:
                    messagebox.showerror("Error", f"Debe ingresar {n_inputs} pesos")
                    return
            
            threshold = None
            if self.threshold_entry.get():
                threshold = float(self.threshold_entry.get())
            
            learning_rate = float(self.learning_rate_entry.get())
            
            self.perceptron.initialize_parameters(weights, threshold, learning_rate)
            
            messagebox.showinfo("Éxito", "Perceptrón inicializado correctamente")
            
        except ValueError:
            messagebox.showerror("Error", "Verifique los valores ingresados")
    
    def start_training(self):
        if self.perceptron is None:
            messagebox.showerror("Error", "Primero inicialice el perceptrón")
            return
        
        X, y = self.data_loader.get_data()
        if X is None:
            messagebox.showerror("Error", "No hay datos cargados")
            return
        
        try:
            max_iterations = int(self.max_iter_entry.get())
            max_error = float(self.max_error_entry.get())
            
            # Entrenamiento
            success, iterations, final_error = self.perceptron.train(X, y, max_iterations, max_error)
            
            # Actualizar gráfica
            self.ax.clear()
            self.ax.plot(self.perceptron.error_history, 'b-')
            self.ax.set_xlabel('Iteraciones')
            self.ax.set_ylabel('Error promedio')
            self.ax.set_title('Evolución del Error durante el Entrenamiento')
            self.ax.grid(True)
            self.canvas.draw()
            
            if success:
                messagebox.showinfo("Éxito", 
                                  f"Entrenamiento completado en {iterations} iteraciones\n"
                                  f"Error final: {final_error:.6f}\n"
                                  f"Pesos finales: {self.perceptron.weights}\n"
                                  f"Umbral final: {self.perceptron.threshold}")
            else:
                messagebox.showwarning("Advertencia", 
                                     f"Se alcanzó el máximo de iteraciones\n"
                                     f"Error final: {final_error:.6f}")
            
        except ValueError:
            messagebox.showerror("Error", "Verifique los parámetros de entrenamiento")
    
    def test_pattern(self):
        if self.perceptron is None:
            messagebox.showerror("Error", "Primero entrene el perceptrón")
            return
        
        try:
            pattern = list(map(float, self.test_entry.get().split(',')))
            if len(pattern) != self.perceptron.n_inputs:
                messagebox.showerror("Error", f"El patrón debe tener {self.perceptron.n_inputs} valores")
                return
            
            prediction = self.perceptron.predict(pattern)
            self.result_label.config(text=f"Resultado: {prediction}")
            
        except ValueError:
            messagebox.showerror("Error", "Verifique el formato del patrón")