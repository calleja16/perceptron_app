import tkinter as tk
from gui import PerceptronApp

def main():
    root = tk.Tk()            #Crear ventana principal
    app = PerceptronApp(root) #Crea aplicacion
    root.mainloop()           #Inicia el loop de la aplicacion

if __name__ == "__main__":
    main()

    