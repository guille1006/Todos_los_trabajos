import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Crear la ventana principal de Tkinter
root = tk.Tk()
root.title("Gráfica en Tkinter con Matplotlib")
root.geometry("600x500")

# Crear una figura de Matplotlib
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)

# Generar datos y crear una gráfica en la figura
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)
ax.set_title("Gráfica de Seno")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Convertir la figura de Matplotlib en un widget de Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Ejecutar la aplicación de Tkinter
root.mainloop()
