import tkinter as tkinter
from tkinter import filedialog
from tkinter import messagebox
from knnv2.ui.constants.button.Constants import *
from knnv2.ui.constants.label.Constants import *
from knnv2.ui.constants.window.Constants import *


def browse_file():
    global file_path
    file_path = filedialog.askopenfilename()
    max = 30
    if file_path:
        if len(file_path) > max:
            half_length = int(max / 2)
            truncated_text = file_path[:half_length] + "..."
        else:
            truncated_text = file_path
        choose_label.config(text="File Path: " + truncated_text)
        messagebox.showinfo("Información", "Archivo seleccionado y guardado con éxito: " + file_path)
    else:
        return None


def open_new_window():
    new_window = tkinter.Toplevel(window)
    new_window.title("Nueva Instancia")
    new_window.minsize(X_SIZE, Y_SIZE)

    # Crear un Label en la nueva ventana con la imagen
    new_label = tkinter.Label(new_window)
    new_label.place(x=390, y=20)
    logo = tkinter.PhotoImage(file='images/logo.png')
    new_label.config(image=logo)
    new_label.image = logo  # Es importante mantener una referencia a la imagen


window = tkinter.Tk()
window.minsize(X_SIZE, Y_SIZE)
window.geometry(str(X_SIZE) + "x" + str(Y_SIZE))

# Logo configuration
label = tkinter.Label(window, text="Logo")
logo = tkinter.PhotoImage(file='images/logo.png')
label.place(x=390, y=20)
label.config(image=logo)

# Upload dataset file section
upload_label = tkinter.Label(window, text="Upload dataset file")
upload_label.place(x=145, y=170)

choose_label = tkinter.Label(window, text="Choose file to upload")
choose_label.place(x=150, y=195)

file_path_label = tkinter.Label(window, text="")
file_path_label.place(x=370, y=160)

browse_button = tkinter.Button(window, text="Browse file", command=browse_file)
browse_button.place(x=500, y=190)

# Botón para abrir una nueva ventana
new_instance_label = tkinter.Label(window, text="Crear nueva instancia")
new_instance_label.place(x=145, y=290)
new_window_button = tkinter.Button(window, text="Nueva instancia", command=open_new_window)
new_window_button.place(x=500, y=300)

file_path = None

window.mainloop()




