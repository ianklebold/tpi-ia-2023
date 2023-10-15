import tkinter as tkinter
from tkinter import filedialog
from tkinter import messagebox
from knnv2.ui.constants.button.Constants import *
from knnv2.ui.constants.label.Constants import *
from knnv2.ui.constants.window.Constants import *

global row_counter
row_counter = 0  # Variable global para el contador de filas
max_fields = 5  # Número máximo de campos


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
        return file_path


def add_entry_field():
    global row_counter
    if row_counter < max_fields:
        new_entry = tkinter.Entry(entry_frame)
        new_entry.grid(row=row_counter, column=0, padx=10, pady=5)
        entry_fields.append(new_entry)

        new_button = tkinter.Button(entry_frame, text="+", command=add_entry_field)
        new_button.grid(row=row_counter, column=1, padx=10)

        if row_counter > 0:
            previous_data = entry_fields[row_counter - 1].get()
            entry_fields[row_counter - 1].config(state=tkinter.NORMAL)
            entry_fields[row_counter - 1].delete(0, tkinter.END)
            entry_fields[row_counter - 1].insert(0, previous_data)
            entry_fields[row_counter - 1].config(state=tkinter.DISABLED)

        row_counter += 1
        if row_counter == max_fields:
            add_button.config(state=tkinter.DISABLED)
        save_button.config(state=tkinter.NORMAL)
        save_button.grid(row=row_counter-1) #Actualizamos ubicacion de boton guardar


def save_data():
    data = [entry.get() for entry in entry_fields if entry.get()]
    if data:
        print("Datos ingresados:", data)
    else:
        messagebox.showwarning("Advertencia", "No se han ingresado datos válidos")


window = tkinter.Tk()
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

# Crear un Frame para contener los campos de entrada y botones
entry_frame = tkinter.Frame(window)
entry_frame.place(x=150, y=240)

# Botón para agregar campos de entrada (inicialmente habilitado)
add_button = tkinter.Button(entry_frame, text="+", command=add_entry_field, state=tkinter.NORMAL)
add_button.grid(row=row_counter, column=1, padx=10)

entry_fields = []  # Lista para almacenar los campos de entrada

# Botón para guardar los datos (inicialmente deshabilitado)
save_button = tkinter.Button(entry_frame, text="Guardar", command=save_data, state=tkinter.DISABLED)
save_button.grid(row=row_counter, column=2, padx=50)

file_path = None

window.mainloop()




