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
            # Recortar a la mitad y agregar puntos de continuación
            half_length = int(max/2)  # La mitad de 200
            truncated_text = file_path[:half_length] + "..."
        else:
            truncated_text = file_path
        choose_label.config(text="File Path: " + truncated_text)
    else:
        return None

    messagebox.showinfo("Información", "Archivo seleccionado y guardado con éxito: " + file_path)
    return file_path


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

file_path = None  # Variable para almacenar la dirección del archivo

window.mainloop()




