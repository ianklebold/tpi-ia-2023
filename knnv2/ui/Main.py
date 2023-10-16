import tkinter as tkinter
from tkinter import filedialog
from tkinter import messagebox
from knnv2.ui.constants.button.Constants import *
from knnv2.ui.constants.label.Constants import *
from knnv2.ui.constants.window.Constants import *

# Define los campos como variables globales
age_entry = None
education_var = None
sex_var = None
is_smoking_var = None
cig_per_day_entry = None
bmi_entry = None
diabp_entry = None
glucose_entry = None
bpmeds_var = None
stroke_var = None
hypert_var = None
diabetes_var = None
totchol_entry = None
sysbp_entry = None
hr_entry = None
save_button = None
new_window = None

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

def save_data():
    # Recuperar los datos ingresados por el usuario
    age = age_entry.get()
    education = education_var.get()
    sex = sex_var.get()
    is_smoking = is_smoking_var.get()
    if(is_smoking):
        cig_per_day = cig_per_day_entry.get()
    else:
        cig_per_day = 0
    bmi = bmi_entry.get()
    diabp = diabp_entry.get()
    glucose = glucose_entry.get()
    bpmeds = bpmeds_var.get()
    stroke = stroke_var.get()
    hypert = hypert_var.get()
    diabetes = diabetes_var.get()
    totchol = totchol_entry.get()
    sysbp = sysbp_entry.get()
    heart_rate = hr_entry.get()

    # Realizar acciones con los datos recuperados
    # Por ejemplo, puedes imprimirlos en la consola
    print("Age:", age)
    print("Education:", education)
    print("Sex:", sex)
    print("Is Smoking:", is_smoking)
    print("Cig Per Day:", cig_per_day)
    print("BMI:", bmi)
    print("DiaBP:", diabp)
    print("Glucose:", glucose)
    print("BPMeds:", bpmeds)
    print("Prevalent Stroke:", stroke)
    print("Prevalent Hyp:", hypert)
    print("Diabetes:", diabetes)
    print("totChol:", totchol)
    print("SysBp:", sysbp)
    print("HeartRate:", heart_rate)

    new_window.destroy()



def open_new_window():
    global age_entry
    global education_var
    global sex_var
    global is_smoking_var
    global cig_per_day_entry
    global bmi_entry
    global diabp_entry
    global glucose_entry
    global bpmeds_var
    global stroke_var
    global hypert_var
    global diabetes_var
    global totchol_entry
    global sysbp_entry
    global hr_entry
    global save_button
    global new_window

    new_window = tkinter.Toplevel(window)
    new_window.title("Nueva Instancia")
    new_window.minsize(1100, 1000)

    # Logo configuration
    new_label = tkinter.Label(new_window, text="Logo")
    new_label.place(x=390, y=20)
    logo = tkinter.PhotoImage(file='images/logo.png')
    new_label.config(image=logo)
    new_label.image = logo  # Es importante mantener una referencia a la imagen

    # Formulario
    form_frame = tkinter.Frame(new_window)
    form_frame.place(x=350, y=200)  # Ajusta la posición para dejar espacio para el logo

    # Grupo 1: Demographics
    demo_label = tkinter.Label(form_frame, text="Demographic")
    demo_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

    age_label = tkinter.Label(form_frame, text="Age")
    age_label.grid(row=1, column=0)
    age_entry = tkinter.Spinbox(form_frame, from_=0, to=150)
    age_entry.grid(row=1, column=1, pady=(0, 5))

    education_label = tkinter.Label(form_frame, text="Education")
    education_label.grid(row=2, column=0, pady=(0, 5))
    education_var = tkinter.BooleanVar()
    education_check = tkinter.Checkbutton(form_frame, text="Completed", variable=education_var)
    education_check.grid(row=2, column=1, pady=(0, 5))

    sex_label = tkinter.Label(form_frame, text="Sex")
    sex_label.grid(row=3, column=0, pady=(0, 5))
    sex_var = tkinter.StringVar()
    sex_var.set("W")
    sex_radio_w = tkinter.Radiobutton(form_frame, text="W", variable=sex_var, value="W")
    sex_radio_m = tkinter.Radiobutton(form_frame, text="M", variable=sex_var, value="M")
    sex_radio_w.grid(row=3, column=1, pady=(0, 5))
    sex_radio_m.grid(row=3, column=2, pady=(0, 5))

    # Grupo 2: Health Information
    health_label = tkinter.Label(form_frame, text="Health Information")
    health_label.grid(row=4, column=0, columnspan=2, pady=(0, 20))

    is_smoking_label = tkinter.Label(form_frame, text="Is Smoking?")
    is_smoking_label.grid(row=5, column=0, pady=(0, 5))
    is_smoking_var = tkinter.BooleanVar()
    is_smoking_check = tkinter.Checkbutton(form_frame, text="Yes", variable=is_smoking_var)
    is_smoking_check.grid(row=5, column=1, pady=(0, 5))

    cig_per_day_label = tkinter.Label(form_frame, text="Cig Per Day")
    cig_per_day_label.grid(row=6, column=0, pady=(0, 5))
    cig_per_day_entry = tkinter.Spinbox(form_frame, from_=0, to=100)
    cig_per_day_entry.grid(row=6, column=1, pady=(0, 5))
    cig_per_day_entry.config(state=tkinter.NORMAL if is_smoking_var.get() else tkinter.DISABLED)

    def enable_cig_per_day():
        cig_per_day_entry.config(state=tkinter.NORMAL if is_smoking_var.get() else tkinter.DISABLED)

    is_smoking_var.trace("w", lambda *args: enable_cig_per_day())

    bmi_label = tkinter.Label(form_frame, text="BMI")
    bmi_label.grid(row=7, column=0, pady=(0, 5))
    bmi_entry = tkinter.Entry(form_frame)
    bmi_entry.grid(row=7, column=1, pady=(0, 5))

    diabp_label = tkinter.Label(form_frame, text="DiaBP")
    diabp_label.grid(row=8, column=0, pady=(0, 5))
    diabp_entry = tkinter.Entry(form_frame)
    diabp_entry.grid(row=8, column=1, pady=(0, 5))

    # Grupo 3: Lab Results
    lab_label = tkinter.Label(form_frame, text="Lab Results")
    lab_label.grid(row=9, column=0, columnspan=2, pady=(0, 20))

    glucose_label = tkinter.Label(form_frame, text="Glucose")
    glucose_label.grid(row=10, column=0, pady=(0, 5))
    glucose_entry = tkinter.Spinbox(form_frame, from_=0, to=500)
    glucose_entry.grid(row=10, column=1, pady=(0, 5))

    bpmeds_label = tkinter.Label(form_frame, text="BPMeds")
    bpmeds_label.grid(row=11, column=0, pady=(0, 5))
    bpmeds_var = tkinter.BooleanVar()
    bpmeds_check = tkinter.Checkbutton(form_frame, text="Yes", variable=bpmeds_var)
    bpmeds_check.grid(row=11, column=1, pady=(0, 5))

    # Grupo 4: Medical History
    history_label = tkinter.Label(form_frame, text="Medical History")
    history_label.grid(row=12, column=0, columnspan=2, pady=(0, 5))

    stroke_label = tkinter.Label(form_frame, text="Prevalent Stroke")
    stroke_label.grid(row=13, column=0, pady=(0, 5))
    stroke_var = tkinter.BooleanVar()
    stroke_check = tkinter.Checkbutton(form_frame, text="Yes", variable=stroke_var)
    stroke_check.grid(row=13, column=1, pady=(0, 5))

    hypert_label = tkinter.Label(form_frame, text="Prevalent Hyp")
    hypert_label.grid(row=14, column=0, pady=(0, 5))
    hypert_var = tkinter.BooleanVar()
    hypert_check = tkinter.Checkbutton(form_frame, text="Yes", variable=hypert_var)
    hypert_check.grid(row=14, column=1, pady=(0, 5))

    diabetes_label = tkinter.Label(form_frame, text="Diabetes")
    diabetes_label.grid(row=15, column=0, pady=(0, 5))
    diabetes_var = tkinter.BooleanVar()
    diabetes_check = tkinter.Checkbutton(form_frame, text="Yes", variable=diabetes_var)
    diabetes_check.grid(row=15, column=1, pady=(0, 5))

    # Grupo 5: Laboratory Values
    lab_values_label = tkinter.Label(form_frame, text="Laboratory Values")
    lab_values_label.grid(row=16, column=0, columnspan=2, pady=(0, 5))

    totchol_label = tkinter.Label(form_frame, text="totChol")
    totchol_label.grid(row=17, column=0, pady=(0, 5))
    totchol_entry = tkinter.Spinbox(form_frame, from_=0, to=500)
    totchol_entry.grid(row=17, column=1, pady=(0, 5))

    sysbp_label = tkinter.Label(form_frame, text="SysBp")
    sysbp_label.grid(row=18, column=0, pady=(0, 5))
    sysbp_entry = tkinter.Entry(form_frame)
    sysbp_entry.grid(row=18, column=1, pady=(0, 5))

    hr_label = tkinter.Label(form_frame, text="HeartRate")
    hr_label.grid(row=19, column=0, pady=(0, 5))
    hr_entry = tkinter.Spinbox(form_frame, from_=0, to=200)
    hr_entry.grid(row=19, column=1, pady=(0, 5))

    # Botón "Guardar"
    save_button = tkinter.Button(form_frame, text="Guardar", command=save_data)
    save_button.grid(row=21, column=0, columnspan=2, pady=10)  # Agrega espaciado entre el botón y el formulario

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




