import tkinter as tkinter

from knnv2.ui.constants.button.Constants import *
from knnv2.ui.constants.label.Constants import *
from knnv2.ui.constants.window.Constants import *


window = tkinter.Tk()
window.title(WINDOW_TITLE_TEMPLATE)
window.minsize(X_SIZE, Y_SIZE)


label = tkinter.Label(window, text=UPLOAD_DATA_SET_TEMPLATE)
logo = tkinter.PhotoImage(file='images/logo.png')
label.config(image=logo)
label.pack()

buttonGo = tkinter.Button(window, text=GO_TEMPLATE)
buttonGo.pack()

window.mainloop()