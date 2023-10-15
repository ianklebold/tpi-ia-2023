import tkinter as tkinter

from knnv2.ui.constants.button.Constants import *
from knnv2.ui.constants.label.Constants import *
from knnv2.ui.constants.window.Constants import *


window = tkinter.Tk()
window.geometry(str(X_SIZE)+"x"+str(Y_SIZE))

#Logo configuration
label = tkinter.Label(window, text="Logo")
logo = tkinter.PhotoImage(file='images/logo.png')
label.place(x=390, y=20)
label.config(image=logo)


tkinter.Label(window, text="Sample Label2")
tkinter.Label(window, text="Sample Label3")
tkinter.Label(window, text="Sample Label4")

#def buttonGoPress():
#    print("Go!")


#buttonGo = tkinter.Button(window, text=GO_TEMPLATE)
#buttonGo.config(command=buttonGoPress)
#buttonGo.config(state='disabled')
#buttonGo.pack()


window.mainloop()