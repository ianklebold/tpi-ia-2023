import tkinter as tk

window = tk.Tk()
window.title("Knn Predictions")
window.minsize(830,1000)

buttonGo = tk.Button(window, text="Go") #Here you specify the parent of the component buttonGo and another configuration
buttonGo.pack() #For able to see the component on window you need to execute pack method





window.mainloop() # Show the window on the screen