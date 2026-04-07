import pyautogui
import random
import string
import tkinter as tk

# (kill switch control)
running = True

# window stuff
root = tk.Tk()
root.title("Status")

label = tk.Label(root, text="Running...", font=("Arial", 14))
label.pack(padx=20, pady=20)

# kill switch function
def stop_program(event=None):
    global running
    running = False
    root.destroy()   

# 🔹 BIND ESC KEY
root.bind("<Escape>", stop_program)

# 🔁 
def run_cycle():
    global running

    if not running:
        return  

    # randomizer
    length = random.randint(5, 10)
    text = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

    # type
    pyautogui.write(text)

    # 🔹delay
    time_var1=random.randint(1000,10000)
    root.after(time_var1, delete_text, text)


def delete_text(text):
    global running

    if not running:
        return

    for _ in range(len(text)):
        pyautogui.press('backspace')

    # 🔹 MOUSE MOVEMENT
    dx = random.randint(-5, 5)
    dy = random.randint(-5, 5)
    
    pyautogui.moveRel(dx, dy, duration=1)
    pyautogui.click()

    time_var3=random.randint(3000,4000)
    # 🔁 SCHEDULE NEXT CYCLE
    root.after(time_var3, run_cycle)


# 🔹 START FIRST CYCLE
time_var4=random.randint(1000,5000)
root.after(time_var4, run_cycle)

# 🔹 START GUI LOOP (ALWAYS LAST)
root.mainloop()###########################################################
###############################################q
######################o