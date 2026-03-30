import tkinter as tk
from tkinter import filedialog
import pandas as pd
import pickle
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# ================= LOAD MODELS =================

with open("dementia_model.pkl", "rb") as f:
    patient_model = pickle.load(f)

mri_model = load_model(
    r"C:\Users\nitis\Desktop\ALZHEIMER'S PROJECT\MRI_dataset\mri_cnn_model.h5"
)

mri_path = ""

# ================= CONDITION FUNCTION =================

def get_condition(memory, thinking, decision):

    avg = (memory + thinking + decision) / 3

    if avg >= 8:
        return "Normal Cognitive Function"
    elif avg >= 6:
        return "Mild Cognitive Impairment"
    elif avg >= 4:
        return "Early Alzheimer"
    elif avg >= 2:
        return "Moderate Alzheimer"
    else:
        return "Severe Alzheimer"


# ================= MAIN WINDOW =================

root = tk.Tk()
root.title("Early Alzheimer Disease Detection System")
root.state("zoomed")
root.config(bg="#E3F2FD")

title = tk.Label(
    root,
    text="Early Alzheimer Disease Detection System (CNN + ML)",
    font=("Helvetica", 28, "bold"),
    bg="#E3F2FD",
    fg="#0D47A1"
)
title.pack(pady=20)

container = tk.Frame(root, bg="#E3F2FD")
container.pack()

# ================= LEFT PANEL =================

left_panel = tk.Frame(container, bg="#BBDEFB",
                      bd=2, relief="solid",
                      width=380, height=420)

left_panel.grid(row=0, column=0, padx=25)
left_panel.pack_propagate(False)

tk.Label(left_panel,
         text="Patient Details",
         font=("Helvetica", 18, "bold"),
         bg="#BBDEFB").pack(pady=15)

form = tk.Frame(left_panel, bg="#BBDEFB")
form.pack(pady=25)

def field(name, row):

    tk.Label(form,
             text=name + " :",
             font=("Times New Roman", 16, "bold"),
             bg="#BBDEFB",
             width=12,
             anchor="w").grid(row=row, column=0, pady=14)

    e = tk.Entry(form,
                 font=("Times New Roman", 15),
                 width=20,
                 bd=2,
                 relief="solid",
                 justify="center")

    e.grid(row=row, column=1, pady=14)

    return e


age_entry = field("Age", 0)
memory_entry = field("Memory", 1)
thinking_entry = field("Thinking", 2)
decision_entry = field("Decision", 3)

# ================= MIDDLE PANEL =================

middle_panel = tk.Frame(container, bg="#E1BEE7",
                        bd=2, relief="solid",
                        width=380, height=420)

middle_panel.grid(row=0, column=1, padx=25)
middle_panel.pack_propagate(False)

tk.Label(middle_panel,
         text="MRI Scan",
         font=("Helvetica", 18, "bold"),
         bg="#E1BEE7").pack(pady=15)

image_label = tk.Label(middle_panel, bg="#E1BEE7")
image_label.pack()

# ================= RIGHT PANEL =================

right_panel = tk.Frame(container, bg="white",
                       bd=2, relief="solid",
                       width=380, height=420)

right_panel.grid(row=0, column=2, padx=25)
right_panel.pack_propagate(False)

tk.Label(right_panel,
         text="Patient Report",
         font=("Helvetica", 18, "bold"),
         bg="white").pack(pady=15)

report_text = tk.Text(right_panel,
                      font=("Consolas", 13),
                      bd=0)

report_text.pack(fill="both", expand=True, padx=10)

# ================= FUNCTIONS =================

def upload_mri():

    global mri_path

    file_path = filedialog.askopenfilename(
        filetypes=[("Images", "*.png *.jpg *.jpeg")]
    )

    if file_path:

        mri_path = file_path

        img = Image.open(file_path)
        img = img.resize((250, 250))

        imgTk = ImageTk.PhotoImage(img)

        image_label.config(image=imgTk)
        image_label.image = imgTk


# ================= Predict Patient Details =================

def predict_details_only():

    report_text.delete(1.0, tk.END)

    try:

        age = float(age_entry.get())
        memory = float(memory_entry.get())
        thinking = float(thinking_entry.get())
        decision = float(decision_entry.get())

        patient_pred = patient_model.predict(
            [[age, memory, thinking, decision]]
        )[0]

        result = "Alzheimer Detected" if patient_pred == 1 else "Normal"

        condition = get_condition(memory, thinking, decision)

        report_text.insert(tk.END,
f"""PATIENT DETAILS REPORT
--------------------------
Disease Status : {result}

Memory Score   : {memory}/10
Thinking Score : {thinking}/10
Decision Score : {decision}/10

Condition      : {condition}
""")

        report_text.config(fg="red" if patient_pred == 1 else "green")

    except:
        report_text.insert(tk.END, "Invalid Patient Input")


# ================= Predict MRI =================

def predict_mri_only():

    report_text.delete(1.0, tk.END)

    if mri_path == "":
        report_text.insert(tk.END, "Upload MRI First")
        return

    try:

        memory = float(memory_entry.get())
        thinking = float(thinking_entry.get())
        decision = float(decision_entry.get())

        img = Image.open(mri_path)
        img = img.resize((128,128))
        img = img.convert("L")

        img_array = np.array(img)/255.0
        img_array = img_array.reshape(1,128,128,1)

        mri_pred = mri_model.predict(img_array)[0][0]
        mri_pred = 1 if mri_pred > 0.5 else 0

        result = "Alzheimer Detected" if mri_pred == 1 else "Normal"

        condition = get_condition(memory, thinking, decision)

        report_text.insert(tk.END,
f"""MRI REPORT
--------------------------
Disease Status : {result}

Memory Score   : {memory}/10
Thinking Score : {thinking}/10
Decision Score : {decision}/10

Condition      : {condition}
""")

        report_text.config(fg="red" if mri_pred == 1 else "green")

    except:
        report_text.insert(tk.END, "Enter valid scores")


# ================= Combined Result =================

def predict_combined():

    report_text.delete(1.0, tk.END)

    try:

        age = float(age_entry.get())
        memory = float(memory_entry.get())
        thinking = float(thinking_entry.get())
        decision = float(decision_entry.get())

        patient_pred = patient_model.predict(
            [[age, memory, thinking, decision]]
        )[0]

    except:
        report_text.insert(tk.END, "Invalid Patient Input")
        return

    if mri_path == "":
        report_text.insert(tk.END, "Upload MRI First")
        return

    img = Image.open(mri_path)
    img = img.resize((128,128))
    img = img.convert("L")

    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1,128,128,1)

    mri_pred = mri_model.predict(img_array)[0][0]
    mri_pred = 1 if mri_pred > 0.5 else 0

    final_pred = 1 if (patient_pred == 1 or mri_pred == 1) else 0

    result = "Alzheimer Detected" if final_pred == 1 else "Normal"

    condition = get_condition(memory, thinking, decision)

    report_text.insert(tk.END,
f"""FINAL COMBINED REPORT
--------------------------
Patient Model Result : {"Alzheimer Detected" if patient_pred==1 else "Normal"}
MRI Model Result     : {"Alzheimer Detected" if mri_pred==1 else "Normal"}

FINAL RESULT         : {result}

Memory Score   : {memory}/10
Thinking Score : {thinking}/10
Decision Score : {decision}/10

Condition      : {condition}
""")

    report_text.config(fg="red" if final_pred == 1 else "green")


# ================= Upload Dataset =================

def upload_dataset():

    report_text.delete(1.0, tk.END)

    file = filedialog.askopenfilename(filetypes=[("CSV","*.csv")])

    if file:

        df = pd.read_csv(file)

        preds = patient_model.predict(
            df[['Age','Memory','Thinking','Decision']]
        )

        total = len(preds)
        alz = sum(preds)
        normal = total - alz

        report_text.insert(tk.END,
f"""DATASET REPORT
--------------------------
Total Patients : {total}
Alzheimer      : {alz}
Normal         : {normal}
""")


# ================= BUTTONS =================

buttons = tk.Frame(root, bg="#E3F2FD")
buttons.pack(pady=25)

def btn(text,color,cmd,col):

    tk.Button(
        buttons,
        text=text,
        bg=color,
        fg="white",
        font=("Arial",14,"bold"),
        width=18,
        command=cmd
    ).grid(row=0,column=col,padx=15)


btn("Upload MRI","#6A1B9A",upload_mri,0)
btn("Predict Details","#2E7D32",predict_details_only,1)
btn("Predict MRI","#1565C0",predict_mri_only,2)
btn("Combined Result","#EF6C00",predict_combined,3)
btn("Upload Dataset","#00838F",upload_dataset,4)
btn("Exit","#C62828",root.destroy,5)

root.mainloop()