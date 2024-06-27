import tkinter as tk
from tkinter import ttk

# Placeholder functions for the actions
def run_drl_optimization():
    print("Running DRL Optimization...")

def stop_training():
    print("Training stopped.")

# Main application window
root = tk.Tk()
root.title("Neuraflow")
root.geometry("800x600")
root.configure(bg='#111518')

# Header
header_frame = tk.Frame(root, bg='#283139', pady=10, padx=20)
header_frame.pack(fill=tk.X)

app_title = tk.Label(header_frame, text="Neuraflow", fg="white", bg="#283139", font=("Manrope", 16, "bold"))
app_title.pack(side=tk.LEFT)

menu_frame = tk.Frame(header_frame, bg='#283139')
menu_frame.pack(side=tk.RIGHT)

for item in ["Dashboard", "Models", "Datasets", "Pipelines"]:
    menu_button = tk.Button(menu_frame, text=item, fg="white", bg="#283139", font=("Manrope", 10, "bold"))
    menu_button.pack(side=tk.LEFT, padx=10)

help_button = tk.Button(menu_frame, text="Help", fg="white", bg="#283139", font=("Manrope", 10, "bold"), padx=10)
help_button.pack(side=tk.LEFT, padx=10)

profile_image = tk.Label(menu_frame, bg='#283139', width=10, height=10)
profile_image.pack(side=tk.LEFT, padx=10)

# Training status
status_frame = tk.Frame(root, bg='#111518', pady=20, padx=20)
status_frame.pack(fill=tk.X)

status_label = tk.Label(status_frame, text="Training Run: 2023-09-13-05-00-00", fg="white", bg="#111518", font=("Manrope", 24, "bold"))
status_label.pack()

epoch_label = tk.Label(status_frame, text="Epoch 1/10", fg="white", bg="#111518", font=("Manrope", 14))
epoch_label.pack()

progress = ttk.Progressbar(status_frame, orient="horizontal", length=400, mode="determinate")
progress["value"] = 30
progress.pack(pady=10)

batch_label = tk.Label(status_frame, text="Batch 300/1000", fg="#9cacba", bg="#111518", font=("Manrope", 10))
batch_label.pack()

# Hyperparameter adjustments
param_frame = tk.Frame(root, bg='#111518', pady=20, padx=20)
param_frame.pack(fill=tk.X)

def create_param_row(parent, text, value):
    row_frame = tk.Frame(parent, bg='#111518')
    row_frame.pack(fill=tk.X, pady=5)
    
    label = tk.Label(row_frame, text=text, fg="white", bg="#111518", font=("Manrope", 14))
    label.pack(side=tk.LEFT, padx=10)
    
    scale = ttk.Scale(row_frame, from_=0, to=100, orient="horizontal")
    scale.set(value)
    scale.pack(fill=tk.X, padx=10, expand=True)
    
    value_label = tk.Label(row_frame, text=str(value), fg="white", bg="#111518", font=("Manrope", 10))
    value_label.pack(side=tk.RIGHT, padx=10)
    
create_param_row(param_frame, "Batch Size", 32)
create_param_row(param_frame, "Learning Rate", 0.001)
create_param_row(param_frame, "Epochs", 10)

# Buttons
button_frame = tk.Frame(root, bg='#111518', pady=20)
button_frame.pack(fill=tk.X)

cancel_button = tk.Button(button_frame, text="Cancel", fg="white", bg="#283139", font=("Manrope", 10, "bold"), command=stop_training)
cancel_button.pack(side=tk.RIGHT, padx=10)

save_button = tk.Button(button_frame, text="Save", fg="white", bg="#2094f3", font=("Manrope", 10, "bold"), command=run_drl_optimization)
save_button.pack(side=tk.RIGHT, padx=10)

info_label = tk.Label(root, text="You can edit the hyperparameters while training is paused or stopped.", fg="#9cacba", bg="#111518", font=("Manrope", 10), pady=10)
info_label.pack()

root.mainloop()
