# OS packages:
import os
import sys
import wget
import tarfile
import time
# PyTorch packages
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# Numpy package
import numpy as np
# Plotting
import matplotlib.pyplot as plt
# tkinter UI library
import tkinter as tk
from tkinter import ttk, filedialog
from ctypes import windll

import env
from console import Console

DIR_NAME = os.path.dirname(__file__)


class WindowFrame:
    def __init__(self, notebook, f_text='Notebook frame', f_width=400, f_height=280):
        self.notebook = notebook
        self.frame = ttk.Frame(notebook, width=f_width, height=f_height)
        self.frame.pack(fill='both', expand=True)
        self.notebook.add(self.frame, text=f_text)


def open_file_dialog(target):
    filename = filedialog.askopenfilename(initialdir=DIR_NAME,
                                          title="Select a file")
    target = filename


def window_frame_dataset_prep(notebook):
    frame = WindowFrame(notebook, f_text="Dataset preparation")

    video_file_name = tk.StringVar()

    # video
    video_file_label = ttk.Label(frame.frame, textvariable=video_file_name)
    video_file_btn_explore = ttk.Button(frame.frame, text="Browse files", command=lambda: open_file_dialog(video_file_name))

    video_file_label.grid(column=1, row=1)
    video_file_btn_explore.grid(column=1, row=2)


def before_start():
    return


def main():
    # run logic before tinker window main loop starts
    before_start()

    # run tinker UI
    root = tk.Tk()
    root.title('Driver fatigue detection system')

    # get the screen dimension
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # find the center point
    center_x = int(screen_width / 2 - env.WINDOW_INIT_WIDTH / 2)
    center_y = int(screen_height / 2 - env.WINDOW_INIT_HEIGHT / 2)

    # set the position of the window to the center of the screen
    root.geometry(f'{env.WINDOW_INIT_WIDTH}x{env.WINDOW_INIT_HEIGHT}+{center_x}+{center_y}')

    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, expand=True)

    # create frames
    window_frame_dataset_prep(notebook)

    # run main loop
    root.mainloop()


if __name__ == '__main__':
    main()
