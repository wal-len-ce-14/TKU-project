import train
import tkinter as tk
from tkinter import filedialog
import threading 
import ctypes


window = tk.Tk()
window.title("專題")
window.geometry('960x600')
window.resizable(False, False)
window.iconbitmap("./icon.ico")

#img_dir
img_dir = tk.Label(text="img_dir", font=('Arial',15,'bold'))
img_dir.grid(column=0, row=1, pady=(3,8), padx=(15,10),columnspan=1)
img_dir_E = tk.Entry(font=('Arial',15))
img_dir_E.insert(0, "* 必填")
img_dir_E.grid(column=1, row=1, pady=8, ipadx=50,columnspan=3)

def loadimg_dir():
    file_path = filedialog.askdirectory()
    if img_dir_E.get() is not None:
        img_dir_E.delete(0,'end')
    img_dir_E.insert(0,file_path) 

img_dir_button = tk.Button(text="...", height=1, command=loadimg_dir)
img_dir_button.grid(column=4, row=1, pady=8, padx=(10,10), columnspan=1)

#mask_dir
mask_dir = tk.Label(text="mask_dir", font=('Arial',15,'bold'))
mask_dir.grid(column=0, row=2, pady=8, padx=(15,10),columnspan=1)
mask_dir_E = tk.Entry(font=('Arial',15))
mask_dir_E.insert(0, "* 訓練遮罩 必填")
mask_dir_E.grid(column=1, row=2, pady=8, ipadx=50,columnspan=3)

def loadmask_dir():
    file_path = filedialog.askdirectory()
    if mask_dir_E.get() is not None:
        mask_dir_E.delete(0,'end')
    mask_dir_E.insert(0,file_path) 

img_dir_button = tk.Button(text="...", height=1, command=loadmask_dir)
img_dir_button.grid(column=4, row=2, pady=8, padx=(10,10), columnspan=1)

#show_dir
show_dir = tk.Label(text="show_dir", font=('Arial',15,'bold'))
show_dir.grid(column=0, row=3, pady=8, padx=(15,10),columnspan=1)
show_dir_E = tk.Entry(font=('Arial',15))
show_dir_E.insert(0, "* 訓練結果 必填")
show_dir_E.grid(column=1, row=3, pady=8, ipadx=50,columnspan=3)

def loadshow_dir():
    file_path = filedialog.askdirectory()
    if show_dir_E.get() is not None:
        show_dir_E.delete(0,'end')
    show_dir_E.insert(0,file_path) 

img_dir_button = tk.Button(text="...", height=1, command=loadshow_dir)
img_dir_button.grid(column=4, row=3, pady=8, padx=(10,10), columnspan=1)

#checkpoint
checkpoint = tk.Label(text="checkP", font=('Arial',15,'bold'))
checkpoint.grid(column=0, row=4, pady=8, padx=(15,10),columnspan=1)
checkpoint_E = tk.Entry(font=('Arial',15))
checkpoint_E.grid(column=1, row=4, pady=8, ipadx=50,columnspan=3)

def loadcheckpoint():
    file_path = filedialog.asksaveasfilename(defaultextension='.pth.tar',
                                                filetypes=[
                                                ("checkpoint",".pth.tar")
                                            ])
    if checkpoint_E.get() is not None:
        checkpoint_E.delete(0,'end')
    checkpoint_E.insert(0,file_path) 

img_dir_button = tk.Button(text="...", height=1, command=loadcheckpoint)
img_dir_button.grid(column=4, row=4, pady=8, padx=(10,10), columnspan=1)

#load
load = tk.Label(text="loadP", font=('Arial',15,'bold'))
load.grid(column=0, row=5, pady=8, padx=(15,10),columnspan=1)
load_E = tk.Entry(font=('Arial',15))
load_E.grid(column=1, row=5, pady=8, ipadx=50,columnspan=3)

def loadload():
    file_path = filedialog.askopenfilename()
    if load_E.get() is not None:
        load_E.delete(0,'end')
    load_E.insert(0,file_path) 

img_dir_button = tk.Button(text="...", height=1, command=loadload)
img_dir_button.grid(column=4, row=5, pady=8, padx=(10,10), columnspan=1)

batch = tk.Label(text="batch = ", font=('Arial',15,'bold'))
batch.grid(column=0, row=6, pady=8, padx=(15,10),columnspan=1)
batch_E = tk.Entry(width=10, font=('Arial',15))
batch_E.insert(0, "10")
batch_E.grid(column=1, row=6, pady=8, sticky=tk.W)

epoch = tk.Label(text="epoch = ", font=('Arial',15,'bold'))
epoch.grid(column=2, row=6, pady=8, columnspan=1, sticky=tk.W)
epoch_E = tk.Entry(width=5, font=('Arial',15))
epoch_E.insert(0, "200")
epoch_E.grid(column=3, row=6, pady=8, ipadx=30, columnspan=1, sticky=tk.W)

lr = tk.Label(text="lr   = ", font=('Arial',15,'bold'))
lr.grid(column=0, row=7, pady=8, padx=(15,10),columnspan=1)
lr_E = tk.Entry(width=10, font=('Arial',15))
lr_E.insert(0, "0.0001")
lr_E.grid(column=1, row=7, pady=8, sticky=tk.W)

lit = tk.Label(text="lit = ", font=('Arial',15,'bold'))
lit.grid(column=2, row=7, pady=8, columnspan=1, sticky=tk.W)
lit_E = tk.Entry(width=5, font=('Arial',15))
lit_E.insert(0,"0")
lit_E.grid(column=3, row=7, pady=8, ipadx=30, columnspan=1, sticky=tk.W)

img_size = tk.Label(text="imgsize = ", font=('Arial',15,'bold'))
img_size.grid(column=0, row=8, pady=8, padx=(5,10),columnspan=1)
img_size_X = tk.Entry(width=10, font=('Arial',15))
img_size_Y = tk.Entry(width=10, font=('Arial',15))
img_size_X.insert(0,"384")
img_size_Y.insert(0,"384")
img_size_X.grid(column=1, row=8, pady=8, sticky=tk.W)
X_label = tk.Label(text="X", font=('Arial',15,'bold'))
X_label.grid(column=2, row=8, pady=8, columnspan=1, sticky=tk.E+tk.W)
img_size_Y.grid(column=3, row=8, pady=8, columnspan=1, sticky=tk.W)

logs = tk.Text(width=60, height=30, font=('Arial',10), state="disable")

logs.grid(column=6, row=1, columnspan=3, rowspan=9, padx=(10,10))

event_stop = threading.Event()
def stopf():
    event_stop.set()
    logs.configure(state="normal")
    logs.insert(tk.END, "[*] stop!!!")
    logs.insert(tk.END, "\n")
    logs.configure(state="disabled")

stop = tk.Button(window, text="STOP", font=('Arial',15,'bold'), background='#FF9595', command=stopf)
stop.grid(column=7, row=10)



import json

# previous settings
def set():
    try:
        jfile = open('./set/setting.json', 'r')
        setting = json.load(jfile)
    
        img_dir_E.delete(0,'end')
        img_dir_E.insert(0, setting['img_dir'])
        mask_dir_E.delete(0,'end')
        mask_dir_E.insert(0, setting["mask_dir"])
        show_dir_E.delete(0,'end')
        show_dir_E.insert(0, setting["show_dir"])
        checkpoint_E.delete(0,'end')
        checkpoint_E.insert(0, setting["checkpoint"])
        load_E.delete(0,'end')
        load_E.insert(0, setting["load"])
        batch_E.delete(0,'end')
        batch_E.insert(0, setting["batch"])
        epoch_E.delete(0,'end')
        epoch_E.insert(0, setting["epoch"])
        lr_E.delete(0,'end')
        lr_E.insert(0, setting["lr"])
        lit_E.delete(0,'end')
        lit_E.insert(0, setting["lit"])
        img_size_X.delete(0,'end')
        img_size_X.insert(0, setting["img_size_X"])
        img_size_Y.delete(0,'end')
        img_size_Y.insert(0, setting["img_size_Y"])

    except Exception as e:
        logs.configure(state="normal")
        logs.insert(tk.END, "[-] ")
        logs.insert(tk.END, e)
        logs.insert(tk.END, "\n")
        logs.configure(state="disabled")
    

    jfile.close()



setting = tk.Button(text="上一次設定", height=1, command=set, background='#bef')
setting.grid(column=0, columnspan=2, row=0, ipadx=3, padx=(15,0), pady=(5,0), sticky=tk.W)

var1 = tk.BooleanVar()
var2 = tk.BooleanVar()

save_para1 = tk.Checkbutton(text="保存", state='normal', variable=var1)
save_para1.grid(column=4, row=9, pady=(40,8) ,sticky=tk.W)
save_para2 = tk.Checkbutton(text="保存", state='normal', variable=var2)
save_para2.grid(column=4, row=10, sticky=tk.W)



from Net import UNet 
from Net import UNet_plus
from Net import CNN
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"


def _train():
    if var1.get() == True:
        try:
            jfile = open('./set/setting.json', 'w')
        except Exception as e:
            logs.configure(state="normal")
            logs.insert(tk.END, "[-] ")
            logs.insert(tk.END, e)
            logs.insert(tk.END, "\n")
            logs.configure(state="disabled")
        
        record = {
            "img_dir": img_dir_E.get(),
            "mask_dir": mask_dir_E.get(),
            "show_dir": show_dir_E.get(),
            "checkpoint": checkpoint_E.get(),
            "load": load_E.get(),
            "batch": int(batch_E.get()),
            "epoch": int(epoch_E.get()),
            "lr": float(lr_E.get()),
            "lit": int(lit_E.get()),
            "img_size_X": int(img_size_X.get()),
            "img_size_Y": int(img_size_Y.get()),
        }
        json.dump(record, jfile,  indent=2)
        jfile.close() 
    my_Unet = UNet_plus(1, 1).to(device)    
    segmentation = train.set_model(
        my_Unet,
        int(batch_E.get()),
        float(lr_E.get()),
        img_dir_E.get(),
        mask_dir_E.get(),
        show_dir_E.get(),
        int(img_size_Y.get()),
        int(img_size_X.get()),
        load_E.get(),
        checkpoint_E.get(),
        int(lit_E.get()),
    )
    train.log_record(logs, "[+] segmentation model set")
    t1 = threading.Thread(
        target=train.train_loop,
        args=(
            segmentation,
            device,
            int(epoch_E.get()),
            logs,
            event_stop
        )
    )
    t1.start()

train_btn = tk.Button(text="SEGMENTATION TRAINING", font=('Arial',15,'bold'), command=_train, background='#ccc')
train_btn.grid(column=0, row=9, pady=(40,8), padx=(10,0), ipadx=34, columnspan=5)

def _train_ben_or_mal():
    
    if var2.get() == True:
        print(var2.get())
        try:
            jfile = open('./set/setting.json', 'w')
        except Exception as e:
            logs.configure(state="normal")
            logs.insert(tk.END, "[-] ")
            logs.insert(tk.END, e)
            logs.insert(tk.END, "\n")
            logs.configure(state="disabled")
        
        record = {
            "img_dir": img_dir_E.get(),
            "mask_dir": mask_dir_E.get(),
            "show_dir": show_dir_E.get(),
            "checkpoint": checkpoint_E.get(),
            "load": load_E.get(),
            "batch": int(batch_E.get()),
            "epoch": int(epoch_E.get()),
            "lr": float(lr_E.get()),
            "lit": int(lit_E.get()),
            "img_size_X": int(img_size_X.get()),
            "img_size_Y": int(img_size_Y.get()),
        }

        json.dump(record, jfile,  indent=2)
        jfile.close()
    my_CNN = CNN(1, 3, int(img_size_Y.get()), int(img_size_X.get())).to(device)

    # train_t2 = threading.Thread(
    #     target=train.train_ben_or_mal, args=(
    #         my_Unet,
    #         device,
    #         int(epoch_E.get()),
    #         int(batch_E.get()),
    #         float(lr_E.get()),
    #         img_dir_E.get(),
    #         int(img_size_Y.get()),
    #         int(img_size_X.get()),
    #         show_dir_E.get(),
    #         True,
    #         int(lit_E.get()),
    #         checkpoint_E.get(),
    #         load_E.get(),
    #         logs,
    #         event_stop
    #     )
    # )
    # train_t2.start()
    # logs.see(tk.END)

    determine = train.set_model(
        my_CNN,
        int(batch_E.get()),
        float(lr_E.get()),
        img_dir_E.get(),
        mask_dir_E.get(),
        show_dir_E.get(),
        int(img_size_Y.get()),
        int(img_size_X.get()),
        load_E.get(),
        checkpoint_E.get(),
        int(lit_E.get()),
    )
    

ben_or_mal_btn = tk.Button(text="DISTINGUISH TRAINING", font=('Arial',15,'bold'), command=_train_ben_or_mal, background='#ccc')
ben_or_mal_btn.grid(column=0, row=10, pady=8, padx=(10,0), ipadx=50, columnspan=5)

logs.see(tk.END)
window.mainloop()