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

img_dir_E = tk.Entry(font=('Arial',15))
img_dir_E.insert(0, "* 必填")


def loadimg_dir():
    file_path = filedialog.askdirectory()
    if img_dir_E.get() is not None:
        img_dir_E.delete(0,'end')
    img_dir_E.insert(0,file_path) 

img_dir_button_1 = tk.Button(text="...", height=1, command=loadimg_dir)


#mask_dir
mask_dir = tk.Label(text="mask_dir", font=('Arial',15,'bold'))

mask_dir_E = tk.Entry(font=('Arial',15))
mask_dir_E.insert(0, "* 訓練遮罩 必填")


def loadmask_dir():
    file_path = filedialog.askdirectory()
    if mask_dir_E.get() is not None:
        mask_dir_E.delete(0,'end')
    mask_dir_E.insert(0,file_path) 

img_dir_button_2 = tk.Button(text="...", height=1, command=loadmask_dir)


#show_dir
show_dir = tk.Label(text="show_dir", font=('Arial',15,'bold'))

show_dir_E = tk.Entry(font=('Arial',15))
show_dir_E.insert(0, "* 訓練結果 必填")


def loadshow_dir():
    file_path = filedialog.askdirectory()
    if show_dir_E.get() is not None:
        show_dir_E.delete(0,'end')
    show_dir_E.insert(0,file_path) 

img_dir_button_3 = tk.Button(text="...", height=1, command=loadshow_dir)


#checkpoint
checkpoint = tk.Label(text="checkP", font=('Arial',15,'bold'))

checkpoint_E = tk.Entry(font=('Arial',15))


def loadcheckpoint():
    file_path = filedialog.asksaveasfilename(defaultextension='.pth.tar',
                                                filetypes=[
                                                ("checkpoint",".pth.tar")
                                            ])
    if checkpoint_E.get() is not None:
        checkpoint_E.delete(0,'end')
    checkpoint_E.insert(0,file_path) 

img_dir_button_4 = tk.Button(text="...", height=1, command=loadcheckpoint)


#load
load = tk.Label(text="loadP", font=('Arial',15,'bold'))

load_E = tk.Entry(font=('Arial',15))


def loadload():
    file_path = filedialog.askopenfilename()
    if load_E.get() is not None:
        load_E.delete(0,'end')
    load_E.insert(0,file_path) 

img_dir_button_5 = tk.Button(text="...", height=1, command=loadload)


batch = tk.Label(text="batch = ", font=('Arial',15,'bold'))

batch_E = tk.Entry(width=10, font=('Arial',15))
batch_E.insert(0, "10")


epoch = tk.Label(text="epoch = ", font=('Arial',15,'bold'))

epoch_E = tk.Entry(width=5, font=('Arial',15))
epoch_E.insert(0, "200")


lr = tk.Label(text="lr   = ", font=('Arial',15,'bold'))

lr_E = tk.Entry(width=10, font=('Arial',15))
lr_E.insert(0, "0.0001")


lit = tk.Label(text="lit = ", font=('Arial',15,'bold'))

lit_E = tk.Entry(width=5, font=('Arial',15))
lit_E.insert(0,"0")


img_size = tk.Label(text="imgsize = ", font=('Arial',15,'bold'))

img_size_X = tk.Entry(width=10, font=('Arial',15))
img_size_Y = tk.Entry(width=10, font=('Arial',15))
img_size_X.insert(0,"384")
img_size_Y.insert(0,"384")

X_label = tk.Label(text="X", font=('Arial',15,'bold'))



logs = tk.Text(width=60, height=30, font=('Arial',10), state="disable")



event_stop = threading.Event()
def stopf():
    event_stop.set()
    logs.configure(state="normal")
    logs.insert(tk.END, "[*] stop!!!")
    logs.insert(tk.END, "\n")
    logs.configure(state="disabled")

stop = tk.Button(window, text="STOP", font=('Arial',15,'bold'), background='#FF9595', command=stopf)




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


var1 = tk.BooleanVar()
var2 = tk.BooleanVar()

save_para1 = tk.Checkbutton(text="保存", state='normal', variable=var1)

save_para2 = tk.Checkbutton(text="保存", state='normal', variable=var2)




from Net import CNN, resNet, UNet_plus, UNet
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
    my_model = resNet(1,3,int(img_size_Y.get()))
    determine = train.set_model(
        my_model,
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
    train.log_record(logs, "[+] determine model set")
    t2 = threading.Thread(
        target=train.deter_loop,
        args=(
            determine,
            device,
            int(epoch_E.get()),
            logs,
            event_stop
        )
    )
    t2.start()
    

ben_or_mal_btn = tk.Button(text="DISTINGUISH TRAINING", font=('Arial',15,'bold'), command=_train_ben_or_mal, background='#ccc')



def usemodel():
    global contr
    
    if contr == 1:
        contr = 0
        for w in window.winfo_children():
            if w == use:
                use.config(text="train model")
            else:
                w.grid_forget()
        layout(1)

    else:
        print('ewfwwfwf')
        contr = 1
        for w in window.winfo_children():
            if w == use:
                use.config(text="use model")
            else:
                w.grid_forget()
        layout()
contr = 1
use = tk.Button(text="use model", font=('Arial',10,'bold'), background='#bef', command=usemodel)
# test page
# choose test picture
test_picture = tk.Label(text="test_picture", font=('Arial',15,'bold'))

test_picture_E = tk.Entry(font=('Arial',15))
test_picture_E.insert(0, "* 必填")
def loadtest():
    file_path = filedialog.askopenfilename()
    if test_picture_E.get() is not None:
        test_picture_E.delete(0,'end')
    test_picture_E.insert(0,file_path) 
test_picture_button = tk.Button(text="...", height=1, command=loadtest)
# choose seg model
seg_model = tk.Label(text="seg_model", font=('Arial',15,'bold'))
seg_model_E = tk.Entry(font=('Arial',15))
seg_model_E.insert(0, "* 必填")
def loadseg():
    file_path = filedialog.askopenfilename()
    if seg_model_E.get() is not None:
        seg_model_E.delete(0,'end')
    seg_model_E.insert(0,file_path) 
seg_model_button = tk.Button(text="...", height=1, command=loadseg)
# choose det model
det_model = tk.Label(text="det_model", font=('Arial',15,'bold'))
det_model_E = tk.Entry(font=('Arial',15))
det_model_E.insert(0, "* 必填")
def loaddeter():
    file_path = filedialog.askopenfilename()
    if det_model_E.get() is not None:
        det_model_E.delete(0,'end')
    seg_model_E.insert(0,file_path) 
det_model_button = tk.Button(text="...", height=1, command=loaddeter)

def layout(l=0):
    if l == 0:
        img_dir.grid(column=0, row=1, pady=(3,8), padx=(15,10),columnspan=1)
        img_dir_E.grid(column=1, row=1, pady=8, ipadx=50,columnspan=3)
        img_dir_button_1.grid(column=4, row=1, pady=8, padx=(10,10), columnspan=1)
        mask_dir.grid(column=0, row=2, pady=8, padx=(15,10),columnspan=1)
        mask_dir_E.grid(column=1, row=2, pady=8, ipadx=50,columnspan=3)
        img_dir_button_2.grid(column=4, row=2, pady=8, padx=(10,10), columnspan=1)
        show_dir.grid(column=0, row=3, pady=8, padx=(15,10),columnspan=1)
        show_dir_E.grid(column=1, row=3, pady=8, ipadx=50,columnspan=3)
        checkpoint.grid(column=0, row=4, pady=8, padx=(15,10),columnspan=1)
        checkpoint_E.grid(column=1, row=4, pady=8, ipadx=50,columnspan=3)
        img_dir_button_3.grid(column=4, row=3, pady=8, padx=(10,10), columnspan=1)
        img_dir_button_4.grid(column=4, row=4, pady=8, padx=(10,10), columnspan=1)
        load.grid(column=0, row=5, pady=8, padx=(15,10),columnspan=1)
        load_E.grid(column=1, row=5, pady=8, ipadx=50,columnspan=3)
        img_dir_button_5.grid(column=4, row=5, pady=8, padx=(10,10), columnspan=1)
        batch.grid(column=0, row=6, pady=8, padx=(15,10),columnspan=1)
        batch_E.grid(column=1, row=6, pady=8, sticky=tk.W)
        epoch.grid(column=2, row=6, pady=8, columnspan=1, sticky=tk.W)
        epoch_E.grid(column=3, row=6, pady=8, ipadx=30, columnspan=1, sticky=tk.W)
        lr.grid(column=0, row=7, pady=8, padx=(15,10),columnspan=1)
        lr_E.grid(column=1, row=7, pady=8, sticky=tk.W)
        lit.grid(column=2, row=7, pady=8, columnspan=1, sticky=tk.W)
        lit_E.grid(column=3, row=7, pady=8, ipadx=30, columnspan=1, sticky=tk.W)
        img_size.grid(column=0, row=8, pady=8, padx=(5,10),columnspan=1)
        img_size_X.grid(column=1, row=8, pady=8, sticky=tk.W)
        img_size_Y.grid(column=3, row=8, pady=8, columnspan=1, sticky=tk.W)
        X_label.grid(column=2, row=8, pady=8, columnspan=1, sticky=tk.E+tk.W)
        logs.grid(column=6, row=1, columnspan=3, rowspan=9, padx=(10,10))
        stop.grid(column=7, row=10)
        setting.grid(column=0, columnspan=2, row=0, ipadx=3, padx=(15,0), pady=(5,0), sticky=tk.W)
        save_para1.grid(column=4, row=9, pady=(40,8) ,sticky=tk.W)
        save_para2.grid(column=4, row=10, sticky=tk.W)
        train_btn.grid(column=0, row=9, pady=(40,8), padx=(10,0), ipadx=34, columnspan=5)
        ben_or_mal_btn.grid(column=0, row=10, pady=8, padx=(10,0), ipadx=50, columnspan=5)
        use.grid(column=1, row=0, pady=(5,0), padx=(0,0), ipadx=0, columnspan=1)
    elif l == 1:
        test_picture.grid(column=0, row=1, pady=(3,8), padx=(15,10),columnspan=1)
        test_picture_E.grid(column=1, row=1, pady=8, ipadx=50,columnspan=3)
        test_picture_button.grid(column=4, row=1, pady=8, padx=(10,10), columnspan=1)
        seg_model.grid(column=0, row=2, pady=(3,8), padx=(15,10),columnspan=1)
        seg_model_E.grid(column=1, row=2, pady=8, ipadx=50,columnspan=3)
        seg_model_button.grid(column=4, row=2, pady=8, padx=(10,10), columnspan=1)
        det_model.grid(column=0, row=3, pady=(3,8), padx=(15,10),columnspan=1)
        det_model_E.grid(column=1, row=3, pady=8, ipadx=50,columnspan=3)
        det_model_button.grid(column=4, row=3, pady=8, padx=(10,10), columnspan=1)

layout()
logs.see(tk.END)
window.mainloop()