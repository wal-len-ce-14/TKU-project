import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import data
from dataset import data_for_ben_or_mal
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import random_split
import cv2 as cv 
import tkinter as tk

def log_record(logs, str):
    if logs != '':
        logs.configure(state="normal")
        logs.insert(tk.END, str+'\n')
        logs.configure(state="disabled")
        logs.see(tk.END)

def set_model( 
    model,
    batch_size=10,
    lr=0.0001,
    full_img="",
    full_mask="",
    show_dir='',
    img_height=224,
    img_width=224,
    load='',
    save='',
    lit_n=0
):
    #### load model parameter ####
    if load != '':
        Load(model, torch.load(load))
        print(f"load file from {load}")
    train_transform = A.Compose(
            [
                A.Resize(img_height, img_width),
                ToTensorV2()
            ],
        )
    ###################
    #### Create data ####
    # --mask
    if full_mask != '':
        full_data = data(full_img, full_mask, train_transform)
        if lit_n != 0:
            full_data, _ = random_split(full_data, [lit_n, len(full_data)-lit_n])
        tr_s = int(len(full_data)*0.9)
        te_s = len(full_data) - tr_s
        train_data, test_data = random_split(full_data, [tr_s, te_s])

        train_Loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
        test_Loader = DataLoader(test_data, batch_size, shuffle=False, drop_last=True)
    
    # --deter
    full_data = data_for_ben_or_mal(full_img, train_transform)
    if lit_n != 0:
        full_data, _ = random_split(full_data, [lit_n, len(full_data)-lit_n])
    
    tr_s = int(len(full_data)*0.9)
    te_s = len(full_data) - tr_s
    train_data, test_data = random_split(full_data, [tr_s, te_s])

    train_deter_Loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    test_deter_Loader = DataLoader(test_data, batch_size, shuffle=False, drop_last=True)
    #### optimizer, loss_function, ... ####
    optimizer = optim.Adam(model.parameters(), lr)
    loss_f = nn.CrossEntropyLoss() 
    
    set = {
        "train_Loader": train_Loader,
        "test_Loader": test_Loader,
        "train_deter": train_deter_Loader,
        "test_deter": test_deter_Loader,
        "optimizer": optimizer,
        "loss_f": loss_f,
        "model": model,
        "show_dir": show_dir,
        "save": save,
        "base_seg": 80,
        "base_det": 60
    }
    return set

def mask_test(
    model,
    device="cpu",
    test_Loader='',
    show_dir='',
    epoch=0
):
    with torch.no_grad():
        all_correct = 0
        all_pixel = 0
        for idx, (x, y) in enumerate(test_Loader):
            x = x.to(device, torch.float32)
            y = y.to(device).unsqueeze(1)
            batch_correct = 0
            preds = torch.sigmoid(model(x))
            ##### 漸層 #####
            p1 = (preds <= 0.5)
            p2 = (preds > 0.4)
            preds_1 = torch.where(p1 & p2, 150, 0)
            p1 = (preds <= 0.4)
            p2 = (preds > 0.35)
            preds_2 = torch.where(p1 & p2, 100, 0)
            p1 = (preds <= 0.35)
            p2 = (preds > 0.3)
            preds_3 = torch.where(p1 & p2, 50, 0)
            preds = torch.where(preds > 0.5, 255, 0)
            preds = preds + preds_1 + preds_2 + preds_3
            preds = np.clip(preds.cpu(), 0, 255)
            #################
            batch_correct = (preds == y.cpu()).sum()
            num_pixel = y.numel()
            all_correct += batch_correct
            all_pixel += num_pixel
            print(f"* accurracy => {(batch_correct/num_pixel)*100}%\n")
            if show_dir != '':
                for i, img in enumerate(preds):                               
                    if idx == 0 and i < 5:
                        img = img.permute(1, 2, 0)
                        cv.imwrite(show_dir+'/{}e{}b{}.jpg'.format(epoch, idx+1, i+1), img.numpy())      # 預測遮罩
                for i, imgx in enumerate(x.cpu()):
                    if idx == 0 and i < 5 and epoch == 1:
                        imgx = imgx.permute(1, 2, 0)
                        cv.imwrite(show_dir+'/_original{}.jpg'.format(i+1), imgx.numpy())            # 原圖
                for i, imgy in enumerate(y.cpu()):
                    if idx == 0 and i < 5 and epoch == 1:
                        imgy = imgy.permute(1, 2, 0)
                        cv.imwrite(show_dir+'/__target{}.jpg'.format(i+1), imgy.numpy())           # 原遮罩  
    return round(float(all_correct/all_pixel)*100, 5)

def deter_test(
    model,
    device="cpu",
    test_Loader='',
    show_dir='',
    epoch=0
):
    with torch.no_grad():
        all_correct = 0
        all_pixel = 0
        for idx, (x, y) in enumerate(test_Loader):
            x = x.to(device, torch.float32)
            y = y.to(device)  
            preds = torch.sigmoid(model(x)) 
            preds = preds.cpu()
            y = y.cpu()
            preds_p = preds / preds.sum(dim=1).unsqueeze(0).transpose(0,1)
            preds = torch.where(preds > 0.5, 1, 0)
            all_correct += (preds * y).sum()
            all_pixel += y.numel()
            print(f"tt = {(preds * y).sum()}")
            print(f"preds_nurmalize = \n{preds}")
            print(f"true_nurmalize = \n{y}")
            print(f"num_correct: {all_correct}, num_pixel: {all_pixel}\n")
            print(f"* accurracy => {round(float(all_correct/all_pixel)*100, 2)}%\n")
            

            benign = torch.tensor([1, 0, 0])
            malignant = torch.tensor([0,1,0])
            normal = torch.tensor([0,0,1])
            if show_dir != '':
                for i, imgx in enumerate(x.cpu()):
                    if idx == 0 and i < 5 and epoch == 1:
                        imgx = imgx.permute(1, 2, 0) 
                        if y[i].equal(benign):
                            cv.imwrite(show_dir+'/_{}benign.jpg'.format(i+1), imgx.numpy())            # 原圖
                        elif y[i].equal(malignant):
                            cv.imwrite(show_dir+'/_{}malignant.jpg'.format(i+1), imgx.numpy())            # 原圖
                        elif y[i].equal(normal):
                            cv.imwrite(show_dir+'/_{}normal.jpg'.format(i+1), imgx.numpy())            # 原圖
                for i, imgx in enumerate(x.cpu()):
                    if idx == 0 and i < 5:
                        imgx = imgx.permute(1, 2, 0)

                        if preds[i].equal(benign):
                            cv.imwrite(show_dir+'/{}e{}b{}_benign_{}%.jpg'.format(epoch, idx+1, i+1, round(float(preds_p[i][0])*100, 2)), imgx.numpy())      # 預測
                            # print(f"---benign---> {epoch}e{idx}b{i}")
                        elif preds[i].equal(malignant):
                            cv.imwrite(show_dir+'/{}e{}b{}_malignant_{}%.jpg'.format(epoch, idx+1, i+1, round(float(preds_p[i][1])*100, 2)), imgx.numpy())      # 預測
                            # print(f"---malignant---> {epoch}e{idx}b{i}")
                        elif preds[i].equal(normal):
                            cv.imwrite(show_dir+'/{}e{}b{}_normal_{}%.jpg'.format(epoch, idx+1, i+1, round(float(preds_p[i][2])*100, 2)), imgx.numpy())      # 預測
                            # print(f"---normal---> {epoch}e{idx}b{i}")
                        else:
                            cv.imwrite(show_dir+'/{}e{}b{}_uncertain.jpg'.format(epoch, idx+1, i+1), imgx.numpy())
    return round(float(all_correct/all_pixel)*100, 2)

def deter_loop(
    set,
    device="cpu",
    epochs=0,
    logs='',
    stop=None    
):
    try:
        print("in deter")
        for epoch in range(1, epochs+1):
            if stop.is_set() and stop != None:
                log_record(logs, f"[*] epoch stop")
                stop.clear()
                break   
            epoch_loss = 0
            for idx, (data, target) in enumerate(set["train_deter"]):
                if stop.is_set() and stop != None:
                    log_record(logs, f"[*] optimize stop")
                    break
                data = data.to(device, torch.float32)
                target = target.to(device, torch.float32)
                prediction = set["model"](data)
                loss = set["loss_f"](prediction, target)
                set["optimizer"].zero_grad()
                loss.backward()
                set["optimizer"].step()
                epoch_loss += loss.item()
                print(f"\t\tBatch {idx+1} done, with loss = {loss}")
                log_record(logs, f"[+] Batch {idx+1} done, with loss = {loss}")
            
            accuracy = round(deter_test(
                set["model"],
                device,
                set["test_deter"],
                set["show_dir"],
                epoch
            ),2)
            
            log_record(logs, f"[+] epoch {epoch+1} done, with loss = {epoch_loss/len(set['train_deter'])}")
            log_record(logs, f"[+] accuracy => {accuracy}%")
            if accuracy > set["base_det"]:
                set["base_det"] = accuracy
                torch.save(set["model"], f'./model/det/e{epoch}acc{accuracy}%.pth')
                log_record(logs, f"[+] save model in ./model/det/e{epoch}acc{accuracy}%.pth")
        log_record(logs, f"[+] END")
    except Exception as e:
         print("Error!!!")  
         log_record(logs, f"[-] {e}")
         log_record("[*] Error")

def train_loop(
    set,
    device="cpu",
    epochs=0,
    logs='',
    stop=None
):
    try:
        for epoch in range(1, epochs+1):
            if stop.is_set() and stop != None:
                log_record(logs, f"[*] epoch stop")
                stop.clear()
                break
            epoch_loss = 0
            for idx, (data, target) in enumerate(set["train_Loader"]):
                if stop.is_set() and stop != None:
                    log_record(logs, f"[*] optimize stop")
                    break
                data = data.to(device, torch.float32)
                target = target.to(device, torch.float32).unsqueeze(1)
                prediction = torch.sigmoid(set["model"](data))
                target = torch.where(target > 127, 1., 0.)
                
                # print(set["loss_f"])
                
                set["optimizer"].zero_grad()
                loss = nn.BCELoss()(prediction,target)
                loss.backward()
                set["optimizer"].step()
                epoch_loss += loss.item()
                print(f"\t\tBatch {idx+1} done, with loss = {loss}")
                log_record(logs, f"[+] Batch {idx+1} done, with loss = {loss}")
            
            accuracy = round(mask_test(
                set["model"],
                device,
                set["test_Loader"],
                set["show_dir"],
                epoch
            ),2)
            log_record(logs, f"[+] epoch {epoch+1} done, with loss = {epoch_loss/len(set['train_Loader'])}")
            log_record(logs, f"[+] accuracy => {accuracy}%")
            if accuracy > set["base_seg"]:
                set["base_seg"] = accuracy
                torch.save(set["model"], f'./model/seg/e{epoch}acc{accuracy}%.pth')
                log_record(logs, f"[+] save model in ./model/seg/e{epoch}acc{accuracy}%.pth")

        log_record(logs, f"[+] END")
    except Exception as e:
         print("Error!!!")  
         log_record(logs, f"[-] {e}")

def testmodel(
  model_seg,
  model_det,
  test_img,
  img_size=224,      
):
    seg = torch.load(model_seg)
    det = torch.load(model_det)
    to_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            ToTensorV2()
        ],
    )
    image = to_transform(image=test_img)["image"]
    preds_seg = torch.sigmoid(seg(image))
    preds_seg = torch.where(preds_seg > 0.5, 255, 0)
    preds_seg = preds_seg.squeeze(0).permute(1, 2, 0)

    preds_det_o = torch.sigmoid(det(image))
    preds_det = torch.where(preds_det_o > 0.5, 1, 0)
    benign = torch.tensor([1, 0, 0])
    malignant = torch.tensor([0,1,0])
    normal = torch.tensor([0,0,1])
    if preds_det.equal(benign):
        det = f"benign_{preds_det_o[0]*100}%"                  
    elif preds_det.equal(malignant):
        det = f"malignant_{preds_det_o[1]*100}%"
    elif preds_det.equal(normal):
        det = f"normal_{preds_det_o[2]*100}%"
    else:
        det = "not sure"  

    from PIL import Image, ImageTk
    preds_seg_PLT = cv.cvtColor(preds_seg, cv.COLOR_BGR2RGB)
    img_PLT = Image.fromarray(preds_seg_PLT)
    img = ImageTk.PhotoImage(img_PLT)
    return {
        "seg": preds_seg,   # picture: cv
        "seg_PLT": img,   # picture: PLT
        "det": det          # deter: string
    }


def Totest( 
    model,
    load_file,
    test_dir,
    output_dir,
    img_height = 224,
    img_width = 224
    ):
    Load(model, torch.load(load_file))
    test_imgs = os.listdir(test_dir)
    with torch.no_grad():
        for i, index in enumerate(test_imgs):
            img_path = os.path.join(test_dir, index)
            test_img = np.array(cv.resize(cv.imread(img_path, cv.IMREAD_GRAYSCALE),(img_height, img_width)))
            train_transform = A.Compose(
                [
                    A.Resize(img_height, img_width),
                    ToTensorV2()
                ],
            )
            aa = train_transform(image=test_img)
            test_img = aa['image'].to(torch.float32).unsqueeze(0)
            preds = torch.sigmoid(model(test_img))
            preds = torch.where(preds > 0.5, 255, 0)
            preds = preds.squeeze(0).permute(1, 2, 0)
            cv.imwrite(output_dir + '/0{}_z.jpg'.format(i+1), preds.numpy())

def Save(state, filename):
    print(f"=> Saveing checkpoint to {filename}")
    torch.save(state, filename)

def Load(model, checkpoint):
    # print("=> Loading checkpoint from ")
    model.load_state_dict(checkpoint['state_dict'])

def main():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 1000
    batch_size = 20
    learning_rate = 4e-4
    img_dir = "C:/Users/walle/English_path/net_img/Dataset_BUSI_with_GT/images"
    mask_dir = "C:/Users/walle/English_path/net_img/Dataset_BUSI_with_GT/masks"
    test_dir = "C:/Users/walle/English_path/net_img/Dataset_BUSI_with_GT/test"
    img_height = 384
    img_width = 384
    lit_n = 0       # 一次測幾個資料 設為0表示全部都測
    load = "C:/Users/TKU-STAFF/Desktop/wall/checkpoint/test.pth.tar "    # 是否要載入既有的模組參數
    save = True     # 是否要保存既有的模組參數
    load_file = "C:/Users/walle/English_path/net_img/Dataset_BUSI_with_GT/checkpoint/checkpoint_determine_01epoch.pth.tar"  # 載入模組參數位置
    save_file = "C:/Users/walle/English_path/net_img/Dataset_BUSI_with_GT/checkpoint/checkpoint_determine_01epoch.pth.tar"  # 保存模組參數位置
    
    source_dir = "C:/Users/walle/English_path/net_img/Dataset_BUSI_with_GT/normal"
    original_dir = "C:/Users/walle/English_path/net_img/Dataset_BUSI_with_GT/images"
    masks_dir = "C:/Users/walle/English_path/net_img/Dataset_BUSI_with_GT/masks"

    from Net import UNet 
    from Net import UNet_plus
    from Net import CNN

    my_CNN = CNN(1, 3, img_height, img_width) # 只能是灰階圖 通道為1 通道3要改"dataset.py"



if __name__ == "__main__":
    main()



