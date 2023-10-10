import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import data
from dataset import data_for_ben_or_mal
import os
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import random_split
import cv2 as cv 
import tkinter as tk
import threading


# writer = SummaryWriter('C:/Users/walle/English_path/net_img/Brain_Tumor/logs')  # tensorboard 參考檔案位置


def train_ben_or_mal(
                    model,
                    device="cpu",
                    epochs=5,
                    batch_size=64,
                    lr=0.001,
                    full_img="",
                    img_height=224,
                    img_width=224,
                    show_dir="",
                    save_checkpoint=True,
                    lit_n=0,
                    save="",
                    load_file='',
                    logs=None,
                    event_stop=None,
                    val_precent=0.1,
                    ):
    try:
        if load_file != '':
            Load(model, torch.load(load_file))
            print(f"load file from {load_file}")

        

        print(f"====  Training   ====")
        train_transform = A.Compose(
            [
                A.Resize(img_height, img_width),
                ToTensorV2()
            ],
        )
    #### Create data ####
        full_data = data_for_ben_or_mal(full_img, train_transform)
        if lit_n != 0:
            full_data, _ = random_split(full_data, [lit_n, len(full_data)-lit_n])
        
        tr_s = int(len(full_data)*(1 - val_precent))
        
        te_s = len(full_data) - tr_s
        train_data, test_data = random_split(full_data, [tr_s, te_s])

        train_Loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
        test_Loader = DataLoader(test_data, batch_size, shuffle=False, drop_last=True)

    #### optimizer, loss_function, ... ####

        optimizer = optim.Adam(model.parameters(), lr)
        loss_f = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
        global_step = 0
        decrease = 10

    #### training ####

        for epoch in range(1, epochs+1):

            if event_stop != None:
                if event_stop.is_set() and event_stop != None:
                    print("2stop happend!!!!")
                    if logs != None:
                        logs.configure(state="normal")
                        logs.insert(tk.END, "[*] END\n")
                        logs.configure(state="disabled")
                    event_stop.clear()
                    print(event_stop)
                    break

            print(f"====  epoch {epoch}   ====")
            if logs != None:
                logs.configure(state="normal")
                logs.insert(tk.END, f"====  epoch {epoch}   ====\n")
                logs.configure(state="disabled")
            epoch_loss = 0
            checkpoint = {'state_dict': model.state_dict(), 'optmizer': optimizer.state_dict()}
            optimizer = optim.Adam(model.parameters(), lr)


            for idx, (i_data, target) in enumerate(train_Loader):

                if event_stop != None:
                    if event_stop.is_set():
                        print("stop happend!!!!")
                        break

                i_data = i_data.to(device, torch.float32)
                target = target.to(device, torch.float32)

                prediction = model(i_data)
                loss = loss_f(prediction, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                epoch_loss += loss.item()

                print(f"\t\tBatch {idx+1} done, with loss = {loss}")
                if logs != None:
                    logs.configure(state="normal")
                    logs.insert(tk.END, f"[+] Batch {idx+1} done, with loss = {loss}\n")
                    logs.configure(state="disabled")
                    logs.see(tk.END)
                
            print(f"############ {epoch} epoch loss = {epoch_loss / len(train_Loader)} lr = {lr} ############")
            if logs != None:
                logs.configure(state="normal")
                logs.insert(tk.END, f"[+] {epoch} epoch loss = {epoch_loss / len(train_Loader)}, lr = {lr}\n")
                logs.configure(state="disabled")
                logs.see(tk.END)
            # if epoch == decrease:
            #     if lr > 2e-5:
            #         lr = lr*0.5
            #         decrease *= 2
            #     else:
            #         lr = lr*0.8
            #         decrease *= 3
            if save_checkpoint:
                Save(checkpoint, save)

    #### check accurracy ####

            print("====  check accurracy  ====\n")
            if logs != None:
                logs.configure(state="normal")
                logs.insert(tk.END, "====  check accurracy  ====\n")
                logs.configure(state="disabled")
                logs.see(tk.END)
            benign = torch.tensor([1, 0, 0])
            malignant = torch.tensor([0,1,0])
            normal = torch.tensor([0,0,1])

            with torch.no_grad():
                num_correct = 0
                num_pixel = 0
                
                for idx, (x, y) in enumerate(test_Loader):
                    
                    if event_stop != None:
                        if event_stop.is_set():
                            print("stop happend!!!!")
                            break
                    x = x.to(device, torch.float32)
                    y = y.to(device)
                    preds = torch.sigmoid(model(x))
                    preds = preds.cpu()
                    y = y.cpu()

                    # print(f"pred.shape = {preds.shape}")

                    preds_p = preds / preds.sum(dim=1).unsqueeze(0).transpose(0,1)

                    

                    print(f"y = \n{y}")
                    preds = torch.where(preds > 0.5, 1, 0)
                    num_correct += (preds == y).sum()
                    num_pixel += y.numel()
                    print(f"preds_nurmalize = \n{preds}")


                    print(f"num_correct: {num_correct}, num_pixel: {num_pixel}\n")
                    print(f"* accurracy => {round(float(num_correct/num_pixel)*100, 2)}%\n")
                    if logs != None:
                        logs.configure(state="normal")
                        logs.insert(tk.END, f"[+] accurracy => {round(float(num_correct/num_pixel)*100, 2)}%\n")
                        logs.configure(state="disabled")

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
                                    # print(f"---uncertain---> {epoch}e{idx}b{i}")
    except Exception as e:
        print(e)
        if logs != None:
            logs.configure(state="normal")
            logs.insert(tk.END, "[-] ")
            logs.insert(tk.END, e)
            logs.insert(tk.END, "\n")
            logs.configure(state="disabled")


def train(
        model,
        device="cpu",
        epochs=5,
        batch_size=64,
        lr=0.001,
        full_img="",
        full_mask="",
        img_height=224,
        img_width=224,
        show_dir="",
        lit_n=0,
        load_file='',
        save="",
        logs=None,
        event_stop=None,
        save_checkpoint=True,
        val_precent=0.1
):
    try:
        if load_file != '':
            Load(model, torch.load(load_file))
            print(f"load file from {load_file}")

        print(f"====  Training   ====")
        train_transform = A.Compose(
            [
                A.Resize(img_height, img_width),
                ToTensorV2()
            ],
        )
    #### Create data ####
        full_data = data(full_img, full_mask, train_transform)
        if lit_n != 0:
            full_data, _ = random_split(full_data, [lit_n, len(full_data)-lit_n])
        tr_s = int(len(full_data)*(1-val_precent))
        
        te_s = len(full_data) - tr_s
        train_data, test_data = random_split(full_data, [tr_s, te_s])

        train_Loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
        test_Loader = DataLoader(test_data, batch_size, shuffle=False, drop_last=True)
    #### optimizer, loss_function, ... ####

        optimizer = optim.Adam(model.parameters(), lr)
        loss_f = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
        global_step = 0
        decrease = 10
    #### training ####

        for epoch in range(1, epochs+1):

            if event_stop != None:
                if event_stop.is_set() and event_stop != None:
                    print("2stop happend!!!!")
                    if logs != None:
                        logs.configure(state="normal")
                        logs.insert(tk.END, "[*] END\n")
                        logs.configure(state="disabled")
                    event_stop.clear()
                    print(event_stop)
                    break

            print(f"====  epoch {epoch}   ====")
            if logs != None:
                logs.configure(state="normal")
                logs.insert(tk.END, f"====  epoch {epoch}   ====\n")
                logs.configure(state="disabled")
            epoch_loss = 0
            checkpoint = {'state_dict': model.state_dict(), 'optmizer': optimizer.state_dict()}
            optimizer = optim.Adam(model.parameters(), lr)



            for idx, (i_data, target) in enumerate(train_Loader):

                if event_stop != None:
                    if event_stop.is_set():
                        print("stop happend!!!!")
                        break

                i_data = i_data.to(device, torch.float32)
                target = target.to(device, torch.float32).unsqueeze(1)

                prediction = model(i_data)
                target = torch.where(target > 127, 1., 0.)
                loss = loss_f(prediction, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                epoch_loss += loss.item()

                print(f"\t\tBatch {idx+1} done, with loss = {loss}")
                if logs != None:
                    logs.configure(state="normal")
                    logs.insert(tk.END, f"[+] Batch {idx+1} done, with loss = {loss}\n")
                    logs.configure(state="disabled")
                    logs.see(tk.END)
                # writer.add_scalar('loss', loss, epoch)                  # tensorboard 輸出畫線
                
            
            # print(f"====  {epoch} end  ====")
            print(f"[+] {epoch} epoch loss = {epoch_loss / len(train_Loader)}, lr = {lr}")
            if logs != None:
                logs.configure(state="normal")
                logs.insert(tk.END, f"[+] {epoch} epoch loss = {epoch_loss / len(train_Loader)}, lr = {lr}\n")
                logs.configure(state="disabled")
                logs.see(tk.END)
            
            # if epoch == decrease:
            #     if lr > 2e-5:
            #         lr = lr*0.5
            #         decrease *= 2
            #     else:
            #         lr = lr*0.8
            #         decrease *= 3

            if save_checkpoint:
                Save(checkpoint, save)
            #### check accurracy ####

            print("====  check accurracy  ====\n")
            if logs != None:
                logs.configure(state="normal")
                logs.insert(tk.END, "====  check accurracy  ====\n")
                logs.configure(state="disabled")
                logs.see(tk.END)

            with torch.no_grad():
                for idx, (x, y) in enumerate(test_Loader):
                    x = x.to(device, torch.float32)
                    y = y.to(device).unsqueeze(1)
                    if event_stop != None:
                        if event_stop.is_set():
                            print("stop happend!!!!")
                            break

                    num_correct = 0
                    num_pixel = 0
                    
                    preds = torch.sigmoid(model(x))
                    
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
                    # print("y ",(y == 255).sum(), (y == 0).sum())
                    # print("preds ",( preds == 255).sum(), (preds == 0).sum())
                    preds = np.clip(preds.cpu(), 0, 255)

                    num_correct = (preds == y.cpu()).sum()
                    num_pixel = y.numel()

                    # print(f"num_correct: {num_correct}, num_pixel: {num_pixel}\n")
                    print(f"* accurracy => {(num_correct/num_pixel)*100}%\n")
                    if logs != None:
                        logs.configure(state="normal")
                        logs.insert(tk.END, f"[+] accurracy => {(num_correct/num_pixel)*100}%\n")
                        logs.configure(state="disabled")
                        logs.see(tk.END)
                    
                    
                    ######################################################################
                    #####################        輸出測試圖片       #######################
                    ######################################################################

                    # preds = np.clip(preds + preds_1, 0, 255)

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
                        # for i, tumor in enumerate(x):
                        #     if idx == 0 and i < 5:
                        #         tumor = tumor.permute(1, 2, 0)
                        #         mask = torch.where(preds[i] > 200, 0, 255) 
                        #         mask = mask.permute(1, 2, 0)
                        #         tumor = torch.clip(tumor + mask, 0, 255)
                        #         cv.imwrite(show_dir+'/{}e{}b{}_t.jpg'.format(epoch, idx+1, i+1), tumor.numpy())
                    
                    ######################################################################
            # writer.add_scalar('accurracy %', (num_correct/num_pixel)*100, epoch)
    except Exception as e:
        print("Error!!!")
        if logs != None:
            logs.configure(state="normal")
            logs.insert(tk.END, "[-] ")
            logs.insert(tk.END, e)
            logs.insert(tk.END, "\n")
            logs.configure(state="disabled")

def Totest( model,
            load_file,
            test_dir,
            output_dir,
            img_height = 480,
            img_width = 480,
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

    my_Unet = CNN(1, 3, img_height, img_width) # 只能是灰階圖 通道為1 通道3要改"dataset.py"

    # apt = input("select option\n"
                

    #             "\t1 training\n"
    #             "\t2 testing\n"
    #             "\t3 data classification\n"
    #             "\t4 test...\n"
    #             ">> "
    #             )
    # if apt == "1":
    #     apt = input("select option\n"
    #                 f"\t1 load checkpoint from {load_file}\n"
    #                 "\t2 load checkpoint\n"
    #                 "\t3 dont load checkpoint\n"
    #                 ">> "
    #                 )
    #     if apt == '3':
    #         load = False
    #     else:
    #         load = True
    #         if apt == '2':
    #             load_file = input("Enter the checkpoint file location: ")

    #     # if load:
    #     #     Load(my_Unet, torch.load(load_file))
    #     #     print(f"load file from {load_file}")
        # train(my_Unet, 
        #     device, 
        #     epochs, 
        #     batch_size, 
        #     learning_rate,
        #     img_dir,
        #     mask_dir,
        #     img_height,
        #     img_width,
        #     val_precent=0.1,
        #     save_checkpoint=save,
        #     lit_n=lit_n,
        #     test_dir=test_dir,
        #     save=save_file,
        #     load_file=load_file
        #     )
            
        
    # if apt == '2':
    #     Totest( my_Unet,
    #         load_file,
    #         test_dir,
    #         test_dir,
    #         img_height = img_height,
    #         img_width = img_width,
    #         )
        
    # if apt == '3':
    #     classification(source_dir, original_dir, masks_dir)
    
    # if apt == '4':
    #     window = tk.Tk()
    #     window.title("專題")
    #     window.geometry('380x400')
    #     window.resizable(False, False)
    #     window.mainloop()
    train_ben_or_mal(
                    my_Unet,
                    device,
                    epochs,
                    batch_size,
                    learning_rate,
                    img_dir,
                    img_height,
                    img_width,
                    show_dir=test_dir,
                    save_checkpoint=save,
                    lit_n=lit_n,
                    save=save_file,
                    load_file=load_file
                    )

if __name__ == "__main__":
    main()



















