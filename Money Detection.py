import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np
import cv2

money = 0
_capture, _frame = False, None


def Bank_Note_Classification(rgb):
    blur = cv2.GaussianBlur(rgb, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 0, 130])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def Combine(rgb, mask, coinmode=0):
    if coinmode != 0:    
        mask = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(rgb, rgb, mask=mask)
    return result

def bank_20(rgb, state=0):  
    blur = cv2.GaussianBlur(rgb, (7, 7), 0)
    hsv = cv2.cvtColor(blur,cv2.COLOR_RGB2HSV)
    lower = np.array([30, 0, 130])
    upper = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    if state != 0:
        result = Combine(rgb, mask)
        result, count = bank_contour(result)
        global money 
        money += (count * 20)
        return result
    else:
        return mask
    
def bank_50(rgb, state=0): 
    blur = cv2.GaussianBlur(rgb, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    lower = np.array([70, 0, 130])
    upper = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    if state != 0:
        result = Combine(rgb, mask)
        result, count = bank_contour(result)
        global money 
        money += (count * 50)
        return result
    else:
        return mask

def bank_100(rgb, state=0): 
    blur = cv2.GaussianBlur(rgb, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    lower = np.array([2, 30, 170])
    upper = np.array([8, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    if state != 0:
        result = Combine(rgb,mask)
        result, count = bank_contour(result)
        global money 
        money += (count * 100)
        return result
    else:
        return mask

def bank_500(rgb, state=0): 
    blur = cv2.GaussianBlur(rgb, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    lower = np.array([160, 20, 165])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    if state != 0:
        result = Combine(rgb, mask)
        result, count = bank_contour(result)
        global money 
        money += (count * 500)
        return result
    else:
        return mask

def bank_1000(rgb, state=0): 
    blur = cv2.GaussianBlur(rgb, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    lower = np.array([9, 20, 130])
    upper = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    if state != 0:
        result = Combine(rgb, mask)
        result, count = bank_contour(result)
        global money 
        money += (count * 1000)
        return result
    else:
        return mask

def coins(rgb,state=0):
    blur = cv2.GaussianBlur(rgb, (7, 7), 0)
    hsv = cv2.cvtColor(blur,cv2.COLOR_RGB2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    if state != 0:
        result = Combine(rgb, mask)
        result = coin_count(result)
        return result
    else:
        return mask

def bank_contour(img):
    number_of_bank = 0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 0, 200, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if (area > 50000):
            cv2.drawContours(img, contours, i, (0, 255, 0), 2)
            number_of_bank += 1

    return img, number_of_bank

def coin_count(img):
    global money
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 30, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    area_list = []
    
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        area_list.append(area)
  
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if (area > 7000) and (area < 8000):
            money += 10
    
        elif (area > 6000) and (area < 7000):
            money += 5
    
        elif (area > 5000) and (area < 6000):
            money += 2
        
        elif (area > 4000) and (area < 5000):
            money += 1
    
        elif (area > 3500) and (area < 4000):
            money += 0.5
            
    return img

def interface_template(rgb):
    cv2.putText(rgb, f"Total Money {money}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    return rgb

def detect(img):
    global img_bank, img_coin, img_opt, bank, coin, output, money
    money = 0
    
    mask = Bank_Note_Classification(img)
        
    result_bank = Combine(img, mask)
    result_20 = bank_20(result_bank,state=1)
    result_50 = bank_50(result_bank,state=1)
    result_100 = bank_100(result_bank,state=1)
    result_500 = bank_500(result_bank,state=1)
    result_1000 = bank_1000(result_bank,state=1)    
    result_bank = cv2.resize(result_bank, (500, 500))
    
    img_bank = ImageTk.PhotoImage(Image.fromarray(result_bank))
    bank.create_image(0, 0, image=img_bank, anchor=tk.NW)
    bank.update()
    
    result = Combine(img, mask, coinmode=1)
    result_coin = coins(result,state=1)
    result = cv2.resize(result, (500, 500))
    
    img_coin = ImageTk.PhotoImage(Image.fromarray(result))
    coin.create_image(0, 0, image=img_coin, anchor=tk.NW)
    coin.update()
    
    img = cv2.resize(img,(500,500))
    img_opt = interface_template(img)
    img_opt = ImageTk.PhotoImage(Image.fromarray(img_opt))
    output.create_image(0, 0, image=img_opt, anchor=tk.NW)
    output.update()

def open_cam():
    global _capture, _frame
    
    if cam['text'] == "Open Camera":
        _capture, _frame = True, None
        cam.config(text="Capture Image")
        capture_img()
    elif cam['text'] == "Capture Image":
        _capture = False
        cam.config(text="Open Camera")
        detect(cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB))
        
    return

def capture_img():
    global _capture, _frame, photo, camera
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened() & _capture:
        ret, _frame = cap.read()
        
        if not ret:
            break
        
        rgb = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (500, 500))
        
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        camera.create_image(0, 0, image=photo, anchor=tk.NW)
        camera.update()
    
    _frame = cv2.resize(_frame, (1000, 1000))
    cap.release()
    

def open_vdo():
    global photo, camera
    
    file_types = [('Image Files', '*.jpg;*.png')]
    file_path = filedialog.askopenfilename(filetypes=file_types)
    
    if file_path:
        img = cv2.imread(file_path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (500, 500))
        
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        camera.create_image(0, 0, image=photo, anchor=tk.NW)
        camera.update()
        
        detect(cv2.resize(rgb, (1000, 1000)))
    return

root = tk.Tk()
root.geometry("1200x1000")
root.title("Money Detection")
root.resizable(False, False)

camera = tk.Canvas(root, width=500, height=500, bg='grey')
camera.grid(row=0, column=0)

output = tk.Canvas(root, width=500, height=500, bg='grey')
output.grid(row=0, column=1)

bank = tk.Canvas(root, width=500, height=500, bg='grey')
bank.grid(row=1, column=0)

coin = tk.Canvas(root, width=500, height=500, bg='grey')
coin.grid(row=1, column=1)

cam = tk.Button(root, text='Open Camera', width=20, height=2, command=open_cam)
cam.grid(row=0, column=2)

img = tk.Button(root, text='Open Image', width=20, height=2, command=open_vdo)
img.grid(row=1, column=2)

root.mainloop()