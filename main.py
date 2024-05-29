import cv2
import easyocr
es_reader = easyocr.Reader(['es'])
en_reader = easyocr.Reader(['en'])
import pyttsx3
import time
from ultralytics import YOLO
model=YOLO('best.pt')
import numpy as np
import webcolors
import numpy as np
import webcolors
import torch
from collections import OrderedDict
from torchvision import transforms, models
import os
import torch.nn as nn
from PIL import Image
import requests
from pyzbar.pyzbar import decode
from functions import load_checkpoint, configuration_options, create_bb, closest_color, find_dominant_color, make_sentence, speak, ocr, yes_or_no, more_info


class_mapping={'checked': 0, 'graphic': 1, 'plain': 2, 'stripe': 3}
get_label={0: 'checked', 1: 'grafica', 2: 'plain', 3: 'stripe'}
get_label_es=[{0: 'de cuadros', 1: 'grafico', 2: 'liso', 3: 'de rallas'},{0: 'de cuadros', 1: 'grafica', 2: 'lisa', 3: 'de rallas'}]
pattern_model = load_checkpoint('20240514_resnet50.pth', class_mapping)
colors_es=[{'red':'rojo','blue':'azul','yellow':'amarillo','white':'blanco','black':'negro','green':'verde','grey':'gris','magenta':'Morado','cyan':'azul','pink':'rosa','orange':'naranja'},{'red':'roja','blue':'azul','yellow':'amarilla','white':'blanca','black':'negra','green':'verde','grey':'gris','magenta':'Morada','cyan':'azul','pink':'rosa','orange':'naranja'}]
labels=['sunglass','hat','jacket','shirt','pants','shorts','skirt','dress','bag','shoe']
labels_es=['gafas de sol','gorra','chaqueta','camiseta','pantalones','pantalones cortos','falda','vestido','bolso','zapatos']
spanish=False
im_size = 224
transformation = transforms.Compose(
            [   transforms.Resize(256),
                transforms.CenterCrop(im_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
gender=[1,1,1,1,0,0,1,0,0,0]
food=False
clothes=False
run=True
first=True

vid = cv2.VideoCapture(0)
vid.set(3,200)
vid.set(4,200)

while(run==True):
    #inside infinity loop
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    
    #reading function
    if food:
        
        if first==True:
            
            start_time=time.time()

        first=False
        processed_codes = set()
        barcode_detected=False
        for barcode in decode(frame):
            barcode_detected = True
            myData = barcode.data.decode('utf-8')

        
            if myData not in processed_codes:  
                processed_codes.add(myData)  
                
                api_key = "3nb94dtsnold4frn9glj3fib7iix7h6a" 
                api_url = f"https://api.barcodelookup.com/v3/products?barcode={myData}&formatted=y&key={api_key}"
                response = requests.get(api_url)

                if response.status_code == 200:
                    json_data = response.json()

                    title = json_data.get('products')[0].get('title')
                    conversation_text = f"You got {title}. ¿Do you want more information about this food?"
                    speak(conversation_text,spanish)
                    want_more_info,input_audio=yes_or_no()
                    if want_more_info:
                        more_info(input_audio,json_data)

                else:
                    print("Failed to fetch data from API")

                start_time = time.time() 

        elapsed_time = time.time() - start_time
        print(elapsed_time)
        if not barcode_detected and elapsed_time > 7:
            
            food=ocr(spanish,frame,food)

    #clothes prediction model
    if clothes:
        images=model.predict(frame,show=False, conf=0.4, line_thickness=3)

        bounding_boxes=create_bb(images)
        
        if len(bounding_boxes)>0:
            for i in range(0,len(bounding_boxes)):
                #Crop image from frame and bounding boxes
                height, width, _ = frame.shape
                x1,y1,x2,y2, class_label, confi = bounding_boxes[i]
                bbox_tensor=[x1,y1,x2,y2]
                bbox_tensor = [int(t.item()) for t in bbox_tensor]
                x1,y1,x2,y2=bbox_tensor
                cropped_image = frame[y1:y2 , x1:x2]
                
                #Find the most dominant color in the cropped image
                if confi.item()>0.7:
                    color=find_dominant_color(cropped_image)
                    b,g,r=color

                    #find color name from rgb
                    closest_color_name = closest_color([r,g,b])
                
                    #Predict clothes patterns
                    pattern_model.eval()
                    img=PIL_image = Image.fromarray(np.uint8(cropped_image)).convert('RGB')
                    img = transformation(img)
                    image = img.view([1, img.shape[0], img.shape[1], img.shape[2]])
                    
                    with torch.no_grad():
                        output = pattern_model.forward(image)
                        probabilities = torch.softmax(output,dim=1)
                        predictions = torch.argmax(probabilities)
                        print(probabilities)
                        
                        frase=make_sentence(probabilities,predictions,spanish,get_label_es,gender,class_label,labels_es,colors_es,closest_color_name,get_label,labels)
                    
                    speak(frase,spanish)
                    print(frase)
                    clothes=False
    
    seconds=time.time()
    while (seconds+0.5)>time.time():

        food,clothes,spanish,run,first=configuration_options(food,clothes,spanish,run,first)
        
vid.release()
# Destroy all the windows
cv2.destroyAllWindows() 