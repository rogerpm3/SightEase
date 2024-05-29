import cv2
import easyocr
es_reader = easyocr.Reader(['es'])
en_reader = easyocr.Reader(['en'])
import pyttsx3
import time
from ultralytics import YOLO
model=YOLO('C:/Users/roger/Desktop/Social/best.pt')
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

def load_checkpoint(filepath, class_mapping):

    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        num_classes = len(class_mapping)

        if "resnet50" in checkpoint["arch"]:
            pattern_model = models.resnet50(pretrained=True)
            num_ftrs = pattern_model.fc.in_features
            classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc", nn.Linear(num_ftrs, num_classes)),
                    ("output", nn.LogSoftmax(dim=1)),
                ] ))
            pattern_model.fc = classifier

        for param in pattern_model.parameters():
            param.requires_grad = False

        pattern_model.class_to_idx = checkpoint["class_to_idx"]        
        pattern_model.load_state_dict(checkpoint["model_state_dict"])

        return pattern_model

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
read=False
clothes=False
run=True


def configuration_options(read,clothes,spanish,run):
    if cv2.waitKey(1) & 0xFF == ord('r'):
        if read:
            read=False
            speak('Read Mode Off')
        else:
            speak('Read Mode')
            read=True
    if cv2.waitKey(1) & 0xFF == ord('c'):
        if clothes:
            speak('Clothes color mode off')
            clothes=False
            time.sleep(2)
        else:
            speak('Clothes color mode')
            time.sleep(2)
            clothes=True
    if cv2.waitKey(1) & 0xFF == ord('e'):
        if spanish:
            speak('Language changing to English')
            spanish=False
            time.sleep(2)
        else:
            speak('Language changing to Spanish')
            time.sleep(2)
            spanish=True
    if cv2.waitKey(1) & 0xFF == ord('q'):

        run=False

    return read,clothes,spanish,run

def create_bb(images):
    bounding_boxes=[]
    for prediction in images:
        boxes=prediction.boxes.cpu()
        xyxy=boxes.xyxy
        classes=boxes.cls
        confidences=boxes.conf
        for i in range(0,len(xyxy)):
            x,y,w,h=xyxy[i]
            class_label=classes[i]
            conf=confidences[i]
            if conf.item()>0.6:
                bounding_boxes.append([x, y, w, h,class_label,conf])
    return bounding_boxes

def closest_color(rgb_tuple):
    rgbbw=['red','blue','yellow','white','black','green','grey','magenta','cyan','pink','orange']

    min_distance = 1000000000000000
    min_color=''   
    #print(rgb_tuple)
    for color_name in rgbbw:
    #for _, color_name in webcolors.CSS3_HEX_TO_NAMES.items():
        try:
            # Convert color name to RGB tuple
            named_color_rgb1 = webcolors.name_to_rgb(color_name)
            if color_name=='green':
                named_color_rgb1=[0,255,0]
            named_color_rgb=[]
            for i in range(0,3):
                val=named_color_rgb1[i]
                if val==0:
                    val=35
                elif val==255:
                    val=210
                elif val>190:
                    val-=30
                named_color_rgb.append(val)

            pythonishit=[int((x**2)+1) for x in named_color_rgb]

            suma=pythonishit[0]+pythonishit[1]+pythonishit[2]

            named_color_per=[]
            for i in range(0,3):
                named_color_per.append(named_color_rgb[i]/suma)
        except ValueError:
            continue
        

        pythonishit=[int(x**2) for x in rgb_tuple]
        suma=pythonishit[0]+pythonishit[1]+pythonishit[2]+1
        per_tuple=[]
        for i in range(0,3):
            per_tuple.append(rgb_tuple[i]/suma)
        # Calculate the difference between each RGB component
        distance_r = abs(rgb_tuple[0] - named_color_rgb[0])
        distance_g = abs(rgb_tuple[1] - named_color_rgb[1])
        distance_b = abs(rgb_tuple[2] - named_color_rgb[2])
        perdistance_r = abs(per_tuple[0] - named_color_per[0])
        perdistance_r=((perdistance_r+3)**2)-9
        perdistance_g = abs(per_tuple[1] - named_color_per[1])
        perdistance_g=((perdistance_g+3)**2)-9
        perdistance_b = abs(per_tuple[2] - named_color_per[2])
        perdistance_b=((perdistance_b+3)**2)-9
        
            

        # Compute total distance
        perdistance_total= perdistance_b+perdistance_g+perdistance_r
        total_distance = (distance_r + distance_g + distance_b)*perdistance_total
        if color_name=='grey':
            total_distance+=5
        if color_name=='pink' or color_name=='orange':
            total_distance+=1

        
        # Update min_color regardless of distance
        if total_distance < min_distance:

            if color_name=='pink' and (rgb_tuple[0]-rgb_tuple[1])>=10 and (rgb_tuple[0]-rgb_tuple[2])>10:
                min_distance = total_distance
                min_color = color_name
            elif color_name!='pink':
                min_distance = total_distance
                min_color = color_name
        if (total_distance-5)<0.4 and color_name=='grey':
            min_distance = total_distance-5
            min_color = color_name
        
        if (total_distance-1)<0.3 and color_name=='pink' and (rgb_tuple[0]-rgb_tuple[1])>=10 and (rgb_tuple[0]-rgb_tuple[2])>10:
            min_distance = total_distance-1
            min_color = color_name
        if (total_distance-1)<0.3 and color_name=='orange':
            min_distance = total_distance-1
            min_color = color_name
    if min_color=='magenta':
        min_color='Purple'
    if min_color=='cyan':
        min_color='Blue'
    return min_color

def find_dominant_color(image):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))
    
    pixels = np.float32(pixels)
    
    # Define criteria and apply k-meansq
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    dominant_color = np.uint8(centers[0])
    
    return dominant_color

def make_sentence():
    if probabilities[0][predictions.item()]>=0.60:
        if spanish:

            pattern=get_label_es[gender[int(class_label.item())]][predictions.item()]
            frase=labels_es[int(class_label.item())],colors_es[gender[int(class_label.item())]][closest_color_name],pattern
        else:
            pattern=get_label[predictions.item()]
            frase=pattern,closest_color_name,labels[int(class_label.item())]
    else: 
        if spanish:
            frase=labels_es[int(class_label.item())][int(class_label.item())],colors_es[gender[int(class_label.item())][closest_color_name]]
        else:
            frase=closest_color_name,labels[int(class_label.item())]
    return frase

def speak(text):

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


    
vid = cv2.VideoCapture(0)
vid.set(3,200)
vid.set(4,200)


while(run==True):
    #inside infinity loop
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    
    #reading function
    if read:
        if spanish:
            result = es_reader.readtext(frame)
        else:
            result = en_reader.readtext(frame)
        for detection in result:
            print(detection)
            if len(detection)>1:
                speak(detection[1])

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
                if confi.item()>0.6:
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
                        
                        frase=make_sentence()
                    
                    speak(frase)
                    print(frase)
    
    seconds=time.time()
    while (seconds+0.5)>time.time():

        read,clothes,spanish,run=configuration_options(read,clothes,spanish,run)
        
vid.release()
# Destroy all the windows
cv2.destroyAllWindows() 