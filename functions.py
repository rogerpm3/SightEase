import cv2
import easyocr
es_reader = easyocr.Reader(['es'])
en_reader = easyocr.Reader(['en'])
import time
from ultralytics import YOLO
model=YOLO('best.pt')
import numpy as np
import webcolors
import numpy as np
import webcolors
import torch
from collections import OrderedDict
from torchvision import models
import os
import torch.nn as nn
from gtts import gTTS
from playsound import playsound
import os
import speech_recognition as sr

def yes_or_no():
    r = sr.Recognizer()
    detected=False
    MyText=''
    while (detected==False):
            try:
                    
                    # use the microphone as source for input.
                with sr.Microphone() as source2:
                    MyText=''
                    r.adjust_for_ambient_noise(source2, duration=0.2)
                    print('speak')
                    audio2 = r.listen(source2)
                    print('recognizing')
                    MyText = r.recognize_google(audio2)
                    MyText = MyText.lower()
                    print(MyText)
                
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
                    
            except sr.UnknownValueError:
                print("unknown error occurred")
            
            if 'yes' in MyText or 'si' in MyText:
                detected=True
                return True,MyText
            elif 'no' in MyText:
                detected=True
                return False,MyText
            
            
def ocr(spanish,frame,food):
    if spanish:
        result = es_reader.readtext(frame)
    else:
        result = en_reader.readtext(frame)
    for detection in result:
        print(detection)
        if len(detection)>1 and detection[2]>0.95:
            print(detection[2])
            speak(detection[1],spanish)
            food=False
    return food

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

def configuration_options(food,clothes,spanish,run,first):
    if cv2.waitKey(1) & 0xFF == ord('l'):
        r = sr.Recognizer()
        detected=False
            # exceptions at the runtime
        while (detected==False):
            try:
                    
                    # use the microphone as source for input.
                with sr.Microphone() as source2:
                    MyText=''
                    r.adjust_for_ambient_noise(source2, duration=0.2)
                    print('speak')
                    audio2 = r.listen(source2)
                    print('recognizing')
                    MyText = r.recognize_google(audio2)
                    MyText = MyText.lower()
                    print(MyText)
                
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
                    
            except sr.UnknownValueError:
                print("unknown error occurred")

            if 'food' in MyText:
                food=True
                detected=True
                first=True
            
            if 'clothes' in MyText:
                speak('Clothes color mode',spanish)
                time.sleep(2)
                clothes=True
                detected=True

            if 'language' in MyText:
                if spanish:
                    speak('Language changing to English',spanish)
                    spanish=False
                    time.sleep(2)
                    detected=True

                else:
                    spanish=True
                    speak('Cambiando idioma a español',spanish)
                    time.sleep(2)
                    
                    detected=True


    if cv2.waitKey(1) & 0xFF == ord('f'):
        food=True
        first=True
        time.sleep(1)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        speak('Clothes color mode',spanish)
        time.sleep(2)
        clothes=True

    if cv2.waitKey(1) & 0xFF == ord('e'):
        if spanish:
            spanish=False
            speak('Language changing to English',spanish)
            
            time.sleep(2)
        else:
            speak('Cambiando idioma a español',spanish)
            time.sleep(2)
            spanish=True
    if cv2.waitKey(1) & 0xFF == ord('q'):
    
        run=False

    MyText=''
    return food,clothes,spanish,run,first

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
            if conf.item()>0.7:
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

def make_sentence(probabilities,predictions,spanish,get_label_es,gender,class_label,labels_es,colors_es,closest_color_name,get_label,labels):
    if probabilities[0][predictions.item()]>=0.7:
        if spanish:

            pattern=get_label_es[gender[int(class_label.item())]][predictions.item()]
            frase=labels_es[int(class_label.item())],colors_es[gender[int(class_label.item())]][closest_color_name],pattern
            frase=' '.join(frase)
        else:
            pattern=get_label[predictions.item()]
            frase=pattern,closest_color_name,labels[int(class_label.item())]
            frase=' '.join(frase)
    else: 
        if spanish:
            frase=labels_es[int(class_label.item())][int(class_label.item())],colors_es[gender[int(class_label.item())][closest_color_name]]
            frase=' '.join(frase)

        else:
            frase=closest_color_name,labels[int(class_label.item())]
            frase=' '.join(frase)
    return frase

def speak(text, spanish):
    if spanish==True:
        lang='es'
    else:
        lang='en'
    # Create a gTTS object
    tts = gTTS(text=text, lang=lang)

    # Save the audio file
    tts.save("output.mp3")
    time.sleep(0.2)
    playsound("output.mp3")
    time.sleep(0.5)
    os.remove("output.mp3")



def nutrition(dictionary):
    clean_dictionary = {}
    x = dictionary["nutrition_facts"]
    a = x.split(",")

    for i in (a):
        i = i.split(" ")
        info = i[-2:]
        info_str = " ".join(info)
        name = i[:-2]


        try:
          list_name = name.remove(" ")
          name_str = " ".join(list_name).lower()
        except:
          name_str = "".join(name).lower()

        clean_dictionary[name_str] = info_str


    return clean_dictionary



def getclean_dictionary(dic_dirty):
    dic = dic_dirty['products']
    dictionary = dic[0]
    clean_dictionary = nutrition(dictionary)

    size = dictionary["size"]
    clean_dictionary["size"] = size

    ingredients = dictionary["ingredients"]
    clean_dictionary["ingredients"] = ingredients

    return clean_dictionary


'''
def translate_spa_eng(text,mode):

    translator = Translator()

    if mode == "eng_esp":
      translation = translator.translate(text, src='en', dest='es')
      output = translation.text
      return output

    elif mode == "esp_eng":
      translation = translator.translate(text, src='es', dest='en')
      output = translation.text
      return output
'''



def more_info(input_audio, dictionary):

    keywords = ["size", "fat", "salt", "protein"]
    clean_dictionary = getclean_dictionary(dictionary)

    audio  = input_audio.lower().split()

    output = []

    for k in keywords:

        if k in audio:
            info = clean_dictionary[k]
            if (info != " "):
                if k == "fat":
                    k = "saturated fat"
                text = f"it has {info} of {k}"
                output.append(text)

            else:
                text = f"i do not have information about {info}"
                output.append(text)


    output_sentence = " and ".join(output)

    print(output_sentence)
    speak(output_sentence,False)
