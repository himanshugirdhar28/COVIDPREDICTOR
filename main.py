from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivy_garden.xcamera.xcamera import ButtonBehavior
from kivy.metrics import dp
from kivy.graphics import *
# from jnius import autoclass
# from android.permissions import request_permissions, Permission
# from android.storage import app_storage_path, primary_external_storage_path
# from os.path import join
# from os import path,mkdir
# from android import loadingscreen
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from kivy.properties import Clock
from kivy.uix.label import Label
from kivy.graphics import Ellipse
from kivy.uix.label import Label
from kivy.lang import Builder
from camera4kivy.preview import Preview
from PIL import Image
from kivy.core.audio import SoundLoader
import time
import pickle
import numpy as np
from PIL import Image
from kivy.properties import NumericProperty
import math


# loadingscreen.hide_loading_screen()

Window.size = (1080/3.5, 2408/3.5)


# Environment = autoclass('android.os.Environment')
# Window.fullscreen='auto'



kv = '''
FloatLayout:
    Label:
        text: 'CovidPredictor'
        size_hint: None, None
        color: "#00BFFF"
        font_size: dp(30)
        pos_hint: {'center_x': 0.5, 'top': 0.97}
'''

class knn:
    def __init__(self):
        self.pos = None
        self.neg = None

    def fit(self, pos_list, neg_list):
        self.pos=pos_list
        self.neg=neg_list

    def predict(self, i):
        distance_vector_pos=[]
        distance_vector_neg=[]
        for j in self.pos:
            distance_vector_pos.append(math.dist(i,j))
        for j in self.neg:
            distance_vector_neg.append(math.dist(i,j))
        distance_vector_pos.sort()
        distance_vector_neg.sort()
        count_pos=0
        count_neg=0
        x=0
        y=0
        k=5
        while(count_pos+count_neg<k):
            if(distance_vector_pos[x]<distance_vector_neg[y]):
                x+=1
                count_pos+=1
            else:
                y+=1
                count_neg+=1
        if(count_pos>count_neg):
            result_knn="positive"
        else:
            result_knn="negative"
        return(result_knn)
    


class CircularButton(ButtonBehavior, Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            # print(self.state)
            self.p=(self.center_x-dp(25),dp(55)) 
            self.e=Ellipse(source="photo10.png",size=(dp(50),dp(50)),pos=self.p)
        Clock.schedule_interval(self.update,1/30)
        # Clock.schedule_interval(self.update1,1)
        self.a=CovidPredictor()
    def collide_point(self, x, y):
        if x>self.center_x-dp(25) and x<self.center_x+dp(25) and y>dp(55) and y<dp(105):
            self.a.shootp()
    def update(self,dt):
        self.e.pos=(self.center_x-dp(25),dp(55))     
    # def update1(self,dt):
    #     Window.fullscreen='auto'
kv2 = '''
FloatLayout:
    Label:
        id: covid_result
        text: "Covid Result"
        size_hint: None, None
        color: "#00BFFF"
        font_size: dp(30)
        pos_hint: {'center_x': 0.5, 'top': 0.9}
'''      
class CustomAnalyzer(Preview):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def on_tex(self, *args):
        # Set the exposure for the camera instance
        self.exposure=1
        self.camera.exposure = 1
        super().on_tex(*args)
    def analyze_pixels_callback(self, pixels, size, image_pos,image_scale, mirror):
        self.s=size
        self.ima=image_pos
        global pil_image
        pil_image = Image.frombytes(mode='RGBA', size=size,data= pixels)
        self.w, self.h = pil_image.size
        left = 3.5*self.w/8
        top = 6*self.h / 7
        right = 4.5*self.w/8
        bottom = self.h
        global im
        im=pil_image.crop((left, top, right, bottom))
    def canvas_instructions_callback(self, texture, tex_size, tex_pos):
        (wid,hei)=tex_size
        wid=-1*wid
        (x1,y1)=tex_pos
        Color(1,1,1,0.8)
        Mesh(vertices=[x1-wid,y1,0,0,x1-wid, y1+hei,0,0,x1+3.5*wid/8-wid, y1+hei,0,0,x1+3.5*wid/8-wid,y1,0,0],indices=[0,1,2,3],mode='triangle_fan',color="#00BFFF")
        Mesh(vertices=[x1+3.5*wid/8-wid, y1+hei,0,0,x1+3.5*wid/8-wid,y1+hei/7,0,0,x1+4.5*wid/8-wid,y1+hei/7,0,0,x1+4.5*wid/8-wid,y1+hei,0,0],indices=[0,1,2,3],mode='triangle_fan')
        Mesh(vertices=[x1+4.5*wid/8-wid,y1+0,0,0,x1+4.5*wid/8-wid,y1+hei,0,0,x1,y1+hei,0,0,x1,y1,0,0],indices=[0,1,2,3],mode='triangle_fan')
class CovidPredictor(MDApp,ButtonBehavior,Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.iso = NumericProperty(100)
        # self.white_balance = NumericProperty(4000)
        # self.exposure_time = NumericProperty(1 / 60)
    def getPixels(self,filename):
        img = Image.open(filename, 'r')
        w, h = img.size
        pix = list(img.getdata())
        return [pix[n:n+w] for n in range(0, w*h, w)]
    def rgb_matrix_value(self,sample_rgb_list):
        sum1=0
        sum2=0
        sum3=0
        count=0
        for j in sample_rgb_list:
            for k in j:
                sum1+=k[0]**2
                sum2+=k[1]**2
                sum3+=k[2]**2
                count+=1
        return([(sum1/count)**0.5,(sum2/count)**0.5,(sum3/count)**0.5])
    def result(self,file_path):
        with open("S:\\INFORMATION TECHNOLOGY\\3rd sem\\MINI PROJECT\DATA\\new_Data\\knn_model.pkl","rb") as f:
            self.model=pickle.load(f)
        # print(model.predict("predict.jpg"))
        rgb_matrix=self.getPixels(file_path)
        sample_rgb_value=self.rgb_matrix_value(rgb_matrix)
        return(self.model.predict(sample_rgb_value))
    def build(self):
        global cam
        cam=CustomAnalyzer(aspect_ratio='4:3')
        # print(cam.aspect_ratio)
        cam.flash(state='off')
        global screen
        screen=MDScreen()
        screen.add_widget(cam)
        screen.add_widget(CircularButton())
        self.bd=Builder.load_string(kv)
        screen.add_widget(self.bd) 
        global bd2
        bd2=Builder.load_string(kv2)
        screen.add_widget(bd2)
        with screen.canvas:
            self.nl=Label(text='Align the sample inside \n  the rectangle properly',font_size=dp(20),color=(0,0,0,1))
            self.nl.pos=(Window.width/2-self.nl.width/2,Window.height/2+self.nl.height/2)
        Clock.schedule_interval(self.nlpos,1/30)
        return screen
    def nlpos(self,dt):
        self.nl.pos=(Window.width/2-self.nl.width/2,Window.height/2+self.nl.height/2)
    def abc(self,path):
        pass
    def on_start(self):
        # request_permissions([Permission.CAMERA,Permission.WRITE_EXTERNAL_STORAGE,Permission.READ_EXTERNAL_STORAGE])
        Clock.schedule_once(self.connect_camera)
        
    def connect_camera(self,dt):
        self.v=cam.connect_camera(filepath_callback=self.abc,enable_analyze_pixels=True,resolution=(-1,-1),enable_focus_gesture=False,default_zoom=0,enable_zoom_gesture=False)
    # def on_camera_loaded(self, instance):
    #     #cam.iso=self.iso
    #     cam.exposure_mode = 'auto'  #'off'
    #     #cam.exposure_time=self.exposure_time
    #     cam.exposure_compensation = 0  # set exposure compensation to 0
    #     cam.white_balance_mode='auto' #'off'
    #     #self.cam.white_balance_temperature = self.white_balance
    #     Clock.schedule_interval(self.update, 1.0/30.0)
    def on_stop(self):
        cam.disconnect_camera()
        return super().on_stop()
    def shootp(self):
        # cam.capture_photo()
        date_string = time.strftime("%Y_%m_%d %H.%M.%S")
        # pp = join(primary_external_storage_path(), Environment.DIRECTORY_DCIM,"GlucoPredictor")
        # if not path.isdir(pp):
        #     mkdir(pp)
        # pp=join(pp,date_string)
        pp=date_string
        pp+=".png"
        im.save(pp)
        sound = SoundLoader.load('shutter.wav')
        sound.play()
        bd2.ids.covid_result.text=str((self.result(pp)))
        with screen.canvas:
            self.l=Label(text='Saving...',font_size=dp(12),color=(1,1,1,1))
            self.l.pos=(Window.width/2-self.l.width/2,dp(110))
        Clock.schedule_once(self.saving,0.7)
    def saving(self,dt):
        self.l.pos=(dp(-800),dp(-800))
if __name__ == "__main__":
    CovidPredictor().run()