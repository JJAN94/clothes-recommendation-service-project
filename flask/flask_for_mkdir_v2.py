from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
from flask.helpers import send_from_directory
import numpy as np
from threading import Thread
import pandas as pd
from annoy import AnnoyIndex
#import pickle
import json
import tensorflow as tf
from tensorflow.keras.models import Model
import ssl

global capture
capture=0



#make shots directory to save pics
try:
    os.mkdir('C:\encore_hoon\final_project\shots')
except OSError as error:
    pass

#instatiate flask app  
app = Flask(__name__, template_folder='./final')

camera = cv2.VideoCapture(0)

#json, annoy file 
musinsa=pd.read_csv("FinalMu.csv", index_col="ID_number")
musinsa['img'] = musinsa['img'].apply(lambda x : x.split('/')[-1])

with open('model/mu_index.json','r') as f:
    mu_index = json.load(f)
mu_search = AnnoyIndex(4096, 'angular')
mu_search.load('model/mu_search.ann')

with open('model/insta_index.json','r') as f:
    insta_index = json.load(f)
insta_revers_index = {v:k for k , v in insta_index.items()}
insta_search = AnnoyIndex(4096, 'angular')
insta_search.load('model/insta_search.ann')

captured_img=None

vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))
basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)

# 찍은 사진에 모델에 넣어서 예측값 넣어서 받는 함수 
def get_feature_vector(img):
    img1 = cv2.resize(img, (224, 224))
    feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
    return feature_vector

def match_mu(q): #파일 이름으로 검색하고 찾는다.

  result = mu_search.get_nns_by_vector(q, 5, include_distances=True)
  result = list(zip(result[0], result[1]))
  result = [('{}'.format(mu_index[str(no)]), score) for no, score in result]
  return result

def match_insta(fn): #파일 이름으로 검색하고 찾는다.
  img = cv2.imread(fn)
  q = get_feature_vector(img)[0]
  result = insta_search.get_nns_by_vector(q, 5, include_distances=True)
  result = list(zip(result[0], result[1]))
  result = [('{}'.format(insta_index[str(no)]), score) for no, score in result]
  return result

import os 
import glob # 파일을 찾아주는것 

insta_pair = pd.read_csv("model/insta_pair.csv")

def match_pair(fn) :
    fn_dir, fn_name = os.path.split(fn)
    new_name = fn_name.split('_')[0] + '_' + fn_name.split('_')[1] + '_' + fn_name.split('_')[2]  # 0815수정
    file_names = insta_pair.loc[insta_pair['image_name'].str.contains(new_name),"image_name"].values
    pair_keys=[insta_revers_index[v] for v in file_names]
    result = [(insta_search.get_item_vector(int(v)), 1) for v in pair_keys]
# annoy가 이미 벡터값을 가지고 있음. 
# 1. match_pair의 해당하는 이미지 파일명을 안다. - > 그 파일명으로 index 식별 (pickle 로 부터) 
# 2. index로부터 annoy에서 벡터값을 가지고 온다.
    return result

def search_pair(target) :
  result = match_insta(target)
  print('*'*30)
  print(result)
  matched = False
  for i in range(len(result)):
    print(i)
    print(result[i][0])
    temp = match_pair(result[i][0])
    if temp != None:
      result = temp
      matched = True
      break
  if not matched:
    result = [] 
  r=[]

  for i in range(len(result)) :
    temp = match_mu(result[i][0])
    for j in range(len(temp)):
      print('musinsa',temp[j])
      r.append(temp[j])
  return r

def search_similar(target) :

    img = cv2.imread(target)
    q = get_feature_vector(img)[0]
     temp = match_mu(q)

    return temp

def gen_frames():  # generate frame by frame from camera
    global  capture
    global  captured_img
    while True:
        success, frame = camera.read() 
        if success:
            if(capture):
                
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
                print(p,'!!create a picture!!')
                captured_img=p    
                capture=0
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/')
def index():
    return render_template('main.html')

@app.route('/2nd')
def n_nd():
    return render_template('2nd.html')
    
@app.route('/capture')
def capture():
    return render_template('capture.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            global captured_img
            captured_img=None
            capture=1
            time.sleep(1)
            return render_template('choice_item_type.html',captured_img=captured_img)
 
                                        
    elif request.method=='GET':
        return render_template('capture.html')
    return render_template('capture.html')

@app.route('/choice_item_type')
def choice_item_type():
    global captured_img

    print(captured_img)
    return render_template('choice_item_type.html',captured_img=captured_img)

@app.route('/shots/<path:filename>')
def download_file(filename):
    print(filename)
    return send_from_directory("shots", filename, as_attachment=True)

@app.route('/result_shop', methods=['POST','GET'] )
def result_shop():
    
    if request.method == 'POST':
        captured_img= request.form.get('captured_img')
        request_type = request.form.get('request_type')
        print('request',request_type)
        if request_type == 'similar' :
            result=search_similar(captured_img)
            result=[v[0] for v in result]

        else :
            
            result=search_pair(captured_img)
            print(result)
            result=[v[0] for v in result]

    global musinsa
    result=musinsa.loc[musinsa['img'].isin(result)].reset_index()

    return render_template('result_shop.html',item_list=result)


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=80)
    app.run()


camera.release()
cv2.destroyAllWindows()     