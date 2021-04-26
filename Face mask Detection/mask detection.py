import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from imutils.video import VideoStream
def detect_face(frame,faceNet,model):
    #getting dimensions of the frame
    h,w=frame.shape[:2]
    #construct a blob that can be passed through the pre-trained network
    blob=cv2.dnn.blobFromImage(frame,1,(224,224),(104,177,123))
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections=faceNet.forward()
    faces=[]
    locs=[]
    preds=[]
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence > 0.6:
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startx,starty,endx,endy)=box.astype('int')
            (startx,statry)=(max(0,startx),max(0,starty))
            (endx,endy)=(min(w-1,endx),min(h-1,endy))
            face=frame[starty:endy,startx:endx]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)
            faces.append(face)
            faces.append(face)
            locs.append((startx,starty,endx,endy))
    if len(faces)>0:
        faces=np.array(faces,dtype='float32')
        preds=model.predict(faces,batch_size=10)
    return (locs,preds)
model=load_model('face_mask.h5')
faceNet=cv2.dnn.readNet('deploy.prototxt','res10_300x300_ssd_iter_140000.caffemodel')
cap=VideoStream(0).start()
while True:
    frame=cap.read()
    (locs,preds)=detect_face(frame,faceNet,model)
    for box,pred in zip(locs,preds):
        (startx,starty,endx,endy)=box
        mask,withoutmask=pred
        label='Mask' if mask > withoutmask else 'no mask'
        color =(0,255,0) if label == 'Mask' else (0,0,255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)
        cv2.putText(frame,label,(startx,starty-10),cv2.FONT_HERSHEY_SIMPLEX,.45,color,2)
        cv2.rectangle(frame,(startx,starty),(endx,endy),color,2)
    cv2.imshow('Webcam ',frame)
    if cv2.waitKey(1)==13:
        break

cv2.destroyAllWindows()
cap.stop()
