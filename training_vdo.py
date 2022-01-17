import cv2
import os 
import numpy as np
from PIL import Image

data_dir = "Elon_faces"
path = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]
faces = []
ids = []
for image in path:
    img = Image.open(image).convert("L")
    imgnp = np.array(img,"uint8")
    id = int(os.path.split(image)[1].split(".")[1])
    print(""+str(id))
    faces.append(imgnp)
    ids.append(id)

ids = np.array(ids)

clf = cv2.face.LBPHFaceRecognizer_create()
clf.train(faces,ids)
clf.write("classifier2.xml")
