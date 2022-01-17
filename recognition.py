import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

scaleFactor = 1.1
minNeighbor = 20

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

while(cap.isOpened):
    check, frame = cap.read()
    if check == True:
        gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face_detect = face_cascade.detectMultiScale(gray_img,scaleFactor,minNeighbor)
        for(x,y,w,h) in face_detect:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=3)

            id,confi = clf.predict(gray_img[y:y+h,x:x+w])

            if confi <= 70:
                cv2.putText(frame,"Korn",(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),cv2.LINE_4)
            else:
                cv2.putText(frame,"Unknow_face",(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),cv2.LINE_4)
            if (confi < 100):
                confi = " {0}%".format(round(100 - confi))
            else:
                confi = " {0}%".format(round(100 - confi))
            print(str(confi))
    
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord("e"):
            break
    else:
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
