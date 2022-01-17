import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

scaleFactor = 1.1
minNeighbor = 20
imgID = 0
coords = []

while(cap.isOpened):
    check, frame = cap.read()
    if check == True:
        gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face_detect = face_cascade.detectMultiScale(gray_img,scaleFactor,minNeighbor)

        for(x,y,w,h) in face_detect:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=3)
            cv2.putText(frame,"face",(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),cv2.LINE_4)
            coords = [x,y,w,h]
        if len(coords) == 4:
            id = 1
            result = frame[coords[1]: coords[1]+coords[3], coords[0]: coords[0] + coords[3]]
            cv2.imwrite("face_data/data." + str(id) + "." + str(imgID) + ".jpg",result)
        
        imgID += 1


        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord("e"):
            break
    else:
        break
cv2.waitKey(0)
cv2.destroyAllWindows()