import cv2
import numpy as np
import mysql.connector as mysql
import datetime as dt
import os

#Object của thuật toán nhận dạng khuôn mặt LBPH
recognizer = cv2.face_LBPHFaceRecognizer.create()
#Phát hiện khuôn mặt trong hình ảnh
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#Đọc dữ liệu huấn luyện
recognizer.read('recognizer/trainingData.yml')

def connect():
    connection = mysql.connect(
        host="localhost",
        user="root",
        password="12345678",
        database="face_recognition"
    )
    return connection

#Lấy thông tin của người dùng được nhận dạng dựa trên ID
def getProfile(id):
    con = connect()
    conn = con.cursor()
    query = "select * from Users where id= "+ str(id)
    cursor = conn.execute(query)
    profile = conn.fetchone()
    con.close()
    return profile

#Ghi thông tin vào tệp CSV
def joinIn(name):
    with open("checkin.csv", "r+") as f:
        myDatalist = f.readlines()
        nameList = []
        for line in myDatalist:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = dt.datetime.now()
            dtstr = now.strftime("%H:%M:%S %d/%m/%Y")
            f.writelines(f"\n{name},{dtstr}")

cap = cv2.VideoCapture(0)
fontFace = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    ret, frame = cap.read()
    #Chuyển đổi ảnh từ không gian màu BGR (Blue, Green, Red) sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Trả về danh sách các khuôn mặt được phát hiện trong ảnh
    faces = face_cascade.detectMultiScale(gray)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_array = gray[y:y+h,x:x+w]
        id, confidence = recognizer.predict(roi_array)
        if confidence < 60:
            profile = getProfile(id)
            #Thông tin ảnh
            if(profile != None):
                cv2.putText(frame,"Name: "+str(profile[1]), (x+10, y+h+30), fontFace, 1, (0,255,0), 2)
                joinIn(str(profile[1]))
        else:
            cv2.putText(frame,"Unknow", (x+10, y+h+30), fontFace, 1, (0,255,0), 2)
    cv2.imshow('Image', frame)
    if(cv2.waitKey(1) == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()