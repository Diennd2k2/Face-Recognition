import cv2
import numpy as np
import mysql.connector as mysql
import os

#Kết nối database
def connect():
    connection = mysql.connect(
        host="localhost",
        user="root",
        password="12345678",
        database="face_recognition"
    )
    return connection

#Thêm dữ liệu
def insert(id, name):
    con = connect()
    conn = con.cursor()
    query = "insert into Users(id,name) values("+str(id)+", '"+str(name)+"')"
    cursor = conn.execute(query)
    
    con.commit()
    con.close()

#Nhập dữ liệu
id = input("ID: ")
name = input("Name: ")
insert(id, name)

#Phát hiện khuôn mặt trong ảnh
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#Mở camera
cap = cv2.VideoCapture(0)
sampleNum = 0


while(True):
    #Lấy khung ảnh từ webcam
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Tìm kiếm khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    sampleNum += 1
    #Lưu ảnh đã chụp vào file
    cv2.imwrite('dataset/User.'+str(id)+'.'+str(sampleNum)+'.jpg', gray[y:y+h,x:x+w])

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    if(sampleNum>=200):
        break

cap.release()
cv2.destroyAllWindows()