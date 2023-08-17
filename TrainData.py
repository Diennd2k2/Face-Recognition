import cv2
import numpy as np
import os
from PIL import Image

#Object của thuật toán nhận dạng khuôn mặt LBPH
recognizer = cv2.face_LBPHFaceRecognizer.create()
path = "dataset"

#Trích xuất đặc trưng và id của khuôn mặt từ ảnh được lưu trong thư mục
def getImagesWithID(path):
    #Lấy đường dẫn của dữ liệu ảnh
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        
        #Cắt để lấy id
        id = int(os.path.split(imagePath)[-1].split('.')[1])

        faces.append(faceNp)
        ids.append(id)

    return faces, ids

faces, ids =getImagesWithID(path)

#Huấn luyện mô hình với các khuôn mặt và id tương ứng
recognizer.train(faces,np.array(ids))
print("Training Xong")

if not os.path.exists('recognizer'):
    os.makedirs('recognizer')

recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()