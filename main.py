import numpy as np
import torch
import cv2
import cvzone
import math
from sort import *

# Carregando video e mascara
cap = cv2.VideoCapture('video\\cars_video.mp4')
mask = cv2.imread('mask.png')

# Obter as propriedades do vídeo
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define o codec e cria o objeto VideoWriter
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('video\\output_video.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (frame_width, frame_height))

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# model.conf = 0.65  # NMS confidence threshold
model.iou = 0.15  # NMS IoU threshold
count = 0

# Limite das coordenadas utilizadas para a contagem dos carros
limits = [180, 410, 1279, 410]

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
totalCount = []

while True:
    ret, img = cap.read()
    
    if ret == True:
        # mask = cv2.imread('mask.png')

        imgGraphics = cv2.imread("assets\\graphics_cars.png", cv2.IMREAD_UNCHANGED)
        img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

        imgRegion = cv2.bitwise_and(img, mask)

        count += 1
        if count % 1 != 0:
            continue
        # img = cv2.resize(img,(600,500))

        results = model(imgRegion)
        detections = np.empty((0, 5))

        cv2.line(img, (limits[0],limits[1]), (limits[2],limits[3]), (0,0,255), 3)

        # Exibindo as detecçõs
        # a = results.pandas().xyxy[0]
        # print(a)

        for index, row in results.pandas().xyxy[0].iterrows():
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])

            d = (row['class'])

            # Confidence
            conf = math.ceil((row['confidence'] * 100)) / 100
    
            if d == 2:
                # cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

                rectx1, recty1 = ((x1+x2)/2, (y1+y2)/2)
                rectcenter = int(rectx1), int(recty1)
                cx = rectcenter[0]
                cy = rectcenter[1]

                # cv2.circle(img, (cx,cy), 3, (0,255,0), -1)
                # cv2.putText(img, str(b), (x1,y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray)) 

            resultsTracker = tracker.update(detections)

            for result in resultsTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # print(result)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                cvzone.putTextRect(img, f'id: {int(id)} ', (max(0, x1), max(35, y1)),
                                scale=1, thickness=1, offset=10)

                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                if limits[1] - 15 < cy < limits[3] + 15:
                    if totalCount.count(id) == 0:
                        totalCount.append(id)
                        cv2.line(img, (limits[0],limits[1]), (limits[2],limits[3]), (0,255,0), 3)

            # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
            cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),8)

            # Escrever o frame no arquivo de vídeo
            out.write(img)

            # Exibindo frame atual
            img = cv2.resize(img,(640,360))
            cv2.imshow("Image", img)
            cv2.imshow("ImageRegion", imgRegion)
          
            if cv2.waitKey(1) & 0xFF == ord('s'): 
                break

    else:
        break

# When everything done, release the video capture and video write objects 
cap.release() 
out.release() 

# Closes all the frames 
cv2.destroyAllWindows() 

print("The video was successfully saved") 