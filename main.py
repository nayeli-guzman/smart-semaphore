from PIL import Image
import cv2 
import numpy as np
import time

v1 = './videos/traffic_1.mp4'
v2 = './videos/traffic_2.mp4'
cascade_source = './cars.xml'
car_cascade = cv2.CascadeClassifier(cascade_source)

def process_frame(frame):
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    dil    = cv2.dilate(blur, np.ones((3,3)), iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    final  = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, kernel)
    cars   = car_cascade.detectMultiScale(final, 1.1, 1)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
    return frame, len(cars)

def detect_two_videos(path1, path2):

    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("video is corrupted or not found")
        return
    
    cv2.namedWindow('Comparación de videos')
    
    try:

        while True:
            ret1, f1 = cap1.read()
            ret2, f2 = cap2.read()

            if not ret1 and not ret2:
                break

            if not ret1 and ret2:
                f1 = np.zeros_like(f2)
            if not ret2 and ret1:
                f2 = np.zeros_like(f1)

            out1, count1 = process_frame(f1)
            out2, count2 = process_frame(f2)

            combined = cv2.hconcat([out1, out2])


            text = f"V1: {count1} autos    V2: {count2} autos"

            cv2.rectangle(combined, (0, 0), (combined.shape[1], 30), (0, 0, 0), thickness=-1)
            
            cv2.putText(combined, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Comparación de videos', combined)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
                print("saliendo...")
                return
        
    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_two_videos(v1, v2)
