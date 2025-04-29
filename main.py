from PIL import Image
import cv2 
import numpy as np
import time

v1 = './videos/traffic_1.mp4'
v2 = './videos/traffic_2.mp4'
cascade_source = './cars.xml'
car_cascade = cv2.CascadeClassifier(cascade_source)

# Parámetros del semáforo
AMBER_TIME = 2.0       # segundos fijos en ámbar
MIN_GREEN_TIME = 3.0   # mínimo tiempo en verde por ciclo (segundos)
MAX_GREEN_TIME = 10.0  # máximo tiempo en verde por ciclo (segundos)
EPS = 1e-6             # pequeño valor para evitar división por cero

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
    # Estados y colores para los semáforos
    STATES = ["VERDE", "AMBAR", "ROJO"]
    COLORS = {
        "VERDE": (0, 255, 0),   # BGR - Verde
        "AMBAR": (0, 255, 255), # BGR - Amarillo
        "ROJO": (0, 0, 255)     # BGR - Rojo
    }
    
    # Abrir capturas de video
    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("video is corrupted or not found")
        return
    
    # Crear ventana
    cv2.namedWindow('Demo | Semáforos Inteligentes')
    
    # Estado inicial para cada vía
    semaforo1 = {"estado": "VERDE", "inicio": time.time(), "duracion_verde": MIN_GREEN_TIME}
    semaforo2 = {"estado": "ROJO", "inicio": time.time(), "duracion_verde": MIN_GREEN_TIME}  # Inicia en rojo
    
    # Buffers para guardar frames cuando el semáforo no está en verde
    last_frame1 = None 
    last_frame2 = None
    
    # Contadores de autos históricos para mejor ajuste
    count1_hist = 0
    count2_hist = 0
    
    try:
        while True:
            now = time.time()
            
            # --- VIDEO 1 ---
            # Actualizar estado del semáforo
            if semaforo1["estado"] == "VERDE" and now - semaforo1["inicio"] > semaforo1["duracion_verde"]:
                # Cambia a ámbar
                semaforo1["estado"] = "AMBAR"
                semaforo1["inicio"] = now
            elif semaforo1["estado"] == "AMBAR" and now - semaforo1["inicio"] > AMBER_TIME:
                # Cambia a rojo
                semaforo1["estado"] = "ROJO"
                semaforo1["inicio"] = now
            elif semaforo1["estado"] == "ROJO" and now - semaforo1["inicio"] > semaforo2["duracion_verde"] + AMBER_TIME:
                # Cambia a verde y recalcula tiempo basado en el tráfico
                semaforo1["estado"] = "VERDE"
                semaforo1["inicio"] = now
                # Ajustar duración verde según tráfico (proporción de autos)
                total_count = count1_hist + count2_hist + EPS
                proportion = count1_hist / total_count
                # Duración verde mapeada entre MIN y MAX según la proporción de tráfico
                semaforo1["duracion_verde"] = MIN_GREEN_TIME + (MAX_GREEN_TIME - MIN_GREEN_TIME) * proportion
            
            # Leer frame solo si está en verde
            if semaforo1["estado"] == "VERDE":
                ret1, f1 = cap1.read()
                if ret1:
                    # Guardar frame como último válido
                    last_frame1 = f1.copy()
                else:
                    # Si no hay más frames y no hay guardados, crea negro
                    if last_frame1 is None:
                        f1 = np.zeros((480, 640, 3), dtype=np.uint8)
                    else:
                        f1 = last_frame1.copy()
            else:
                # En estados AMBAR o ROJO, mostrar último frame
                if last_frame1 is None:
                    # Primera ejecución o error de lectura
                    ret1, f1 = cap1.read()
                    if ret1:
                        last_frame1 = f1.copy()
                    else:
                        f1 = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    f1 = last_frame1.copy()
            
            # Procesar frame (detección de autos)
            out1, count1 = process_frame(f1)
            # Actualizar histórico de conteo con suavizado
            count1_hist = 0.7 * count1_hist + 0.3 * count1  # Suavizado exponencial
            
            # --- VIDEO 2 ---
            # Similar a video 1
            if semaforo2["estado"] == "VERDE" and now - semaforo2["inicio"] > semaforo2["duracion_verde"]:
                semaforo2["estado"] = "AMBAR"
                semaforo2["inicio"] = now
            elif semaforo2["estado"] == "AMBAR" and now - semaforo2["inicio"] > AMBER_TIME:
                semaforo2["estado"] = "ROJO"
                semaforo2["inicio"] = now
            elif semaforo2["estado"] == "ROJO" and now - semaforo2["inicio"] > semaforo1["duracion_verde"] + AMBER_TIME:
                semaforo2["estado"] = "VERDE"
                semaforo2["inicio"] = now
                # Ajustar según tráfico
                total_count = count1_hist + count2_hist + EPS
                proportion = count2_hist / total_count
                semaforo2["duracion_verde"] = MIN_GREEN_TIME + (MAX_GREEN_TIME - MIN_GREEN_TIME) * proportion
            
            if semaforo2["estado"] == "VERDE":
                ret2, f2 = cap2.read()
                if ret2:
                    last_frame2 = f2.copy()
                else:
                    if last_frame2 is None:
                        f2 = np.zeros((480, 640, 3), dtype=np.uint8)
                    else:
                        f2 = last_frame2.copy()
            else:
                if last_frame2 is None:
                    ret2, f2 = cap2.read()
                    if ret2:
                        last_frame2 = f2.copy()
                    else:
                        f2 = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    f2 = last_frame2.copy()
            
            out2, count2 = process_frame(f2)
            count2_hist = 0.7 * count2_hist + 0.3 * count2  # Suavizado exponencial
            
            # Asegurar que los frames tengan el mismo tamaño para evitar errores en hconcat
            if out1.shape[0] != out2.shape[0] or out1.shape[1] != out2.shape[1]:
                # Redimensionar para que tengan el mismo tamaño
                target_height = max(out1.shape[0], out2.shape[0])
                target_width = max(out1.shape[1], out2.shape[1])
                if out1.shape[0] != target_height or out1.shape[1] != target_width:
                    out1 = cv2.resize(out1, (target_width, target_height))
                if out2.shape[0] != target_height or out2.shape[1] != target_width:
                    out2 = cv2.resize(out2, (target_width, target_height))
            
            # Dibujar información del semáforo sobre los videos
            # Semáforo 1
            cv2.rectangle(out1, (0, 0), (out1.shape[1], 30), COLORS[semaforo1["estado"]], thickness=-1)
            cv2.putText(out1, f"SEM: {semaforo1['estado']}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(out1, f"V1: {count1} autos", (200, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            # Semáforo 2
            cv2.rectangle(out2, (0, 0), (out2.shape[1], 30), COLORS[semaforo2["estado"]], thickness=-1)
            cv2.putText(out2, f"SEM: {semaforo2['estado']}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(out2, f"V2: {count2} autos", (200, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Combinar videos
            combined = cv2.hconcat([out1, out2])
            
            # Información general
            info_text = f"V1: {count1} autos ({count1_hist:.1f})    V2: {count2} autos ({count2_hist:.1f})    Tiempos: {semaforo1['duracion_verde']:.1f}s / {semaforo2['duracion_verde']:.1f}s"
            cv2.rectangle(combined, (0, combined.shape[0]-30), (combined.shape[1], combined.shape[0]), (0, 0, 0), thickness=-1)
            cv2.putText(combined, info_text, (10, combined.shape[0]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Mostrar resultado
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