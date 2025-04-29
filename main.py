# Eliminar la importación no utilizada de PIL
import cv2
import numpy as np
import time

# --- Constantes ---
V1_PATH = './videos/traffic_3.mp4'
V2_PATH = './videos/traffic_2.mp4'
CASCADE_SOURCE = './cars.xml'

# Parámetros del semáforo
AMBER_TIME = 2.0       # segundos fijos en ámbar
MIN_GREEN_TIME = 3.0   # mínimo tiempo en verde por ciclo (segundos)
MAX_GREEN_TIME = 10.0  # máximo tiempo en verde por ciclo (segundos)
EPS = 1e-6             # pequeño valor para evitar división por cero

# Parámetros de procesamiento y visualización
SMOOTHING_FACTOR = 0.3 # Factor para suavizado exponencial del conteo
ESC_KEY = 27           # Código de la tecla ESC para salir
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480

# Estados y colores para los semáforos
STATES = ["VERDE", "AMBAR", "ROJO"]
COLORS = {
    "VERDE": (0, 255, 0),   # BGR - Verde
    "AMBAR": (0, 255, 255), # BGR - Amarillo
    "ROJO": (0, 0, 255)     # BGR - Rojo
}

# Cargar clasificador Haar Cascade
car_cascade = cv2.CascadeClassifier(CASCADE_SOURCE)
if car_cascade.empty():
    print(f"Error: No se pudo cargar el clasificador Haar desde {CASCADE_SOURCE}")
    exit()

def process_frame(frame):
    """Preprocesa un frame y detecta coches."""
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    # Las operaciones morfológicas pueden necesitar ajustes según el video
    # dil    = cv2.dilate(blur, np.ones((3,3)), iterations=1)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # final  = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, kernel)
    # Usar directamente el blur puede ser suficiente y más rápido
    cars   = car_cascade.detectMultiScale(blur, 1.1, 2) # Ajustar parámetros si es necesario
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
    return frame, len(cars)

def update_semaphore_state(current_time, semaforo, other_semaforo, count_hist, other_count_hist):
    """Actualiza el estado de un semáforo basado en el tiempo y el tráfico."""
    estado_actual = semaforo["estado"]
    tiempo_inicio = semaforo["inicio"]
    duracion_verde_actual = semaforo["duracion_verde"]
    
    if estado_actual == "VERDE" and current_time - tiempo_inicio > duracion_verde_actual:
        semaforo["estado"] = "AMBAR"
        semaforo["inicio"] = current_time
    elif estado_actual == "AMBAR" and current_time - tiempo_inicio > AMBER_TIME:
        semaforo["estado"] = "ROJO"
        semaforo["inicio"] = current_time
    elif estado_actual == "ROJO" and current_time - tiempo_inicio > other_semaforo["duracion_verde"] + AMBER_TIME:
        semaforo["estado"] = "VERDE"
        semaforo["inicio"] = current_time
        # Recalcular duración verde basado en tráfico histórico
        total_count = count_hist + other_count_hist + EPS
        proportion = count_hist / total_count
        semaforo["duracion_verde"] = MIN_GREEN_TIME + (MAX_GREEN_TIME - MIN_GREEN_TIME) * proportion
        # Asegurar que la duración esté dentro de los límites
        semaforo["duracion_verde"] = max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, semaforo["duracion_verde"]))

def get_frame(cap, last_frame, is_green, default_shape):
    """Obtiene un frame del video o usa el último frame si no está en verde."""
    frame = None
    read_success = False
    if is_green:
        ret, frame = cap.read()
        if ret:
            read_success = True
        else:
            # Fin del video o error de lectura, usar último frame si existe
            if last_frame is not None:
                frame = last_frame.copy()
            else:
                # No hay último frame, crear uno negro
                frame = np.zeros((default_shape[0], default_shape[1], 3), dtype=np.uint8)
    else:
        # No está en verde, usar último frame si existe
        if last_frame is not None:
            frame = last_frame.copy()
        else:
            # Intentar leer un frame si no hay buffer (inicio o error previo)
            ret, frame = cap.read()
            if ret:
                read_success = True # Aunque no esté en verde, lo leímos ahora
            else:
                # No hay último frame y no se pudo leer, crear uno negro
                frame = np.zeros((default_shape[0], default_shape[1], 3), dtype=np.uint8)
                
    return frame, read_success

def draw_info(frame, semaforo_state, current_count, video_id):
    """Dibuja el estado del semáforo y el conteo de autos en el frame."""
    color = COLORS[semaforo_state]
    
    # Determinar color del texto basado en el estado del semáforo
    if semaforo_state == "AMBAR":
        text_color = (0, 0, 0)  # Negro para fondo Ámbar
    else:
        text_color = (255, 255, 255) # Blanco para fondo Verde o Rojo
        
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), color, thickness=-1)
    cv2.putText(frame, f"SEM: {semaforo_state}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2) # Usar color de texto determinado
    cv2.putText(frame, f"V{video_id}: {current_count} autos", (frame.shape[1] - 150, 20), # Ajustar posición
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2) # Usar color de texto determinado

def detect_two_videos(path1, path2):
    """Función principal que procesa dos videos y controla los semáforos."""
    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)

    if not cap1.isOpened():
        print(f"Error: No se pudo abrir el video {path1}")
        if cap2.isOpened(): cap2.release()
        return
    if not cap2.isOpened():
        print(f"Error: No se pudo abrir el video {path2}")
        cap1.release()
        return

    # Obtener dimensiones del frame (intentar con el primero que funcione)
    frame_h, frame_w = DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH
    ret1, f1_init = cap1.read()
    if ret1:
        frame_h, frame_w, _ = f1_init.shape
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rebobinar
    else:
        ret2, f2_init = cap2.read()
        if ret2:
            frame_h, frame_w, _ = f2_init.shape
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rebobinar
        else:
            print("Advertencia: No se pudo leer el primer frame de ninguno de los videos.")
            print(f"Usando dimensiones por defecto: {frame_w}x{frame_h}")

    default_shape = (frame_h, frame_w)

    cv2.namedWindow('Comparación de videos', cv2.WINDOW_NORMAL) # Permitir redimensionar

    # Estado inicial para cada vía
    semaforo1 = {"estado": "VERDE", "inicio": time.time(), "duracion_verde": MIN_GREEN_TIME}
    semaforo2 = {"estado": "ROJO", "inicio": time.time(), "duracion_verde": MIN_GREEN_TIME}

    last_frame1 = None
    last_frame2 = None
    count1_hist = 0.0
    count2_hist = 0.0

    try:
        while True:
            current_time = time.time()

            # --- Actualizar Estados de Semáforos ---
            # Se actualizan ambos antes de leer frames para asegurar consistencia
            update_semaphore_state(current_time, semaforo1, semaforo2, count1_hist, count2_hist)
            update_semaphore_state(current_time, semaforo2, semaforo1, count2_hist, count1_hist)

            # --- Procesar Video 1 ---
            frame1, read1_success = get_frame(cap1, last_frame1, semaforo1["estado"] == "VERDE", default_shape)
            if read1_success: # Actualizar buffer solo si se leyó un nuevo frame
                 last_frame1 = frame1.copy()
            # Asegurar tamaño correcto antes de procesar
            if frame1.shape[0] != frame_h or frame1.shape[1] != frame_w:
                 frame1 = cv2.resize(frame1, (frame_w, frame_h))

            processed_frame1, count1 = process_frame(frame1)
            count1_hist = (1 - SMOOTHING_FACTOR) * count1_hist + SMOOTHING_FACTOR * count1
            draw_info(processed_frame1, semaforo1["estado"], count1, 1)

            # --- Procesar Video 2 ---
            frame2, read2_success = get_frame(cap2, last_frame2, semaforo2["estado"] == "VERDE", default_shape)
            if read2_success:
                 last_frame2 = frame2.copy()
            if frame2.shape[0] != frame_h or frame2.shape[1] != frame_w:
                 frame2 = cv2.resize(frame2, (frame_w, frame_h))

            processed_frame2, count2 = process_frame(frame2)
            count2_hist = (1 - SMOOTHING_FACTOR) * count2_hist + SMOOTHING_FACTOR * count2
            draw_info(processed_frame2, semaforo2["estado"], count2, 2)

            # --- Combinar y Mostrar ---
            # Asegurar que ambos frames procesados tengan exactamente el mismo tamaño final
            if processed_frame1.shape != processed_frame2.shape:
                 # Si el redimensionamiento anterior falló o process_frame cambió tamaño
                 target_h = max(processed_frame1.shape[0], processed_frame2.shape[0])
                 target_w = max(processed_frame1.shape[1], processed_frame2.shape[1])
                 processed_frame1 = cv2.resize(processed_frame1, (target_w, target_h))
                 processed_frame2 = cv2.resize(processed_frame2, (target_w, target_h))
                 
            combined = cv2.hconcat([processed_frame1, processed_frame2])

            # Información general en la parte inferior
            info_text = (f"V1: {count1} ({count1_hist:.1f}) T:{semaforo1['duracion_verde']:.1f}s | "
                         f"V2: {count2} ({count2_hist:.1f}) T:{semaforo2['duracion_verde']:.1f}s")
            info_bar_y = combined.shape[0]
            cv2.rectangle(combined, (0, info_bar_y - 30), (combined.shape[1], info_bar_y), (0, 0, 0), thickness=-1)
            cv2.putText(combined, info_text, (10, info_bar_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) # Texto blanco, grosor 2

            cv2.imshow('Comparación de videos', combined)

            if cv2.waitKey(1) & 0xFF == ESC_KEY:
                print("Saliendo...")
                break

    finally:
        print("Liberando recursos...")
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_two_videos(V1_PATH, V2_PATH)