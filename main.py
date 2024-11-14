import cv2
import mediapipe as mp
from scipy.spatial import distance
import threading
import pygame

# Inicializar pygame para reproducir sonidos
pygame.mixer.init()

# Inicializar Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
# Crear una instancia de FaceMesh con umbrales de confianza para detección y seguimiento
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Función para calcular el Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    """
    Calcula la relación de aspecto del ojo utilizando las distancias euclidianas entre puntos clave.
    
    :param eye: Lista de coordenadas de los puntos del ojo.
    :return: Valor EAR calculado.
    """
    # Calcular las distancias euclidianas entre los puntos verticales del ojo
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Calcular la distancia euclidiana entre los puntos horizontales del ojo
    C = distance.euclidean(eye[0], eye[3])
    # Calcular el EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Función para reproducir la alarma en un hilo separado
def play_alarm():
    """
    Reproduce el sonido de alarma en bucle.
    """
    pygame.mixer.music.load('alarma.mp3')  # Cargar el archivo de sonido
    pygame.mixer.music.play(-1)  # Reproducir en bucle

# Función para detener el sonido de la alarma
def stop_alarm():
    """
    Detiene la reproducción del sonido de la alarma.
    """
    pygame.mixer.music.stop()

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

# Reducir la resolución del video para mejorar el rendimiento
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_counter = 0  # Contador de cuadros procesados
skip_frames = 2  # Procesar cada 2 cuadros para reducir la carga

# Umbral de EAR para detectar ojos cerrados
ear_threshold = 0.25
consecutive_frames = 20  # Número de cuadros consecutivos con EAR bajo que indican somnolencia
counter = 0  # Contador de cuadros con EAR bajo
alarm_active = False  # Estado de la alarma

while True:
    ret, frame = cap.read()  # Capturar un cuadro del video
    if not ret:
        break  # Salir del bucle si no se puede capturar el cuadro
    
    frame_counter += 1
    if frame_counter % skip_frames != 0:
        continue  # Procesar solo los cuadros seleccionados
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir el cuadro a RGB
    
    results = face_mesh.process(rgb_frame)  # Procesar el cuadro para detectar puntos faciales
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extraer los puntos de referencia de los ojos
            left_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
            
            # Convertir los puntos a coordenadas (x, y)
            left_eye = [(int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])) for pt in left_eye_landmarks]
            right_eye = [(int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])) for pt in right_eye_landmarks]
            
            # Calcular el EAR para ambos ojos
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            # Promediar el EAR de ambos ojos
            ear = (left_ear + right_ear) / 2.0
            
            if ear < ear_threshold:  # Si el EAR está por debajo del umbral
                counter += 1  # Incrementar el contador
                if counter >= consecutive_frames:  # Si el contador alcanza el número de cuadros consecutivos
                    if not alarm_active:
                        threading.Thread(target=play_alarm).start()  # Iniciar un hilo para reproducir la alarma
                        alarm_active = True
                    # Mostrar un mensaje de advertencia en la imagen
                    cv2.putText(frame, "PELIGRO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                counter = 0  # Reiniciar el contador si el EAR está por encima del umbral
                if alarm_active:
                    threading.Thread(target=stop_alarm).start()  # Iniciar un hilo para detener la alarma
                    alarm_active = False
            
            # Dibujar los puntos de referencia faciales y los contornos
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,  # Usar los contornos faciales
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),  # Especificaciones de dibujo para los puntos
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
            )
    else:
        counter = 0  # Reiniciar el contador si no se detecta rostro
        if alarm_active:
            threading.Thread(target=stop_alarm).start()  # Detener la alarma si no hay rostro
            alarm_active = False
    
    # Mostrar el cuadro con las anotaciones
    cv2.imshow("Frame", frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
