# Reconhecimento basico de gestos usando MediaPipe e OpenCV

import cv2
import mediapipe as mp
import time 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configuração do Modelo
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')

# 2. Função de Callback (o que acontece quando um gesto é detectado)
def save_result(result, output_image, timestamp_ms: int):
    if result.gestures and len(result.gestures) > 0:
        # Pega o nome do gesto com maior pontuação
        top_gesture = result.gestures[0][0].category_name
        print(f"Gesto detectado: {top_gesture}")

options = vision.GestureRecognizerOptions( 
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=save_result
)

# 3. Inicialização
cam = cv2.VideoCapture(0)
recognizer = vision.GestureRecognizer.create_from_options(options)

print("Pressione 'q' para sair...")

while cam.isOpened():
    success, frame = cam.read() 
    if not success:
        print("Falha ao capturar frame da câmera.")
        break
    

    # Inverte a imagem horizontalmente para parecer um espelho 
    frame = cv2.flip(frame, 1)

    # Converte BGR (OpenCV) para RGB (MediaPipe)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # Timestamp em milissegundos (inteiro)
    timestamp_ms = int(time.time() * 1000)
    
    # Processamento Assíncrono
    recognizer.recognize_async(mp_image, timestamp_ms)

    # Exibe o feed da câmera
    cv2.imshow('Reconhecimento de Gestos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Limpeza de recursos
cam.release()
cv2.destroyAllWindows()
recognizer.close()