import cv2
import time

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
else:
    face_detected = False
    detection_start_time = None
    last_face_seen_time = None

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: No se pudo leer la imagen de la cámara")
            break

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            if not face_detected:
                # Inicia el temporizador si se detecta una cara por primera vez
                detection_start_time = time.time()
                face_detected = True
            else:
                elapsed_time = time.time() - detection_start_time
                if elapsed_time >= 3:
                    # Realizar la acción después de que la cara ha sido detectada durante 3 segundos
                    print("¡Cara detectada durante 3 segundos!")
                    last_face_seen_time = time.time()

                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, "Cara detectada", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            if face_detected:
                # Si la cara desaparece, verifica cuánto tiempo ha pasado
                if last_face_seen_time and time.time() - last_face_seen_time > 7:
                    # Si pasan más de 6 segundos sin ver una cara, resetea la detección
                    print("La cara ya no se detecta. Reiniciando detección...")
                    face_detected = False
                    detection_start_time = None
                    last_face_seen_time = None

        # Mostrar la imagen con las caras detectadas
        cv2.imshow('img', img)

        # Presionar 'q' para salir del bucle
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
