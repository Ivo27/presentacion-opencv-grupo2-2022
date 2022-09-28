import cv2
import imutils
import os

clasificador_caras = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture("ivowebcam.mp4")

nombre_persona = "ivowebcam"
path_datos = "Imagenes"
path_persona = path_datos + "/" + nombre_persona

if not os.path.exists(path_persona):
    print("Carpeta Creada: ", path_persona)
    os.makedirs(path_persona)

count = 0

while True:

    ref, frame = cap.read()
    frame_achicado = imutils.resize(frame, height=640)
    frame_rotado_achicado = imutils.rotate(frame_achicado, 0)

    frame_auxiliar = frame_rotado_achicado.copy()
    
    gris = cv2.cvtColor(frame_rotado_achicado, cv2.COLOR_BGR2GRAY)


    caras = clasificador_caras.detectMultiScale(gris, 
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (100,100))

    for (x, y, w, h) in caras:
        cara = frame_auxiliar[y:y+h, x:x+w]
        cara = cv2.resize(cara, (150,150), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(frame_rotado_achicado, (x,y), (x + w, y + h), (0,255,0), 2)
        cv2.imwrite(f"Imagenes/{nombre_persona}/cara_{count}.jpg", cara)
        count += 1

    cv2.imshow(nombre_persona, frame_rotado_achicado)

    tecla = cv2.waitKey(1)
    if tecla == ord("q") or ref == False or count >= 300:
        break


cap.release()
cv2.destroyAllWindows()