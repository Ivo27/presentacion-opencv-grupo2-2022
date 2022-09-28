import cv2
import imutils
import os


path_datos = "Imagenes"
lista_personas = [x for x in os.listdir(path_datos) if not os.path.isfile(x)]
print("Lista de personas: ", lista_personas)


clasificador_caras = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

reconocedor_caras = cv2.face.EigenFaceRecognizer_create()
reconocedor_caras.read("modeloEigenFace.xml")

while True:
    ref, frame = cap.read()

    #frame = imutils.resize(frame, height=640)
    #frame = imutils.rotate(frame, 180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    caras = clasificador_caras.detectMultiScale(gray, 
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (100,100))

    for (x, y, w, h) in caras:
        cara = frame[y:y+h, x:x+w]
        cara_gris = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)
        cara_gris_reducida = cv2.resize(cara_gris, (150,150), interpolation=cv2.INTER_CUBIC)
        resultado = reconocedor_caras.predict(cara_gris_reducida)

        if resultado[1] < 7000:
            cv2.putText(frame, f"{lista_personas[resultado[0]]}", (x,y-5), 1,1.3,(255,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0), 2)
        else:
            cv2.putText(frame, "No se reconoce", (x,y-5), 1,1.3,(255,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0,0,255), 2)

    cv2.imshow("Prediccion", frame)

    tecla = cv2.waitKey(1)
    if tecla == ord("q") or ref == False:
        break


cap.release()
cv2.destroyAllWindows()