import cv2 
import os 
import numpy as np

path_datos = "Imagenes"
lista_personas = [x for x in os.listdir(path_datos) if not os.path.isfile(x)]
print("Lista de personas: ", lista_personas)

labels = []
data_caras = []
label = 0

for nombre in lista_personas:
    path_persona = path_datos + "/" + nombre
    print("Leyendo imagenes")

    lista_archivos = os.listdir(path_persona)
    lista_archivos.pop()
    for archivo in lista_archivos:
        print("Caras: ", nombre + "/" + archivo)
        labels.append(label)
        data_caras.append(cv2.imread(path_persona+"/"+archivo,0))
        imagen = cv2.imread(path_persona+"/"+archivo,0)
        cv2.imshow("imagen", imagen)
        cv2.waitKey(10)
    label += 1

#print("Labels: ", labels)

reconocedor_caras = cv2.face.EigenFaceRecognizer_create()

print("Entrenando...")
reconocedor_caras.train(data_caras, np.array(labels))

reconocedor_caras.write("modeloEigenFace.xml")
print("Modelo almacenado")