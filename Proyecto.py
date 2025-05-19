import cv2
import numpy as np
import pyautogui
import keyboard
from datetime import datetime
import os

# dimensiones mínimas para que se considere X válida
TAM_MIN = 182

# rango del color verde (ajustable si es necesario)
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

print("🟢 Buscando X verde que supere 182x182... Presiona ESC para salir.")

def detecta_X_grande(contours):
    if len(contours) < 2:
        return False, None

    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            x1, y1, w1, h1 = cv2.boundingRect(contours[i])
            x2, y2, w2, h2 = cv2.boundingRect(contours[j])

            cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
            cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2

            distancia = np.hypot(cx2 - cx1, cy2 - cy1)

            if distancia < 250:
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2)
                y_max = max(y1 + h1, y2 + h2)

                w_total = x_max - x_min
                h_total = y_max - y_min

                if w_total >= TAM_MIN or h_total >= TAM_MIN:
                    return True, (x_min, y_min, w_total, h_total)

    return False, None

while True:
    try:
        if keyboard.is_pressed('esc'):
            print("⛔ ESC presionado. Saliendo del programa.")
            break

        screenshot = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hay_X, bbox = detecta_X_grande(contours)

        if hay_X and bbox is not None:
            x, y, w, h = bbox
            print("✅ X verde detectada. Guardando imagen...")

            # dibujar rectángulo
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # crear nombre con timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            nombre_archivo = f"X_detectada_{timestamp}.png"

            # guardar imagen en el mismo directorio
            ruta_actual = os.path.dirname(os.path.abspath(__file__))
            ruta_completa = os.path.join(ruta_actual, nombre_archivo)
            cv2.imwrite(ruta_completa, frame)

            print(f"📸 Imagen guardada como: {nombre_archivo}")
            break  # salimos después de detectar y guardar

    except Exception as e:
        print(f"⚠️ Error: {e}")
        break

cv2.destroyAllWindows()


######################################################################

# reconocimiento.py
import numpy as np
import cv2

nombres_clases = ["circulo", "X", "T"]
w = np.load("pesos_w.npy")
b = np.load("pesos_b.npy")

def extraer_caracteristicas(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.resize(gris, (50, 50))
    _, binaria = cv2.threshold(gris, 127, 1, cv2.THRESH_BINARY_INV)
    densidad = binaria.mean()
    simetria_v = np.abs(binaria[:, :25] - np.fliplr(binaria[:, 25:])).mean()
    return np.array([densidad, simetria_v])

# Ruta de la imagen que dibujaste en Paint
ruta = "./imagenes/pruebacirculo.png"
img = cv2.imread(ruta)

if img is not None:
    x = extraer_caracteristicas(img)
    scores = [np.dot(wi, x) + bi for wi, bi in zip(w, b)]
    pred = np.argmax(scores)
    etiqueta = nombres_clases[pred]

    # Mostrar el resultado sobre la imagen
    img_mostrar = cv2.resize(img, (300, 300))
    cv2.putText(img_mostrar, f"Figura: {etiqueta}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Resultado", img_mostrar)
    print(f"🔍 La imagen fue clasificada como: {etiqueta}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("⚠️ No se pudo leer la imagen de prueba. Asegúrate de guardarla como prueba.png en la carpeta /imagenes/")



##################################################################################################
# entrenamiento.py
import numpy as np
import cv2
import os

nombres_clases = ["circulo", "X", "T"]
personas = []
clases = []

for i, clase in enumerate(nombres_clases):
    for j in range(1, 4):  # SE AJUSTA SEGUN LAS IMAGENES QUE SE TENGAN POR CLASE, (CLASE: CIRCULO, X, T)
        ruta = f"./imagenes/pruebacirculo.png"
        if os.path.exists(ruta):
            img = cv2.imread(ruta)
            gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gris = cv2.resize(gris, (50, 50))
            _, binaria = cv2.threshold(gris, 127, 1, cv2.THRESH_BINARY_INV)

            densidad = binaria.mean()
            simetria_v = np.abs(binaria[:, :25] - np.fliplr(binaria[:, 25:])).mean()
            personas.append(np.array([densidad, simetria_v]))
            clases.append(i)
        else:
            print(f"⚠️ Imagen no encontrada: {ruta}")

personas = np.array(personas)
clases = np.array(clases)

num_clases = len(nombres_clases)
w = np.random.uniform(-1, 1, size=(num_clases, 2))
b = np.random.uniform(-1, 1, size=num_clases)

epocas = 100
tasa = 0.01

for epoca in range(epocas):
    for i in range(len(personas)):
        x = personas[i]
        y = clases[i]
        for c in range(num_clases):
            y_bin = 1 if y == c else 0
            pred = 1 if np.dot(w[c], x) + b[c] > 0 else 0
            error = y_bin - pred
            w[c] += tasa * error * x
            b[c] += tasa * error

np.save("pesos_w.npy", w)
np.save("pesos_b.npy", b)
print("✅ Entrenamiento completo. Pesos guardados.")
