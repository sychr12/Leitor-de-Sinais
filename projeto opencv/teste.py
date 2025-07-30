import cv2
import mediapipe as mp
import csv

# Inicializações
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False)
mp_draw = mp.solutions.drawing_utils

labels = ['A', 'B', 'C', 'D']  # Letras que vamos coletar
coleta_atual = 'A'            # Comece com a letra A
contador = 0

# CSV para salvar
arquivo = open('dados_mao.csv', mode='a', newline='')
escritor = csv.writer(arquivo)

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(img_rgb)

    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            dados = []
            for lm in hand_landmarks.landmark:
                dados.extend([lm.x, lm.y, lm.z])
            
            dados.append(coleta_atual)  # Adiciona o rótulo
            escritor.writerow(dados)
            contador += 1

    cv2.putText(frame, f"Coletando: {coleta_atual} ({contador})", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Coleta de Dados", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key in [ord(l) for l in labels]:
        coleta_atual = chr(key).upper()
        contador = 0

camera.release()
arquivo.close()
cv2.destroyAllWindows()
