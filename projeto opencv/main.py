import cv2
import mediapipe as mp

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode = True, #frame da imagem
    max_num_hands = 1, # limitação (agora ta pra uma mão so)
    min_detection_confidence = 0.7 #Rigor de dectacção
)
mp_draw = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mao = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    results = hands.process(mao)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            frame, "Cabeca", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2
        )

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist = hand_landmarks.landmark[0]
            thumb_cmc = hand_landmarks.landmark[1]
            thumb_mcp = hand_landmarks.landmark[2]
            thumb_ip = hand_landmarks.landmark[3]
            thumb_tip = hand_landmarks.landmark[4]
            index_finger_mcp = hand_landmarks.landmark[5]
            index_finger_pip = hand_landmarks.landmark[6]
            index_finger_dip = hand_landmarks.landmark[7]
            index_finger_tip = hand_landmarks.landmark[8]
            middle_finger_mcp = hand_landmarks.landmark[9]
            middle_finger_pip = hand_landmarks.landmark[10]
            middle_finger_dip = hand_landmarks.landmark[11]
            middle_finger_tip = hand_landmarks.landmark[12]
            ring_finger_mcp = hand_landmarks.landmark[13]
            ring_finger_pip = hand_landmarks.landmark[14]
            ring_finger_dip = hand_landmarks.landmark[15]
            ring_finger_tip = hand_landmarks.landmark[16]
            pinky_mcp = hand_landmarks.landmark[17]
            pinky_pip = hand_landmarks.landmark[18]
            pinky_dip = hand_landmarks.landmark[19]
            pinky_tip = hand_landmarks.landmark[20]

            # Letra A
            if (
                thumb_tip.x < thumb_ip.x
                and index_finger_tip.y > index_finger_pip.y
                and middle_finger_tip.y > middle_finger_pip.y
                and ring_finger_tip.y > ring_finger_pip.y
                and pinky_tip.y > pinky_pip.y
            ):
                cv2.putText(frame, "A", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break

            # Letra B
            if (
                index_finger_tip.y < index_finger_pip.y
                and middle_finger_tip.y < middle_finger_pip.y
                and ring_finger_tip.y < ring_finger_pip.y
                and pinky_tip.y < pinky_pip.y
                and thumb_tip.x < index_finger_mcp.x
            ):
                cv2.putText(frame, "B", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break

            # Letra C
            if (
                index_finger_tip.x > index_finger_pip.x
                and middle_finger_dip.x > middle_finger_pip.x
                and ring_finger_tip.x > ring_finger_pip.x
                and pinky_tip.x > pinky_pip.x
                and thumb_tip.y < thumb_mcp.y
            ):
                cv2.putText(frame, "C", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break

            # Letra D
            if (
                pinky_tip.y > pinky_dip.y
                and ring_finger_tip.y > ring_finger_dip.y
                and middle_finger_tip.y > middle_finger_dip.y
                and thumb_tip.x < middle_finger_dip.x
            ):
                cv2.putText(frame, "D", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break

            # Letra E
            if (
                index_finger_tip.y > index_finger_pip.y
                and middle_finger_tip.y > middle_finger_pip.y
                and ring_finger_tip.y > ring_finger_pip.y
                and pinky_tip.y > pinky_pip.y
                and thumb_ip.x > index_finger_dip.x
            ):
                cv2.putText(frame, "E", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break

    cv2.imshow("Rosto", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
