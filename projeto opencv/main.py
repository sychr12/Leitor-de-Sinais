import cv2
import mediapipe as mp

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
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

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Cabeca", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            
            thumb_tip = hand_landmarks.landmark[4]
            index_pip = hand_landmarks.landmark[6]
            
            
            thumb_ip = hand_landmarks.landmark[3]
            index_finger_dip = hand_landmarks.landmark[7]
            index_finger_tip = hand_landmarks.landmark[8]
            
            middle_tip = hand_landmarks.landmark[12]
            middle_pip = hand_landmarks.landmark[10]

            ring_tip = hand_landmarks.landmark[16]
            ring_pip = hand_landmarks.landmark[14]

            pinky_tip = hand_landmarks.landmark[20]
            pinky_pip = hand_landmarks.landmark[18]

            if thumb_tip.y < thumb_ip.y and index_finger_tip.y < index_pip.y and ring_tip.y < ring_pip.y and pinky_tip.y < pinky_pip.y:
                cv2.putText(frame, 'Mao aberta', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                break
        
            if thumb_tip.y < index_pip.y:
                cv2.putText(frame, 'Polegar para cima', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                break
            if index_finger_tip.x > index_pip.x:
                cv2.putText(frame, 'faz o L', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                break

    cv2.imshow('Rosto', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

