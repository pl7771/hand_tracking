import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)


mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      model_complexity=1,
                      min_detection_confidence=0.6,
                      min_tracking_confidence=0.6)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flippedImage = cv2.flip(imgRGB, 1)

    results = hands.process(flippedImage)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for marks in results.multi_hand_landmarks:
            for id, lm in enumerate(marks.landmark):
                #print(id, lm)
                h, w, c = flippedImage.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 4:
                    cv2.circle(flippedImage, (cx, cy), 11, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(flippedImage, marks, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(flippedImage, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 2)

    cv2.imshow("Image", flippedImage)
    cv2.waitKey(1)