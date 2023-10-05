# importing packages
import cv2
import mediapipe as mp

# used to convert protobuf message to a dictionary
from google.protobuf.json_format import MessageToDict

# building the model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=4)

# reading from webcam
webcam = cv2.VideoCapture(0)

while True:
    success, img = webcam.read()

    # flipping the image for model
    original_img=img.copy() # copying the original image
    img = cv2.flip(img, 1)

    # converting to RGB for model (model needs RGB img)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # passing the image to the model
    results = hands.process(RGB_img)

    # if there is any result (if any hand is detected)
    if results.multi_hand_landmarks:

        if len(results.multi_handedness) == 2: # if two hands exist in the image
            cv2.putText(original_img, 'Both Hands', (250, 56), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

        else: # if only one hand exists in the image
            for i in results.multi_handedness:
                label = MessageToDict(i)['classification'][0]['label']

                if label == 'Left':
                    cv2.putText(original_img, f'{label} Hand', (20, 56), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

                if label == 'Right':
                    cv2.putText(original_img, f'{label} Hand', (460, 56), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('image', original_img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()