import mediapipe as mp
import cv2

# importing the hands landmarks solutions from mediapipe
mp_hands = mp.solutions.hands
# importing the drawing utils of mediapipe, this will help us to show the hand landmarks on the screen
mp_drawing = mp.solutions.drawing_utils

# capture the camera
cam = cv2.VideoCapture(0)

# open the hand detection module with three parameters:
#   max_num_hand: track only given no. of hands on screen at a time
#   min_detection_confidence: show hand landmarks only when the confidence is more than 50%
#   min_tracking_confidence: track hands only when the confidence is more than 50%
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    # starts the camera
    while True:
        success, frame = cam.read()
        if not success:
            print("Cannot detect camera")
            break
        # opencv follows BGR format but mediapipe follows RGB format so convert it BGR to RBG
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Flip the frame by y-axis
        rgb_frame = cv2.flip(rgb_frame, 1)
        # process the frame and detect the hands
        output = hands.process(rgb_frame)

        # if hand detected
        if output.multi_hand_landmarks:
            for hand_landmarks in output.multi_hand_landmarks:
                # show the hand landmarks
                mp_drawing.draw_landmarks(
                    rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

         # convert the flipped frame back to BGR before displaying
        bgr_frame_flipped = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("AI Virtual Mouse", bgr_frame_flipped )
        if cv2.waitKey(1) == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()