# Imports
import mediapipe as mp
import cv2
import pyautogui


# importing the hands landmarks solutions from mediapipe
mp_hands = mp.solutions.hands
# importing the drawing utils of mediapipe, this will help us to show the hand landmarks on the screen
mp_drawing = mp.solutions.drawing_utils

# current screen resolution width and height
screen_width, screen_height = pyautogui.size()

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

        # getting the frame width and height
        frame_height, frame_width, _ = frame.shape

        # opencv follows BGR format but mediapipe follows RGB format so convert it BGR to RBG
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip the frame by y-axis
        rgb_frame = cv2.flip(rgb_frame, 1)

        # process the frame and detect the hands
        output = hands.process(rgb_frame)
        
        # start_point = (0,0)
        # end_point = (50, 50)
        # color = (0,0,255)
        # thickness = 5
        # cv2.rectangle(img=rgb_frame, start_point=start_point, end_point=end_point, color=color, thickness=thickness)

        # if hand detected
        if output.multi_hand_landmarks:
            for hand_landmarks in output.multi_hand_landmarks:
                # show the hand landmarks
                mp_drawing.draw_landmarks(
                    rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # getting the landmarks of hand
                landmarks = hand_landmarks.landmark
                for id, landmark in enumerate(landmarks):
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)

                    if id == 8:
                        cv2.circle(img=rgb_frame, center=(x, y),
                                   radius=15, color=(255, 255, 0), thickness=2)
                        pyautogui.moveTo(x, y)
                    
         # convert the flipped frame back to BGR before displaying
        bgr_frame_flipped = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("AI Virtual Mouse", bgr_frame_flipped)
        if cv2.waitKey(1) == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
