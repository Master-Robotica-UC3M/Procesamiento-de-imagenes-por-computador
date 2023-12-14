import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Accessing the webcam
video = cv2.VideoCapture(0)  # Use 0 for your default webcam

# Load your image
image = cv2.imread('/home/gonecho/Documents/MasterRobotica/PIC_TrabajoFinal/PuzzleGame/apple.jpg')  # Replace 'path_to_your_image.jpg' with your image path
image = cv2.resize(image, (40, 40))  # Resize the image to 40x40 pixels

# Initial position of the image at the center of the frame
initial_pos_x = 320  # Change this value according to your webcam resolution
initial_pos_y = 240  # Change this value according to your webcam resolution

grabbed = False

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to get hand landmarks
    results = hands.process(image_rgb)

    # Draw hand landmarks on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 4:  # Thumb tip landmark ID is 4
                    thumb_x, thumb_y = cx, cy
                elif id == 8:  # Index finger tip landmark ID is 8
                    index_x, index_y = cx, cy

            # Calculate the distance between thumb tip and index finger tip
            dist = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)

            if dist < 50:  # Threshold distance to "grab" the image
                grabbed = True
            else:
                grabbed = False

            if grabbed:
                # Place the image at the index finger tip location
                x_offset = index_x - 20  # Half of the image width (40 / 2)
                y_offset = index_y - 20  # Half of the image height (40 / 2)

                # Ensure the image fits within the frame boundaries
                if x_offset >= 0 and y_offset >= 0 and x_offset + 40 <= w and y_offset + 40 <= h:
                    frame[y_offset:y_offset+40, x_offset:x_offset+40] = image

    # Show the frame with landmarks and image
    cv2.imshow('Hand with Image Overlay', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all windows
video.release()
cv2.destroyAllWindows()
