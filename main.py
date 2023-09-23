import time

from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import random

app = Flask(__name__)

# Initialize MediaPipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Randomly generate initial point coordinates
point_x = random.randint(0, 640)
point_y = 0

# Variables for moving the point
move_speed_x = 5
move_direction_x = 1
move_speed_y = 5
move_direction_y = 1

# Count variables
total_count = 0
catch_count = 0
move_speed_increase = 2

start_time = None
end_time = None
elapsed_time = 0
temp_time = 5

def generate_frames():
    global point_x, point_y, move_direction_x, move_direction_y, total_count, catch_count, move_speed_x,move_speed_y, move_speed_increase, start_time, end_time, elapsed_time, temp_time
    while True:
        # Read frame from webcam
        ret, frame = cap.read()

        # Flip the frame horizontally (comment out this line if you want the original orientation)
        frame = cv2.flip(frame, 1)

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe
        results = hands.process(image)

        # Check if any hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Initialize finger states
                finger_states = {
                    "thumb": False,
                    "index": False,
                    "middle": False,
                    "ring": False,
                    "pinky": False
                }

                # Check if thumb is extended
                if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[
                    mp_hands.HandLandmark.THUMB_IP].x:
                    finger_states["thumb"] = True

                # Check if index finger is extended
                if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
                    finger_states["index"] = True

                # Check if middle finger is extended
                if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
                    finger_states["middle"] = True

                # Check if ring finger is extended
                if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[
                    mp_hands.HandLandmark.RING_FINGER_PIP].y:
                    finger_states["ring"] = True

                # Check if pinky finger is extended
                if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[
                    mp_hands.HandLandmark.PINKY_PIP].y:
                    finger_states["pinky"] = True

                # Check if hand is in a fist
                if not any(finger_states.values()):
                    cv2.putText(frame, "Fist: Yes", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Fist: No", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw circles on landmarks 5, 9, 13, and 17
                for i, landmark in enumerate(hand_landmarks.landmark):
                    if i in [5, 7, 9, 11, 13, 15, 17]:
                        # Get the pixel coordinates of the landmark
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])

                        # Draw a circle on the landmark position
                        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

                        # Check if the point touches the landmark
                        if abs(x - point_x) < 20 and abs(y - point_y) < 20:
                            # Increase the catch count
                            if not any(finger_states.values()):
                                catch_count += 1
                                end_time = time.time()

                                elapsed_time = end_time - start_time
                                if elapsed_time < temp_time:
                                    temp_time = elapsed_time

                            # Reset the point to a random x coordinate at the top of the frame
                                point_x = random.randint(0, frame.shape[1])
                                point_y = 0

                            # Check if catch_count is a multiple of 7
                                if catch_count % 7 == 0:
                                # Increase the move speed by 4
                                    move_speed_y += move_speed_increase
                                    move_speed_x += move_speed_increase+1

        # Draw a large blue point
        cv2.circle(frame, (point_x, point_y), 20, (0, 255, 0), -1)

        # Check if the point reaches the bottom of the frame
        if point_y >= frame.shape[0]:
            # Reset the point to a random x coordinate and y = 0
            point_x = random.randint(0, frame.shape[1])
            point_y = 0

        # Check if the point passes y = 1 and increase the total count
        if point_y > 1 and (point_y - move_speed_y * move_direction_y) <= 1:
            total_count += 1
            start_time = time.time()

        # Display the total count and catch count on the frame
        cv2.putText(frame, "Total Count: " + str(total_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Catch Count: " + str(catch_count), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Elapsed Time: " + str(temp_time) + "s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0, 0, 255), 2)

        # Update the point coordinates
        point_x += move_speed_x * move_direction_x
        point_y += move_speed_y * move_direction_y

        # Change the move direction if the point reaches the boundary
        if point_x >= frame.shape[1] or point_x <= 0:
            move_direction_x *= -1

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as an HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
