import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model from a pickle file
with open('model.pkl', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Better for video stream
    max_num_hands=2,          # Can adjust based on use-case
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Start capturing video
cv2.namedWindow("Frame")
vc = cv2.VideoCapture(0)

while True:
    ret, frame = vc.read()
    if not ret:
        print("Failed to capture video frame")
        break

    # Correct unpacking of frame dimensions
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the original frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Prepare data for prediction
            data_aux = []
            x_ = []
            y_ = []

            for landmark in hand_landmarks.landmark:
                data_aux.extend([landmark.x, landmark.y])  # Flatten the data into a single list
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Calculate bounding box coordinates and ensure integers
            x1, y1 = int(min(x_) * W), int(min(y_) * H)
            x2, y2 = int(max(x_) * W), int(max(y_) * H)

            # Ensure data is in the correct shape for prediction: 1 sample x number of features
            data_aux = np.array([data_aux])  # Wrap in an additional list to create a 2D array
            prediction = model.predict(data_aux)
            prediction_text = f'Prediction: {prediction[0]}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, prediction_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Cleanup
vc.release()
cv2.destroyAllWindows()
