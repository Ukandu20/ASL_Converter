import os
import cv2
import mediapipe as mp
import pickle
import numpy as np

# Disable oneDNN optimizations to ensure consistent behavior across different platforms
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize MediaPipe Hands with specific parameters
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

data_dir = 'assets/ASL_Alphabet_Dataset/asl_alphabet_train'
data = []
labels = []

# Process each image in the dataset
with mp_hands as hands:
    for dir_name in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue  # Skip files that are not directories

        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image {img_path}")
                continue  # Skip processing this image if it cannot be read

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    data_aux = []
                    for landmark in hand_landmarks.landmark:
                        data_aux.extend([landmark.x, landmark.y])
                    data.append(data_aux)
                    labels.append(dir_name)

# Determine the maximum number of features in any sample
max_features = max(len(d) for d in data)

# Pad data to ensure all data points have the same number of features
padded_data = np.zeros((len(data), max_features))
for i, sample in enumerate(data):
    padded_data[i, :len(sample)] = sample

# Save the padded data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': padded_data, 'labels': labels}, f)
    print("Padded data and labels have been saved to 'data.pickle'.")
