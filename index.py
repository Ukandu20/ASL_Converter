import cv2
import numpy as np

# Initialize background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

def detect_hand(frame):
    # Apply background subtraction
    fgMask = backSub.apply(frame)

    # Edge detection
    edges = cv2.Canny(fgMask, 100, 200)

    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on shape properties
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Minimum area to consider
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
                if solidity > 0.8:  # Solidity check to filter out non-solid shapes
                    valid_contours.append(contour)

    # Find the largest valid contour as the hand
    if valid_contours:
        hand_contour = max(valid_contours, key=cv2.contourArea)
        return hand_contour, edges

    return None, edges

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    hand_contour, edge_view = detect_hand(frame)
    if hand_contour is not None:
        # Draw the contour on the frame
        cv2.drawContours(frame, [hand_contour], -1, (255, 0, 0), 2)

    cv2.imshow('Camera Feed', frame)
    cv2.imshow('Edges', edge_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
