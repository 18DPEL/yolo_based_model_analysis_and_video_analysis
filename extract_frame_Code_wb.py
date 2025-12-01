import cv2
import os
import numpy as np

# Create a folder to save the extracted frames
output_folder = r'E:\Railway Project/Extracted_frame'
os.makedirs(output_folder, exist_ok=True)

def extract_unique_frames_from_webcam(threshold=30):
    # Open the webcam (0 is usually the default webcam)
    cap = cv2.VideoCapture(r"rtsp://admin:Admin%40123@192.168.1.64:554/cam/realmonitor?channel=3&subtype=0")

    if not cap.isOpened():
        print("Error: Unable to open the webcam.")
        return

    frame_count = 0  # Initialize frame counter
    last_frame = None  # To store the previous frame

    while True:
        success, frame = cap.read()  # Read each frame from the webcam
        frame=cv2.resize(frame,(1500,750))
        if not success:
            print("Error: Failed to capture frame from webcam.")
            break

        if last_frame is not None:
            # Calculate the absolute difference between the current frame and the last frame
            diff = cv2.absdiff(last_frame, frame)

            # Convert the difference to grayscale
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # Calculate the sum of the pixel differences
            non_zero_count = np.sum(gray_diff > threshold)

            # If the number of different pixels exceeds a threshold, save the frame
            if non_zero_count > 0:
                image_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')
                cv2.imwrite(image_path, frame)
                print(f'Saved: {image_path}')
                frame_count += 1

        # Display the current frame in a window
        cv2.imshow('Webcam', frame)

        # Update the last frame
        last_frame = frame

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to extract unique frames from the webcam
extract_unique_frames_from_webcam()
