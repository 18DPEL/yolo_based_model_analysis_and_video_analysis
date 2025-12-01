import cv2
import os
import numpy as np

# Create a folder to save the extracted frames
output_folder = R'D:\extract_frame_web\EXTRED_PACKED'
os.makedirs(output_folder, exist_ok=True)

def extract_unique_frames_from_video(video_path, threshold=30):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        return

    frame_count = 0  # Initialize frame counter
    last_frame = None  # To store the previous frame

    while True:
        success, frame = cap.read()  # Read each frame from the video
        if not success:
            break  # Exit loop when video ends

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

        # Update the last frame
        last_frame = frame

    cap.release()
    cv2.destroyAllWindows()

# Path to your video file
video_path = r"C:\Users\ayuba\Downloads\WLCCTVNVR_ch2_main_20250303170800_20250303182400.mp4"

# Call the function to extract unique frames
extract_unique_frames_from_video(video_path)
