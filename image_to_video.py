import cv2
import os
import re

def convert_images_to_video(image_folder, video_path, frame_rate):
    # Get a list of image files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    # Sort images by their numeric part in the filename (e.g., image1.jpg -> 1)
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(0)) if match else 0

    images.sort(key=extract_number)
    print(images)

    # Check if there are any images
    if not images:
        print("No images found in the specified folder.")
        return

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error reading the first image: {first_image_path}")
        return

    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi format
    video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

    # Write each image to the video file
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error reading image: {image_path}")
            continue

        # Check if the image has the same dimensions as the first image
        if frame.shape[0] != height or frame.shape[1] != width:
            print(f"Image dimensions mismatch: {image_path}")
            continue

        video.write(frame)

    # Release the VideoWriter object
    video.release()
    print(f"Video saved to {video_path}")

# Example usage
image_folder = r'D:\extract_frame_web\extracted_frames'  # Replace with your image folder path
video_path = r'D:\extract_frame_web\out3.mp4'  # Replace with your desired video path
frame_rate = 20  # Set the desired frame rate (standard values are 24, 30, or 60)

convert_images_to_video(image_folder, video_path, frame_rate)
