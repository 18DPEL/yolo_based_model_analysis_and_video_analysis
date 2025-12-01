import cv2
import time

# Open the default webcam (usually index 0)
cap = cv2.VideoCapture(r'rtsp://admin:Admin%40123@192.168.1.64:554/cam/realmonitor?channel=3&subtype=0')

# Define the codec and create VideoWriter object to save the video
# 'XVID' is the codec for .avi files. You can also use 'MJPG', 'DIVX', etc.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(r'day_4.mp4', fourcc, 20.0, (640, 480))

# Start the timer
start_time = time.time()
duration = 60  # Record for 60 seconds

# Loop to continuously capture frames from the webcam
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Write the frame to the video file
        out.write(frame)

        # Display the frame in a window
        cv2.imshow('Webcam', frame)

        # Check if the specified duration has passed
        if time.time() - start_time > duration:
            break
        
        # Press 'q' to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the webcam and file pointers
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
