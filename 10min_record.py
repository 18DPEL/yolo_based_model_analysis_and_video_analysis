import cv2
import time

# Define the RTSP stream URL and output video file
rtsp_url = "rtsp://admin:Admin%40123@192.168.1.64:554"
output_file = "output_video.mp4"

# Set the duration for recording in seconds (10 minutes = 600 seconds)
record_duration = 120000

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the stream opened successfully
if not cap.isOpened():
    print("Failed to open the RTSP stream.")
    exit()

# Get the frame width, height, and FPS from the stream
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if fps == 0:  # Some RTSP streams may not report FPS correctly
    fps = 25  # Set a default FPS

# Define the codec and create the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 files
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Start the timer
start_time = time.time()

print("Recording started...")

# Read and save frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame. Exiting...")
        break

    # Write the frame to the output file
    # out.write(frame)
    frame=cv2.resize(frame,(860,440))
    # Show the frame (optional, can be commented out)
    cv2.imshow("RTSP Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Check if the recording duration is reached
    if time.time() - start_time > record_duration:
        print("10 minutes reached. Stopping recording...")
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Recording saved as", output_file)
