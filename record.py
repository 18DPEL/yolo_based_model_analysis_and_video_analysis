# import cv2
# import threading
# import time
# from datetime import datetime
# import os

# os.environ["OPENCV_FFMPEG_DEBUG"] = "1"
# os.environ["OPENCV_FFMPEG_THREADS"] = "1"  # Restrict FFmpeg to single thread

# rtsp_url = "rtsp://admin:Admin%40123@192.168.1.64:554/cam/realmonitor?channel=3&subtype=0"

# width, height, fps = 1920, 1080, 30  # Replace with your stream properties
# stop_threads = False

# def record_video():
#     global stop_threads
#     cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Limit buffer size

#     if not cap.isOpened():
#         print("Error: Unable to open RTSP stream for recording.")
#         return

#     recording_start_time = time.time()
#     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_filename = f"recording_{current_time}.mp4"
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

#     print(f"Recording started: {output_filename}")

#     while not stop_threads:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Frame not received for recording. Retrying...")
#             continue

#         video_writer.write(frame)

#         if time.time() - recording_start_time >= 10 * 60:
#             video_writer.release()
#             recording_start_time = time.time()
#             current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#             output_filename = f"recording_{current_time}.mp4"
#             video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
#             print(f"New recording started: {output_filename}")

#     video_writer.release()
#     cap.release()

# def stream_video():
#     global stop_threads
#     cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Limit buffer size

#     if not cap.isOpened():
#         print("Error: Unable to open RTSP stream for streaming.")
#         return

#     while not stop_threads:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Frame not received for streaming. Retrying...")
#             continue

#         resized_frame = cv2.resize(frame, (940, 460))
#         cv2.imshow("RTSP Stream (Resized)", resized_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             stop_threads = True
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def main():
#     global stop_threads
#     try:
#         recording_thread = threading.Thread(target=record_video)
#         streaming_thread = threading.Thread(target=stream_video)

#         recording_thread.start()
#         streaming_thread.start()

#         recording_thread.join()
#         streaming_thread.join()

#     except Exception as e:
#         print(f"Error occurred: {e}")
#     finally:
#         stop_threads = True

# if __name__ == "__main__":
#     main()



import ffmpeg
import cv2
import numpy as np
import time

camera_source = 'rtsp://admin:Admin%40123@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0'

process = (
    ffmpeg
    .input(camera_source)
    .output('pipe:', format='rawvideo', pix_fmt='bgr24', s='640x480')
    .run_async(pipe_stdout=True, pipe_stderr=True)
)

while True:
    in_bytes = process.stdout.read(640,480,3)
    if not in_bytes:
        print("Error: No frame received")
        break

    frame = (
        np.frombuffer(in_bytes, np.uint8)
        .reshape((480, 640, 3))
    )

    cv2.imshow("FFmpeg Decoded Feed", frame)
    if cv2.waitKey(1) == ord('q'):
        break

process.terminate()
cv2.destroyAllWindows()

