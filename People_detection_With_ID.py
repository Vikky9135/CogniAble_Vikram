import cv2
from ultralytics import YOLO
import torch
import numpy as np
import yt_dlp
from deep_sort_realtime.deepsort_tracker import DeepSort

def download_video(youtube_url, output_path):
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

# YouTube video URL and paths
youtube_url = 'https://www.youtube.com/watch?v=GNVTuLHdeSo'
input_video_path = 'input_video.mp4'
output_video_path = 'output_video.mp4'

try:
    download_video(youtube_url, input_video_path)
except Exception as e:
    print(f"Error downloading video: {e}")
    exit(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO('yolov8x.pt')
model.to(device)

deepsort = DeepSort(
    max_age=30,
    n_init=5,  # Increased to improve initial track creation
    nms_max_overlap=0.7,
    max_cosine_distance=0.2,  # Decreased for stricter similarity checks
    embedder="mobilenet",
    embedder_gpu=True
)

video_capture = cv2.VideoCapture(input_video_path)
if not video_capture.isOpened():
    print(f"Error: Unable to open video source: {input_video_path}")
    exit(1)

fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

def filter_detections(detections, min_confidence=0.5):
    return [det for det in detections if det[1] >= min_confidence]

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Finished processing the video")
        break

    frame_count += 1
    if frame_count % 30 == 0:
        progress = (frame_count / total_frames) * 100
        print(f"Processing: {progress:.2f}% complete")

    results = model(frame, stream=True, device=device)
    detections = []

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf)
            class_id = int(box.cls)
            if class_id == 0:
                detections.append(([x1, y1, x2, y2], confidence, 'person'))

    filtered_detections = filter_detections(detections, min_confidence=0.5)
    tracks = deepsort.update_tracks(filtered_detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        x1, y1, x2, y2 = map(int, track.to_tlbr())
        obj_id = track.track_id
        color = (255, 0, 0)
        label_color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

        confidence = track.get_det_conf()
        if confidence is not None:
            label = f'Human: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)

    video_writer.write(frame)
    cv2.imshow('YOLO Human Detection with Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
video_writer.release()
cv2.destroyAllWindows()
