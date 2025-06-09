import cv2
import tempfile
from collections import defaultdict
import time
from detection import detect_vehicles_and_plates as dvp


def process_video(
    st, video_path, model, reader, confidence_threshold=0.5, process_every_n_frames=5
):
    """Process video for plate detection with progress tracking"""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Cannot open video file")
        return None, [], {}

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Setup video writer
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    # Initialize tracking
    all_detections = []
    frame_count = 0
    processed_count = 0
    vehicle_count = defaultdict(int)
    unique_plates = set()

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)

            if frame_count % process_every_n_frames == 0:
                processed_count += 1

                # Process frame
                annotated_frame, detections = dvp.detect_vehicles_and_plates(
                    st, frame, model, reader, confidence_threshold
                )

                # Add metadata to detections
                for detection in detections:
                    detection["frame_number"] = frame_count
                    detection["timestamp"] = frame_count / fps
                    vehicle_count[detection["vehicle_type"]] += 1
                    if detection["plate_text"] != "Not detected":
                        unique_plates.add(detection["plate_text"])

                all_detections.extend(detections)

                # Update status
                status_text.text(
                    f"Processing: {progress*100:.1f}% | Frame {frame_count}/{total_frames} | Detections: {len(all_detections)}"
                )
            else:
                annotated_frame = frame

            out.write(annotated_frame)

    except Exception as e:
        st.error(f"Error during processing: {e}")

    finally:
        cap.release()
        out.release()

    processing_time = time.time() - start_time

    # Statistics
    stats = {
        "total_frames": frame_count,
        "processed_frames": processed_count,
        "total_detections": len(all_detections),
        "unique_plates": len(unique_plates),
        "vehicle_types": dict(vehicle_count),
        "processing_time": processing_time,
        "duration": duration,
        "fps": fps,
    }

    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")

    return temp_output.name, all_detections, stats
