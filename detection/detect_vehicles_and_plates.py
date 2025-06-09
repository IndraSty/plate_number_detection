import cv2

def detect_vehicles_and_plates(st, frame, model, reader, confidence_threshold=0.5):
    """
    Detect vehicles and license plates in a frame
    """
    detections = []
    
    try:
        # YOLO detection
        results = model(frame, conf=confidence_threshold)
        
        # Vehicle classes in COCO dataset
        vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls in vehicle_classes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        vehicle_type = vehicle_classes[cls]
                        
                        # Draw vehicle bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f'{vehicle_type}: {confidence:.2f}', 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Extract vehicle region for plate detection
                        vehicle_roi = frame[y1:y2, x1:x2]
                        
                        # OCR on vehicle region
                        plate_text = "Not detected"
                        plate_confidence = 0.0
                        
                        try:
                            ocr_results = reader.readtext(vehicle_roi)
                            if ocr_results:
                                # Find the most confident text detection
                                best_detection = max(ocr_results, key=lambda x: x[2])
                                if best_detection[2] > 0.5:  # Confidence threshold for OCR
                                    plate_text = best_detection[1].strip()
                                    plate_confidence = best_detection[2]
                                    
                                    # Draw plate text
                                    cv2.putText(annotated_frame, f'Plate: {plate_text}', 
                                              (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        except:
                            pass
                        
                        # Store detection
                        detection = {
                            'vehicle_type': vehicle_type,
                            'vehicle_confidence': confidence,
                            'plate_text': plate_text,
                            'plate_confidence': plate_confidence,
                            'bbox': [x1, y1, x2, y2]
                        }
                        detections.append(detection)
        
        return annotated_frame, detections
    
    except Exception as e:
        st.error(f"Error in detection: {e}")
        return frame, []