import cv2
from ultralytics import YOLO

def run_detection(source = 0, model_path='yolov8n.pt'):
    # Load the YOLO model
    model = YOLO(model_path)

    # Start video capture
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform detection
        results = model(frame)

        # Draw results on the frame
        annotated_frame = results[0].plot()

        # Display the frame with detections
        cv2.imshow('YOLO Detection', annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()