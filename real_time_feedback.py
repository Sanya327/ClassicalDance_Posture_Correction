import cv2
import mediapipe as mp
import torch
import numpy as np
import json
from model import PoseLSTM
from pose_comparison import compare_frame, get_model_score  # Import the comparison functions

mp_pose = mp.solutions.pose

# Load the trained LSTM model
model = PoseLSTM()
model.load_state_dict(torch.load("pose_lstm_model.pth"))  # Load trained model
model.eval()

# Ground truth data (for comparison) - You would load this from a file (e.g., a previously saved json)
ground_truth_file = 'keypoint-extractor/output/pv na-keypoints.json'

def load_ground_truth(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

ground_truth_data = load_ground_truth(ground_truth_file)

def extract_keypoints_from_frame(frame, pose_model):
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = pose_model.process(image)
    if results.pose_landmarks:
        return [lm.x for lm in results.pose_landmarks.landmark] + \
               [lm.y for lm in results.pose_landmarks.landmark] + \
               [lm.z for lm in results.pose_landmarks.landmark]
    return None

def main():
    cap = cv2.VideoCapture(0)
    keypoints_seq = []
    window_size = 30
    frame_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            kp = extract_keypoints_from_frame(frame, pose)
            if kp:
                keypoints_seq.append(kp)
                if len(keypoints_seq) > window_size:
                    keypoints_seq.pop(0)

                if len(keypoints_seq) == window_size:
                    # Get score from LSTM model for the keypoints sequence
                    seq_tensor = torch.tensor([keypoints_seq], dtype=torch.float32)
                    with torch.no_grad():
                        score = model(seq_tensor).item()

                    # Get the ground truth frame from the pre-extracted data (for comparison)
                    ground_truth_frame = ground_truth_data[frame_count] if frame_count < len(ground_truth_data) else None
                    if ground_truth_frame:
                        comparison_score = compare_frame(ground_truth_frame, dict(enumerate(kp)))  # Compare ground truth with user pose

                        # Combine model score and L2 comparison score for feedback
                        final_score = (score + (1 - comparison_score)) / 2  # Just an example of how you could combine these
                        cv2.putText(frame, f"Score: {round(final_score, 2)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Show live feedback
            cv2.imshow("Dance Feedback", frame)

            # Exit if 'Esc' key is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
