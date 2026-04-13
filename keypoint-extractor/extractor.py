import cv2
import mediapipe as mp
import numpy as np
import imageio
import json
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

video_name = 'pv na.mp4'  # include the extension
video_path = 'keypoint-extractor/videos/' + video_name

# Create directories if needed
temp_path = 'keypoint-extractor/temp/annotated_image/'
output_path = 'keypoint-extractor/output/'
os.makedirs(temp_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

cap = cv2.VideoCapture(video_path)

i = 0
annotated_frames = []
keypoints_dict = []

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Video ended or can't be read.")
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        if results.pose_landmarks is not None:
            annotated_pose_landmarks = {str(j): [lmk.x, lmk.y, lmk.z] for j, lmk in enumerate(results.pose_landmarks.landmark)}
            keypoints_dict.append(annotated_pose_landmarks)
            print(f"Pose detected for frame {i}")
        else:
            print(f"No pose detected for frame {i}")

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        frame_path = os.path.join(temp_path, f"{i}.png")
        cv2.imwrite(frame_path, image)
        annotated_frames.append(frame_path)
        print(f"Saved frame: {frame_path}")
        i += 1

        # Optional: Show preview
        # cv2.imshow('MediaPipe Pose', image)
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break

cap.release()
cv2.destroyAllWindows()

# Save GIF from frames
def frames_to_gif(frames, output_name, limit=100):
    images = []
    for i, frame in enumerate(frames):
        if os.path.exists(frame) and i % 5 == 0:  # take 1 in 5 frames
            img = imageio.imread(frame)
            img_resized = cv2.resize(img, (640, 360))  # optional: resize
            images.append(img_resized)
        if len(images) >= limit:
            break
    if not images:
        print("❌ No frames collected. Skipping GIF creation.")
        return
    gif_path = os.path.join(output_path, output_name + '.gif')
    imageio.mimsave(gif_path, images, fps=10)
    print(f"✅ GIF saved at {gif_path}")


frames_to_gif(annotated_frames, video_name.split('.')[0])

# Save keypoints as JSON
if keypoints_dict:
    json_path = os.path.join(output_path, video_name.split('.')[0] + '-keypoints.json')
    with open(json_path, 'w') as fp:
        json.dump(keypoints_dict, fp)
    print(f"✅ Keypoints saved to {json_path}")
else:
    print("❌ No keypoints detected in any frame. JSON not saved.")


#json_path = os.path.join(output_path, video_name.split('.')[0] + '-keypoints.json')
#with open(json_path, 'w') as fp:
 #   json.dump(keypoints_dict, fp)
#print(f"✅ Keypoints saved to {json_path}")
