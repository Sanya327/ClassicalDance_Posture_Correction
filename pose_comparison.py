import numpy as np
import torch
from model import PoseLSTM  # Import the model

def l2_norm(a, b):
    return np.linalg.norm(a - b)

def compare_frame(gt_frame, user_frame):
    '''
    Compares a single frame of keypoints between the ground truth and the user's pose.
    Returns a comparison score.
    '''
    gt = np.array([gt_frame[str(i)][:2] for i in range(33)])  # Get x, y for all keypoints
    user = np.array([user_frame[str(i)][:2] for i in range(33)])
    return l2_norm(gt, user)

# Define a function to compute the model score for a sequence of keypoints
def get_model_score(keypoints_seq, model):
    # Convert keypoints sequence into tensor
    seq_tensor = torch.tensor([keypoints_seq], dtype=torch.float32)
    with torch.no_grad():
        score = model(seq_tensor).item()  # Get score from the model
    return score
