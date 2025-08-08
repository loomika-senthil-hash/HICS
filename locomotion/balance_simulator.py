# locomotion/balance_simulator.py

def check_posture(pose_landmarks):
    """
    Analyzes body posture to detect instability.
    pose_landmarks: a dictionary with body part Y-values (height level)
    """

    left_shoulder_y = pose_landmarks.get("left_shoulder_y", 0)
    right_shoulder_y = pose_landmarks.get("right_shoulder_y", 0)
    left_hip_y = pose_landmarks.get("left_hip_y", 0)
    right_hip_y = pose_landmarks.get("right_hip_y", 0)

    # Difference in shoulders and hips height
    shoulder_diff = abs(left_shoulder_y - right_shoulder_y)
    hip_diff = abs(left_hip_y - right_hip_y)

    print(f"[DEBUG] Shoulder Diff: {shoulder_diff}, Hip Diff: {hip_diff}")

    if shoulder_diff > 30 or hip_diff > 30:
        print("⚠️ Posture unstable, initiating balance correction!")
        return False
    else:
        print("✅ Posture stable.")
        return True
def simulate_dummy_pose():
    # Try changing these values for fun!
    dummy_pose = {
        "left_shoulder_y": 200,
        "right_shoulder_y": 250,   # 50 diff -> unstable
        "left_hip_y": 400,
        "right_hip_y": 395         # 5 diff -> stable
    }

    check_posture(dummy_pose)

if __name__ == "__main__":
    simulate_dummy_pose()
