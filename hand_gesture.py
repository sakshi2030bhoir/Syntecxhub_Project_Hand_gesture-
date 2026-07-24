"""
hand_gesture.py
================
Real-time hand gesture recognition demo.

Pipeline:
  Webcam frame -> MediaPipe HandLandmarker (21 landmarks, Tasks API)
  -> finger extended/folded state (joint-angle based, robust to hand
     size/distance from camera and to rotation)
  -> rule-based gesture classification -> short stability check
  -> mapped action (media key press) -> on-screen overlay

Uses the current MediaPipe "Tasks" API (mediapipe>=0.10). The older
`mp.solutions.hands` API was removed from recent mediapipe releases,
so this script downloads a small model file (hand_landmarker.task,
~ a few MB) the first time it runs and reuses it after that.

Run:
    python hand_gesture.py

Press 'q' in the video window to quit.
Press 'd' to toggle a debug overlay showing raw finger states + angles.
"""

import math
import os
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception:
    # pyautogui needs a display/screen; if it can't initialize we still
    # run the demo and just print what action *would* have fired.
    PYAUTOGUI_AVAILABLE = False

# =====================================================================
# 0. Model download (one-time)
# =====================================================================

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one-time, a few MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


# =====================================================================
# 1. Gesture -> action mapping
#    At least 5 gestures below have a real mapped action; a couple of
#    extra recognized gestures (including an ASL sign) are included
#    with no action, just to show they're detected — wire up an action
#    for any of them by editing this dict.
# =====================================================================

GESTURE_ACTIONS = {
    "Fist": "stop",              # 1
    "Open Palm": "playpause",    # 2
    "Thumbs Up": "volumeup",     # 3
    "Thumbs Down": "volumedown", # 4
    "Peace": "nexttrack",        # 5
    "Rock On": "prevtrack",      # 6 (bonus)
    "I Love You": None,          # ASL "ILY" sign — recognized, no action wired up
    "Pointing": None,
    "OK Sign": None,
}

ACTION_COOLDOWN_SECONDS = 1.2  # avoid re-firing the same action every frame
STABILITY_FRAMES = 6           # how many recent frames to look at
STABILITY_MIN_MATCH = 4        # ...and how many of those must agree before we trust it

# =====================================================================
# 2. Hand landmark indices + skeleton connections (21-point model)
# =====================================================================

FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
FINGER_MCPS = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

# Angle (degrees) at the middle joint of each finger between the
# "upper" and "lower" segments. Close to 180 = straight/extended,
# small = curled/folded. This is robust to hand size, distance from
# the camera, and rotation, since it's a pure joint-angle measurement
# rather than a size-dependent distance ratio.
EXTEND_ANGLE_THRESHOLD = 155

# How close the thumb tip and index tip must be (relative to hand size)
# to count as a pinch, for the OK sign.
OK_PINCH_RATIO = 0.35

# How far the thumb tip must have spread from the index knuckle
# (relative to hand size) to count as "extended" rather than tucked in.
THUMB_SPREAD_THRESHOLD = 0.65

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                    # palm base
]


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _angle_at(vertex, a, b):
    """Angle (degrees) at `vertex` between rays vertex->a and vertex->b."""
    ax, ay = a.x - vertex.x, a.y - vertex.y
    bx, by = b.x - vertex.x, b.y - vertex.y
    mag_a = math.hypot(ax, ay)
    mag_b = math.hypot(bx, by)
    if mag_a == 0 or mag_b == 0:
        return 180.0
    cos_angle = (ax * bx + ay * by) / (mag_a * mag_b)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def get_finger_states(landmarks):
    """Return {finger_name: 1 (extended) or 0 (folded)}.

    Uses the joint angle at each finger's middle knuckle (MCP-PIP-TIP
    for the four fingers). A straight finger has an angle near 180
    degrees; a curled finger has a much smaller angle. This is more
    reliable than a distance-ratio approach because it doesn't depend
    on hand size or exact distance from the camera.
    """
    states = {}

    # Four fingers: angle at the PIP joint between the MCP and TIP.
    for finger in ("index", "middle", "ring", "pinky"):
        mcp = landmarks[FINGER_MCPS[finger]]
        pip = landmarks[FINGER_PIPS[finger]]
        tip = landmarks[FINGER_TIPS[finger]]
        angle = _angle_at(pip, mcp, tip)
        states[finger] = int(angle > EXTEND_ANGLE_THRESHOLD)

    # Thumb: only 2 real joints below the tip, and it moves sideways/out
    # from the palm rather than curling like the others, so a knuckle
    # angle isn't a reliable signal for it. Instead, measure how far the
    # thumb tip has spread away from the base of the palm (using the
    # index knuckle as a palm-center reference), normalized by hand size.
    # When the thumb is tucked in (fist, pointing, peace, etc.) the tip
    # stays close to the palm; when it's extended (open palm, thumbs
    # up/down, rock on, I love you) it moves well away from it.
    wrist = landmarks[0]
    thumb_tip = landmarks[FINGER_TIPS["thumb"]]
    index_mcp = landmarks[FINGER_MCPS["index"]]
    hand_scale = _dist(wrist, landmarks[FINGER_MCPS["middle"]])
    thumb_spread = _dist(thumb_tip, index_mcp) / hand_scale if hand_scale > 0 else 0
    states["thumb"] = int(thumb_spread > THUMB_SPREAD_THRESHOLD)

    return states


def classify_gesture(states, landmarks):
    """Rule-based classifier: maps a finger-state pattern (+ a couple of
    extra geometric checks) to a gesture name."""
    up = [states["thumb"], states["index"], states["middle"], states["ring"], states["pinky"]]
    wrist = landmarks[0]

    # OK sign: thumb + index tips pinched together, other three fingers extended.
    hand_scale = _dist(wrist, landmarks[FINGER_MCPS["middle"]])
    if hand_scale > 0:
        pinch = _dist(landmarks[FINGER_TIPS["thumb"]], landmarks[FINGER_TIPS["index"]]) / hand_scale
        if pinch < OK_PINCH_RATIO and states["middle"] and states["ring"] and states["pinky"]:
            return "OK Sign"

    if up == [0, 0, 0, 0, 0]:
        return "Fist"
    if up == [1, 1, 1, 1, 1]:
        return "Open Palm"
    if up == [1, 0, 0, 0, 0]:
        thumb_tip = landmarks[FINGER_TIPS["thumb"]]
        return "Thumbs Up" if thumb_tip.y < wrist.y else "Thumbs Down"
    if up == [0, 1, 0, 0, 0]:
        return "Pointing"
    if up == [0, 1, 1, 0, 0]:
        return "Peace"
    if up == [1, 0, 0, 0, 1]:
        return "Rock On"
    if up == [1, 1, 0, 0, 1]:
        return "I Love You"  # ASL "ILY" sign: thumb, index, and pinky extended
    return None


# =====================================================================
# 3. Stability filter — require a gesture to show up in most of the
#    last few frames before trusting it, to cut down on flicker/false
#    reads from a single noisy frame.
# =====================================================================

class GestureStabilizer:
    def __init__(self, maxlen=STABILITY_FRAMES, min_match=STABILITY_MIN_MATCH):
        self.history = deque(maxlen=maxlen)
        self.min_match = min_match

    def update(self, raw_gesture):
        self.history.append(raw_gesture)
        if len(self.history) < self.min_match:
            return None
        counts = {}
        for g in self.history:
            if g is None:
                continue
            counts[g] = counts.get(g, 0) + 1
        if not counts:
            return None
        best, count = max(counts.items(), key=lambda kv: kv[1])
        return best if count >= self.min_match else None


# =====================================================================
# 4. Action trigger (with cooldown so it doesn't spam key presses)
# =====================================================================

_last_action_time = {}


def trigger_action(gesture):
    action = GESTURE_ACTIONS.get(gesture)
    if not action:
        return

    now = time.time()
    if now - _last_action_time.get(gesture, 0) < ACTION_COOLDOWN_SECONDS:
        return
    _last_action_time[gesture] = now

    if PYAUTOGUI_AVAILABLE:
        try:
            pyautogui.press(action)
        except Exception as e:
            print(f"[warn] could not trigger action '{action}': {e}")
    else:
        print(f"[demo] would trigger action: {action} (pyautogui unavailable)")


def draw_hand(frame, landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (255, 200, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    return points


# =====================================================================
# 5. Main loop
# =====================================================================

def main():
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_hand_presence_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check that it's connected and not in use by another app.")
        return

    stabilizers = [GestureStabilizer(), GestureStabilizer()]  # one per hand slot
    show_debug = False

    print("Hand Gesture Recognition running.")
    print("  Press 'q' to quit.")
    print("  Press 'd' to toggle the debug overlay (raw finger states).")
    print()
    print("Gestures with actions: Fist(stop), Open Palm(play/pause),")
    print("  Thumbs Up(volume up), Thumbs Down(volume down), Peace(next track),")
    print("  Rock On(previous track). Also recognized: Pointing, OK Sign, I Love You.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural "selfie" view
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    points = draw_hand(frame, hand_landmarks)

                    states = get_finger_states(hand_landmarks)
                    raw_gesture = classify_gesture(states, hand_landmarks)
                    stable_gesture = stabilizers[i % 2].update(raw_gesture)

                    if stable_gesture:
                        trigger_action(stable_gesture)
                        text_x, text_y = points[0][0], points[0][1] + 40  # near wrist
                        cv2.putText(
                            frame, stable_gesture, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
                        )

                    if show_debug:
                        debug_text = " ".join(f"{k}:{v}" for k, v in states.items())
                        cv2.putText(
                            frame, debug_text, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA
                        )
            else:
                for s in stabilizers:
                    s.update(None)

            cv2.imshow("Hand Gesture Recognition (press q to quit, d for debug)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_debug = not show_debug
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
