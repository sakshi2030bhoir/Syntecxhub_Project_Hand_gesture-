# Hand Gesture Recognition — Webcam Demo

Real-time hand gesture recognition using MediaPipe's HandLandmarker,
with joint-angle-based (rotation and hand-size tolerant) gesture
classification and mapped media-key actions.

## How it works
1. **MediaPipe HandLandmarker** detects 21 hand landmarks per hand from your webcam feed.
2. For each of the four main fingers, the script measures the **joint angle** at
   the middle knuckle (between the base and tip segments). A straight finger
   reads close to 180°, a curled one reads much smaller — this is more reliable
   than comparing raw distances or y-coordinates, since it doesn't depend on
   your exact hand size or how far you are from the camera.
3. The thumb (which only has 2 real joints and moves sideways rather than
   curling) is handled separately via distance-from-wrist.
4. A short **stability filter** requires a gesture to show up in at least 4 of
   the last 6 frames before it's trusted, cutting down on flicker/false reads.
5. **pyautogui** simulates the matching media key press (with a cooldown so a
   held gesture doesn't spam the action).
6. The video window overlays the hand skeleton and the detected gesture name.

## Recognized gestures

| Gesture | How to make it | Action |
|---|---|---|
| ✊ Fist | All fingers curled in | Stop |
| ✋ Open Palm | All 5 fingers extended | Play / Pause |
| 👍 Thumbs Up | Only thumb extended, pointing up | Volume up |
| 👎 Thumbs Down | Only thumb extended, pointing down | Volume down |
| ✌️ Peace | Index + middle extended | Next track |
| 🤟 Rock On | Thumb + pinky extended | Previous track |
| 🤟 I Love You (ASL "ILY" sign) | Thumb + index + pinky extended | Recognized, no action wired up |
| ☝️ Pointing | Only index extended | Recognized, no action wired up |
| 👌 OK Sign | Thumb + index pinched together, other 3 extended | Recognized, no action wired up |

**6 gestures have real actions wired up** (Fist, Open Palm, Thumbs Up, Thumbs
Down, Peace, Rock On); 3 more are recognized and shown on-screen but have no
action yet — wire one up in `GESTURE_ACTIONS` at the top of `hand_gesture.py`
in one line.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python hand_gesture.py
```

**First run only:** the script automatically downloads a small MediaPipe
model file (`hand_landmarker.task`, a few MB) into the same folder — needs
internet once, then it's cached locally.

- Press **`q`** in the video window to quit.
- Press **`d`** to toggle a debug overlay showing the raw finger states
  (1 = extended, 0 = folded) for each detected hand.

## If a gesture still isn't being recognized well
- Keep your whole hand in frame, reasonably well lit, and hold the gesture
  steady for about a second (the stability filter needs a few consistent
  frames).
- Turn on debug mode (`d`) and watch which finger flips between 0/1 — if one
  finger consistently misreads for your hand, adjust `EXTEND_ANGLE_THRESHOLD`
  near the top of `hand_gesture.py` (try 145–165; lower = easier to count as
  "extended").
- This uses MediaPipe's newer **Tasks API** (`mediapipe.tasks.python.vision`),
  not the older `mp.solutions.hands` API removed in recent MediaPipe releases —
  make sure you're not mixing in code from an older tutorial that uses
  `mp.solutions.hands`.
