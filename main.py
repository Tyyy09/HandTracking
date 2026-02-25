import os
import time
import random
import cv2

from groq import Groq
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

# ---------------- LLM CONFIG (GROQ) ----------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- COLORS FOR OBJECTS ----------------
COLOR_MAP = {}


def get_color_for_label(label):
    """Assign a consistent random color to each object label."""
    if label not in COLOR_MAP:
        COLOR_MAP[label] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
    return COLOR_MAP[label]


# ---------------- TEXT WRAP ----------------
def wrap_text(text, max_width, font, scale, thickness):
    """Wrap text into multiple lines based on pixel width."""
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test_line = current + " " + word if current else word
        (w, _), _ = cv2.getTextSize(test_line, font, scale, thickness)

        if w <= max_width:
            current = test_line
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def describe_scene_with_llm(objects, hand_centers):
    if objects:
        obj_descriptions = [
            f"{label} at ({x1},{y1},{x2},{y2})"
            for label, (x1, y1, x2, y2) in objects
        ]
        obj_text = "; ".join(obj_descriptions)
    else:
        obj_text = "none"

    if hand_centers:
        hand_descriptions = [
            f"hand {i+1} at ({hx},{hy})"
            for i, (hx, hy) in enumerate(hand_centers)
        ]
        hand_text = "; ".join(hand_descriptions)
    else:
        hand_text = "none"

    prompt = f"""
Objects detected:
{obj_text}

Hands detected:
{hand_text}

Describe in ONE short sentence what the person is doing.
"""

    response = client.chat.completions.create(
        model="allam-2-7b",
        messages=[
            {"role": "system", "content": "You describe visual scenes succinctly."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()


# ---------------- YOLO MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- MEDIAPIPE HANDS ----------------
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

hand_detector = HandLandmarker.create_from_options(options)


# ---------------- HELPERS ----------------
def get_hand_center(landmarks, w, h):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Hand + Object + Groq LLM reasoning running...")

last_sentence = "Analyzing scene..."
last_llm_time = 0
llm_interval = 1.5  # seconds between LLM calls

# Normal window
cv2.namedWindow("Hand + Object + Groq LLM", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand + Object + Groq LLM", 960, 720)

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 0.7
THICKNESS = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    h, w, _ = frame.shape

    # -------- YOLO OBJECT DETECTION --------
    objects = []
    for r in model(frame, stream=True):
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            objects.append((label, (x1, y1, x2, y2)))

            color = get_color_for_label(label)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, y1 - 5),
                        FONT, 0.6, color, 2)

    # -------- MEDIAPIPE HAND TRACKING --------
    mp_image = Image(
        image_format=ImageFormat.SRGB,
        data=frame
    )

    result = hand_detector.detect(mp_image)

    hand_centers = []
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            hx, hy = get_hand_center(hand_landmarks, w, h)
            hand_centers.append((hx, hy))
            cv2.circle(display, (hx, hy), 6, (0, 255, 0), -1)

    # -------- LLM REASONING (THROTTLED) --------
    now = time.time()
    if now - last_llm_time > llm_interval:
        last_sentence = describe_scene_with_llm(objects, hand_centers)
        last_llm_time = now

    # -------- TEXT WRAP + DISPLAY --------
    max_text_width = int(w * 0.9)
    lines = wrap_text(last_sentence, max_text_width, FONT, SCALE, THICKNESS)

    y_offset = 30
    for line in lines:
        cv2.putText(display, line, (10, y_offset),
                    FONT, SCALE, (0, 255, 255), THICKNESS)
        y_offset += 30

    cv2.imshow("Hand + Object + Groq LLM", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
