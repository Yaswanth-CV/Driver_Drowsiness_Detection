import argparse
import time
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import mediapipe as mp
from math import hypot

EYE_MODEL_PATH = "eye_model.tflite"

CNN_THRESHOLD = 0.70       
EAR_THRESHOLD = 0.21      
EYE_CLOSED_MAX_FRAMES = 20

def load_tflite(path):
    interpreter = Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

def tflite_predict(interpreter, input_details, output_details, img):
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])
    return float(out[0][0])

def compute_EAR(landmarks, left=True):
    if left:
        idx = [33, 160, 158, 133, 153, 144]
    else:
        idx = [362, 385, 387, 263, 373, 380]

    def dist(a, b): return hypot(a.x - b.x, a.y - b.y)

    A = dist(landmarks[idx[1]], landmarks[idx[5]])
    B = dist(landmarks[idx[2]], landmarks[idx[4]])
    C = dist(landmarks[idx[0]], landmarks[idx[3]])

    return (A + B) / (2.0 * C)

def crop_eye(frame_rgb, lm, left=True):
    if left:
        c1, c2 = 33, 133
    else:
        c1, c2 = 362, 263

    h, w = frame_rgb.shape[:2]

    x1 = int(lm[c1].x * w) - 20
    y1 = int(lm[c1].y * h) - 20
    x2 = int(lm[c2].x * w) + 20
    y2 = int(lm[c2].y * h) + 20

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    eye = frame_rgb[y1:y2, x1:x2]
    return eye

def main(show_boxes=False):

    # Mediapipe
    mp_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    # Eye CNN
    eye_interpreter, eye_in, eye_out = load_tflite(EYE_MODEL_PATH)

    print(" Loaded Eye TFLite CNN + EAR")

    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    eye_closed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh = mp_mesh.process(frame_rgb)

        cnn_left = cnn_right = 0
        EAR = 0
        is_closed = False

        if mesh.multi_face_landmarks:
            lm = mesh.multi_face_landmarks[0].landmark

            #  EAR 
            EAR_L = compute_EAR(lm, left=True)
            EAR_R = compute_EAR(lm, left=False)
            EAR = (EAR_L + EAR_R) / 2

            #  BOTH EYES CNN
            # Left eye
            LE = crop_eye(frame_rgb, lm, left=True)
            if LE.size > 0:
                LE = cv2.resize(LE, (128, 128)).astype(np.float32) / 255.0
                LE = np.expand_dims(LE, 0)
                cnn_left = tflite_predict(eye_interpreter, eye_in, eye_out, LE)

            # Right eye
            RE = crop_eye(frame_rgb, lm, left=False)
            if RE.size > 0:
                RE = cv2.resize(RE, (128, 128)).astype(np.float32) / 255.0
                RE = np.expand_dims(RE, 0)
                cnn_right = tflite_predict(eye_interpreter, eye_in, eye_out, RE)

            cnn_prob = (cnn_left + cnn_right) / 2

            #  FUSION 
            is_closed = (cnn_prob > CNN_THRESHOLD) or (EAR < EAR_THRESHOLD)

            if is_closed:
                eye_closed_frames += 1
            else:
                eye_closed_frames = 0

        # Final state
        is_drowsy = (eye_closed_frames >= EYE_CLOSED_MAX_FRAMES)

        status = "DROWSY" if is_drowsy else "ALERT"
        color = (0,0,255) if is_drowsy else (0,255,0)

        #  Display Text 
        cv2.putText(frame, f"Status: {status}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.putText(frame, f"CNN_L={cnn_left:.2f}  CNN_R={cnn_right:.2f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, f"EAR={EAR:.3f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, f"Frames Closed={eye_closed_frames}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

        # FPS
        # now = time.time()
        # fps = 1 / (now - prev_time); prev_time = now
        # cv2.putText(frame, f"FPS={fps:.1f}", (20, 180),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Drowsiness - CNN + EAR", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_boxes", action="store_true")
    args = parser.parse_args()
    main(show_boxes=args.show_boxes)
