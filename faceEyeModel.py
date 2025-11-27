import argparse
import time
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import mediapipe as mp
from math import hypot
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


FACE_MODEL_PATH = "face_model.tflite"
EYE_MODEL_PATH  = "eye_model.tflite"

CNN_FACE_THRESHOLD = 0.50     
CNN_EYE_THRESHOLD  = 0.60     

EAR_THRESHOLD = 0.21          

FACE_WEIGHT = 0.40
EYE_WEIGHT  = 0.40
EAR_WEIGHT  = 0.20

EYE_CLOSED_MAX_FRAMES = 10


# Load TFLite Model
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
        idx = [33,160,158,133,153,144]
    else:
        idx = [362,385,387,263,373,380]

    def dist(a,b): return hypot(a.x-b.x, a.y-b.y)

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

    return frame_rgb[y1:y2, x1:x2]


def main(show_boxes=False):

    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    mp_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    face_interpreter, face_in, face_out = load_tflite(FACE_MODEL_PATH)
    eye_interpreter, eye_in, eye_out = load_tflite(EYE_MODEL_PATH)

    print("✅ Loaded Face CNN + Eye CNN (Both) + EAR")

    # Camera
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    eye_closed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # FACE CNN (Drowsy / Alert)
        face_prob = 0
        face_res = mp_face.process(frame_rgb)

        if face_res.detections:
            det = face_res.detections[0]
            box = det.location_data.relative_bounding_box

            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)
            x2 = int((box.xmin + box.width) * w)
            y2 = int((box.ymin + box.height) * h)

            x1, y1 = max(0,x1), max(0,y1)
            x2, y2 = min(w,x2), min(h,y2)

            face_roi = frame_rgb[y1:y2, x1:x2]
            if face_roi.size > 0:
                face_img = cv2.resize(face_roi, (128,128))
                face_img = preprocess_input(face_img.astype(np.float32))
                face_img = np.expand_dims(face_img, 0)
                face_prob = tflite_predict(face_interpreter, face_in, face_out, face_img)

            if show_boxes:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)


        # EYES (CNN LEFT + CNN RIGHT + EAR)
        eye_prob_L = 0
        eye_prob_R = 0
        EAR_val = 0

        mesh = mp_mesh.process(frame_rgb)
        if mesh.multi_face_landmarks:
            lm = mesh.multi_face_landmarks[0].landmark

            EAR_L = compute_EAR(lm, left=True)
            EAR_R = compute_EAR(lm, left=False)
            EAR_val = (EAR_L + EAR_R) / 2

            left_eye = crop_eye(frame_rgb, lm, left=True)
            if left_eye.size > 0:
                L = cv2.resize(left_eye,(128,128)).astype(np.float32)/255.0
                L = np.expand_dims(L,0)
                eye_prob_L = tflite_predict(eye_interpreter, eye_in, eye_out, L)

            right_eye = crop_eye(frame_rgb, lm, left=False)
            if right_eye.size > 0:
                R = cv2.resize(right_eye,(128,128)).astype(np.float32)/255.0
                R = np.expand_dims(R,0)
                eye_prob_R = tflite_predict(eye_interpreter, eye_in, eye_out, R)

            eye_prob = (eye_prob_L + eye_prob_R) / 2

            if show_boxes:
                cv2.rectangle(frame,(10,10),(100,40),(0,255,0),2)

        ear_closed = (EAR_val < EAR_THRESHOLD)
        cnn_closed = (eye_prob > CNN_EYE_THRESHOLD)

        fused = (
            FACE_WEIGHT * face_prob +
            EYE_WEIGHT  * eye_prob +
            EAR_WEIGHT  * (1 - EAR_val)
        )

        if cnn_closed or ear_closed:
            eye_closed_frames += 1
        else:
            eye_closed_frames = 0

        is_drowsy = (eye_closed_frames >= EYE_CLOSED_MAX_FRAMES) or (fused > 0.55)

        status = "DROWSY" if is_drowsy else "ALERT"
        color = (0,0,255) if is_drowsy else (0,255,0)

        # DISPLAY
        cv2.putText(frame, f"Status: {status}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1.5,color,3)

        cv2.putText(frame, f"FaceCNN={face_prob:.2f}", (20,90),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        cv2.putText(frame, f"EyeCNN L={eye_prob_L:.2f} R={eye_prob_R:.2f}",
                    (20,120), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        cv2.putText(frame, f"EAR={EAR_val:.3f}", (20,150),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        cv2.putText(frame, f"ClosedFrames={eye_closed_frames}", (20,180),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(200,200,200),2)

        now = time.time()
        fps = 1/(now - prev_time); prev_time = now
        cv2.putText(frame, f"FPS={fps:.1f}", (20,210),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

        cv2.imshow("Fusion (Face + Both Eyes + EAR)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_boxes", action="store_true")
    args = parser.parse_args()
    main(show_boxes=args.show_boxes)
