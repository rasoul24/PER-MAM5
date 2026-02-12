import cv2
import time
import threading
import torch
import numpy as np
import pyttsx3
import queue
import time
from collections import deque
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


torch.backends.cudnn.benchmark = True


try:
    from medial_axis import trajectoire, vecteur_directeur, check_collisions, generer_alertes_collision
    from description_image import decode_segmap_cython
    from bounding_box import get_bounding_boxes, generer_description, get_position_objets
    from prediction_model import predict_model
    print(" Modules Cython chargés.")
except ImportError as e:
    print(f" Erreur Cython : {e}")


MODEL_DIR = r"C:\Users\rasou\Desktop\PER\Modele_Complet"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CLASSES = np.array([1, 2, 3, 4, 5, 8], dtype=np.int32)

IA_RES = (720, 480)      # Résolution augmentée pour une meilleure précision

DISPLAY_RES = (720, 480)  # Résolution d'affichage

palette_np = np.zeros((14, 3), dtype=np.uint8)
palette = {0:(128,64,128), 1:(244,35,232), 4:(0,255,255), 8:(220,20,60), 11:(0,0,142)}
for cid, col in palette.items(): palette_np[cid] = col
id2label = {4:"Braille Blocks", 5:"Caution Zone", 8:"Pedestrian", 9:"Pole", 11:"Vehicle", 13:"Traffic Sign"}

stop_flag = False
frame_queue = deque(maxlen=1)
voice_queue = queue.PriorityQueue()
global_seg_overlay = np.zeros((DISPLAY_RES[1], DISPLAY_RES[0], 3), dtype=np.uint8)

# --- 1. THREAD AUDIO ---
def voice_loop():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
    except: return
    while not stop_flag:
        try:
            _, message = voice_queue.get(timeout=0.5)
            engine.say(message)
            engine.runAndWait()
            voice_queue.task_done()
        except queue.Empty: continue

# --- 2. THREAD CAPTURE & AFFICHAGE ---
def capture_loop():
    global stop_flag
    #cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_RES[1])

    while not stop_flag:
        success, frame = cap.read()
        if success:
            frame_queue.append(frame)
            # On combine le flux réel et le dernier masque IA calculé
            combined = np.hstack((frame, global_seg_overlay))
            cv2.imshow("Vision Polytech - 256px Precision", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'): stop_flag = True
        else: time.sleep(0.01)
    cap.release()

# --- 3. THREAD MODÈLE (Précision augmentée) ---
def model_loop(model, processor):
    global global_seg_overlay, stop_flag
    model.to(device).eval()
    last_voice_time = 0

    kernel = np.ones((5, 5), np.uint8)

    print(" Boucle IA démarrée...")

    with torch.inference_mode():
        while not stop_flag:
            if not frame_queue:
                time.sleep(0.01)
                continue

            t_start_frame = time.time()

            frame_bgr = frame_queue[-1]
            frame_small = cv2.resize(frame_bgr, IA_RES)
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            prediction = predict_model(model=model, processor=processor, img=frame_rgb, device=device)

            mask = cv2.resize(prediction.astype(np.int16), DISPLAY_RES, interpolation=cv2.INTER_NEAREST)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            img_rgb = decode_segmap_cython(mask, palette_np)
            seg_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            try:
                boxes = get_bounding_boxes(mask, TARGET_CLASSES)
                for obj in boxes:
                    x1, y1, x2, y2 = obj["bbox"]
                    label_name = id2label.get(obj["id"], "Object")
                    cv2.rectangle(seg_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(seg_bgr, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception as e:
                print(f"Erreur calcul boxes : {e}")

            global_seg_overlay = seg_bgr


            t_end_frame = time.time()
            dt = t_end_frame - t_start_frame
            fps = 1.0 / dt if dt > 0 else 0


            fps_text = f"FPS: {fps:.1f} ({dt:.3f}s)"


            cv2.rectangle(seg_bgr, (5, 5), (180, 35), (0, 0, 0), -1)
            cv2.putText(seg_bgr, fps_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            global_seg_overlay = seg_bgr

            print(f"\r {fps_text} ", end="")

            now = time.time()
            if now - last_voice_time > 10.0:

                try:
                    skel = trajectoire(img_rgb, 5).astype(np.uint8)
                    seg_bgr[skel > 0] = (0, 255, 255)

                    v_dir = vecteur_directeur(skel.astype(np.float32))
                    center_x, center_y = DISPLAY_RES[0] // 2, DISPLAY_RES[1] - 50
                    if not np.isnan(v_dir[0]):
                        end_point = (int(center_x + v_dir[0] * 50), int(center_y + v_dir[1] * 50))
                        cv2.arrowedLine(seg_bgr, (center_x, center_y), end_point, (255, 255, 255), 3)

                    boxes = get_bounding_boxes(mask, TARGET_CLASSES)
                    for obj in boxes:
                        x1, y1, x2, y2 = obj["bbox"]
                        label_name = id2label.get(obj["id"], "Object")
                        cv2.rectangle(seg_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        cv2.putText(seg_bgr, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                except Exception as e:
                    print(f"Erreur visuels (skel/boxes) : {e}")

def main2():
    global stop_flag
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR).to(device)

    threads = [
        threading.Thread(target=voice_loop, daemon=True),
        threading.Thread(target=capture_loop, daemon=True),
        threading.Thread(target=model_loop, args=(model, processor), daemon=True)
    ]
    for t in threads: t.start()

    try:
        while not stop_flag: time.sleep(0.5)
    except KeyboardInterrupt: stop_flag = True
    cv2.destroyAllWindows()

def main():
    global stop_flag

    try:
        processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        )
    except Exception as e:
        print(f"Erreur chargement processeur : {e}")
        return

    print(f"Chargement du modèle depuis {MODEL_DIR}...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    ).to(device)

    if device.type == "cuda":
        model = model.half()
        torch.backends.cudnn.benchmark = True

    t_audio = threading.Thread(target=voice_loop, daemon=True)
    t_model = threading.Thread(target=model_loop, args=(model, processor), daemon=True)
    t_cap = threading.Thread(target=capture_loop, daemon=True)

    t_audio.start()
    t_model.start()
    t_cap.start()

    print(" Système prêt et optimisé.")

    try:
        while not stop_flag:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_flag = True


if __name__ == "__main__":

    main()

