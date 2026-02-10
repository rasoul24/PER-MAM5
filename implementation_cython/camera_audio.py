import cv2
import time
import threading
import torch
import numpy as np
import pyttsx3
import queue
from collections import deque
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Optimisation CUDA
torch.backends.cudnn.benchmark = True

# --- IMPORTS CYTHON ---
try:
    from medial_axis import trajectoire, vecteur_directeur, check_collisions, generer_alertes_collision
    from description_image import decode_segmap_cython
    from bounding_box import get_bounding_boxes, generer_description, get_position_objets
    from prediction_model import predict_model
    print(" Modules Cython chargés.")
except ImportError as e:
    print(f" Erreur Cython : {e}")

# --- CONFIGURATION ---
MODEL_DIR = "/home/rasoul/Bureau/PER/Modele_Complet"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CLASSES = np.array([1, 2, 3, 4, 5, 8], dtype=np.int32)

# Résolutions
IA_RES = (720, 480)      # Résolution augmentée pour une meilleure précision
#DISPLAY_RES = (320, 240)  # Résolution d'affichage
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
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
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

    with torch.inference_mode():
        while not stop_flag:
            if not frame_queue:
                time.sleep(0.01); continue

            # A. Récupération et redimensionnement à 256x256 pour la précision
            frame_bgr = frame_queue[-1]
            frame_small = cv2.resize(frame_bgr, IA_RES)
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            # B. Inférence IA
            # On utilise une résolution de 256 pour que les petits objets soient visibles
            prediction = predict_model(model=model, processor=processor, img=frame_rgb, device=device)

            # C. Post-traitement Cython (on reste sur 128 pour la vitesse des calculs géométriques)
            mask = cv2.resize(prediction.astype(np.int16), (720, 480), interpolation=cv2.INTER_NEAREST)
            img_rgb = decode_segmap_cython(mask, palette_np)

            # D. Mise à jour de l'overlay pour l'affichage
            seg_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            global_seg_overlay = cv2.resize(seg_bgr, DISPLAY_RES, interpolation=cv2.INTER_NEAREST)

            # E. Logique vocale
            now = time.time()
            if now - last_voice_time > 10.0:
                try:
                    skel = trajectoire(img_rgb, 5).astype(np.float32)
                    boxes = get_bounding_boxes(mask, TARGET_CLASSES)
                    position_objets = get_position_objets(boxes,vecteur_directeur(skel),id2label)

                    #phrases = generer_description2(boxes, vecteur_directeur(skel), id2label)
                    phrases = generer_description(position_objets)

                    print("phrases = ",phrases)
                    if phrases: voice_queue.put((2, phrases))
                    last_voice_time = now
                except: pass

def main():
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

if __name__ == "__main__":
    main()
