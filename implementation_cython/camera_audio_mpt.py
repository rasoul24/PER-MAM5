import cv2
import time
import threading
import torch
import numpy as np
import pyttsx3
import queue
import time
import math
from collections import deque
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from tracking import ObstacleTracker


torch.backends.cudnn.benchmark = True


try:
    from medial_axis import trajectoire, vecteur_directeur, check_collisions, generer_alertes_collision
    from description_image import decode_segmap_cython
    from bounding_box import get_bounding_boxes, generer_description, get_position_objets
    from prediction_model import predict_model
    print(" Modules Cython chargés.")
except ImportError as e:
    print(f" Erreur Cython : {e}")


#MODEL_DIR = r"C:\Users\rasou\Desktop\PER\Modele_Complet"
MODEL_DIR = r"C:\Users\deniz\OneDrive\Documents\MAM5\PER-MAM5-main\Modele_Complet"
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
    last_collision_time = 0
    
    kernel = np.ones((5, 5), np.uint8)
    
    DYNAMIC_CLASSES = [8, 11]

    # Initialisation du tracker (mémoire de 5 frames)
    tracker = ObstacleTracker(max_disappeared=5)

    print(" Boucle IA démarrée...")

    with torch.inference_mode():
        while not stop_flag:
            if not frame_queue:
                time.sleep(0.01)
                continue

            t_start_frame = time.time()
            now = time.time()

            # --- PRE-PROCESSING ---
            frame_bgr = frame_queue[-1]
            frame_small = cv2.resize(frame_bgr, IA_RES)
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            # --- INFERENCE ---
            prediction = predict_model(model=model, processor=processor, img=frame_rgb, device=device)
            mask = cv2.resize(prediction.astype(np.int16), DISPLAY_RES, interpolation=cv2.INTER_NEAREST)

            # Nettoyage visuel
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            img_rgb = decode_segmap_cython(mask, palette_np)
            seg_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            try:
                # 1. Récupération de TOUTES les boîtes (pour l'affichage global)
                boxes = get_bounding_boxes(mask, TARGET_CLASSES)

                # --- 2. LOGIQUE DE FILTRAGE & TRACKING ---
                
                # On ne garde que les objets DYNAMIQUES pour le tracking
                tracking_rects = []
                for obj in boxes:
                    if obj["id"] in DYNAMIC_CLASSES:
                        tracking_rects.append(obj["bbox"])
                
                # Mise à jour du tracker avec seulement les objets mobiles
                objects, velocities = tracker.update(tracking_rects)
                
                center_x_screen = DISPLAY_RES[0] // 2
                bottom_y_screen = DISPLAY_RES[1]

                # Analyse des mouvements
                for (objectID, centroid) in objects.items():
                    vx, vy = velocities[objectID]
                    cx, cy = centroid
                    
                    # Visualisation : Flèche rouge pour la direction
                    end_point = (int(cx + vx * 10), int(cy + vy * 10))
                    cv2.arrowedLine(seg_bgr, (cx, cy), end_point, (0, 0, 255), 2, tipLength=0.3)
                    
                    # --- DETECTION DE COLLISION ---
                    # 1. Distance : Est-ce proche (< 250px du bas) ?
                    dist_to_user = math.hypot(cx - center_x_screen, cy - bottom_y_screen)
                    
                    # 2. Vecteur : Est-ce que ça descend vers nous (vy > 1.5) ?
                    is_approaching = vy > 1.5
                    
                    if dist_to_user < 250 and is_approaching:
                        # Timer de sécurité (3s) pour l'audio
                        if now - last_collision_time > 3.0:
                            print(f"\n DANGER : Objet {objectID} arrive sur vous !")
                            voice_queue.put((1, "Attention, véhicule en approche !"))
                            last_collision_time = now

                # --- 3. AFFICHAGE VISUEL ---
                for obj in boxes:
                    x1, y1, x2, y2 = obj["bbox"]
                    label_name = id2label.get(obj["id"], "Object")
                    
                    # Code couleur : ROUGE si c'est un danger potentiel, VERT si c'est statique
                    color = (0, 0, 255) if obj["id"] in DYNAMIC_CLASSES else (0, 255, 0)
                    
                    cv2.rectangle(seg_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(seg_bgr, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            except Exception as e:
                print(f"Erreur tracking : {e}")

            global_seg_overlay = seg_bgr

            # --- CALCUL FPS ---
            t_end_frame = time.time()
            dt = t_end_frame - t_start_frame
            fps = 1.0 / dt if dt > 0 else 0
            fps_text = f"FPS: {fps:.1f}"
            
            cv2.rectangle(seg_bgr, (5, 5), (150, 35), (0, 0, 0), -1)
            cv2.putText(seg_bgr, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"\r {fps_text} ", end="")

            # --- 4. DESCRIPTION ENVIRONNEMENT (Priorité basse) ---
            if now - last_voice_time > 6.0:
                try:
                    skel = trajectoire(img_rgb, 5).astype(np.uint8)
                    v_dir = vecteur_directeur(skel.astype(np.float32))
                    
                    positions = get_position_objets(boxes, v_dir, id2label)
                    phrases = generer_description(positions)

                    if phrases:
                        print(f"\n📢 Vocal: {phrases}")
                        voice_queue.put((2, phrases))

                    last_voice_time = now
                except Exception as e: pass

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

