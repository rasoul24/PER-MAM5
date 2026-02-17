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


#MODEL_DIR = r"C:\Users\rasou\Desktop\PER\Modele_Complet"
#MODEL_DIR = "/home/rasoul/Bureau/PER/Modele_Complet"
MODEL_DIR = r"C:\Users\rasou\Desktop\PER\modele_distiler"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TARGET_CLASSES = np.array([1, 2, 3, 4, 5, 8], dtype=np.int32)
TARGET_CLASSES = np.array([0, 1, 4, 5, 6, 8, 9, 11, 12, 13], dtype=np.int32)

IA_RES = (224, 224)      # Résolution augmentée pour une meilleure précision

DISPLAY_RES = (720, 480)  # Résolution d'affichage

palette_np = np.zeros((14, 3), dtype=np.uint8)

#palette = {0:(128,64,128), 1:(244,35,232), 4:(0,255,255), 8:(220,20,60), 11:(0,0,142)}

palette = {
    0:  (128, 64, 128), # Road (Violet)
    1:  (244, 35, 232), # SideWalk (Rose)
    4:  (0, 255, 255),  # Braille Blocks (Cyan - Très visible)
    5:  (220, 190, 40),  # Caution Zone (Orange/Jaune)
    6:  (70, 70, 70),   # Building (Gris)
    8:  (220, 20, 60),  # Pedestrian (Rouge)
    9:  (153, 153, 153),# Pole (Gris clair)
    11: (0, 0, 142),    # Vehicle (Bleu foncé)
    12: (102, 102, 156),# Wall (Mauve/Gris)
    13: (220, 220, 0)   # Traffic Sign (Jaune vif)
}

for cid, col in palette.items(): palette_np[cid] = col

# A tester avec les deux versions
#id2label = {4:"Braille Blocks", 5:"Caution Zone", 8:"Pedestrian", 9:"Pole", 11:"Vehicle", 13:"Traffic Sign"}
id2label = {0:"Road",1:"SideWalk", 4:"Braille Blocks", 5:"Caution Zone",6:"Building", 8:"Pedestrian", 9:"Pole", 11:"Vehicle", 12:"Wall" ,13:"Traffic Sign"}


stop_flag = False
frame_queue = deque(maxlen=1)
voice_queue = queue.PriorityQueue()
global_seg_overlay = np.zeros((DISPLAY_RES[1], DISPLAY_RES[0], 3), dtype=np.uint8)

# --- 1. THREAD AUDIO ---
def voice_loop():
    print("--- Thread Audio : Prêt ---")
    while not stop_flag:
        try:
            # On attend un message pendant 0.5s
            _, message = voice_queue.get(timeout=0.5)

            # INITIALISATION LOCALE (pour éviter le gel du thread)
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)

            print(f"--- Lecture : {message}")
            engine.say(message)
            engine.runAndWait()

            # NETTOYAGE (très important sous Windows)
            del engine

            voice_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Erreur thread vocal : {e}")
            time.sleep(0.1)

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

            #img_rgb = decode_segmap_cython(mask, palette_np)
            img_rgb = decode_segmap_cython(mask.astype(np.int16), palette_np)
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
                    skel = trajectoire(img_rgb, 5).astype(np.float32)
                    seg_bgr[skel > 0] = (0, 255, 255)
                    last_voice_time = now

                    v_dir = vecteur_directeur(skel.astype(np.float32))
                    center_x, center_y = DISPLAY_RES[0] // 2, DISPLAY_RES[1] - 50
                    if not np.isnan(v_dir[0]):
                        end_point = (int(center_x + v_dir[0] * 50), int(center_y + v_dir[1] * 50))
                        cv2.arrowedLine(seg_bgr, (center_x, center_y), end_point, (255, 255, 255), 3)

                    boxes = get_bounding_boxes(mask, TARGET_CLASSES)
                    collisions = check_collisions(skel,boxes)
                    message_collision = generer_alertes_collision(collisions,id2label)


                    positions = get_position_objets(boxes, v_dir, id2label)
                    phrases = generer_description(positions)

                    # 2. Sécurisation : Transformer les listes en chaînes de caractères (strings)
                    # generer_description renvoie souvent une liste de phrases
                    if isinstance(phrases, list):
                        phrase_finale = ". ".join(phrases)
                    else:
                        phrase_finale = str(phrases) if phrases else ""

                    # generer_alertes_collision peut renvoyer None ou une liste selon ton implémentation
                    if isinstance(message_collision, list):
                        collision_finale = ". ".join(message_collision)
                    elif message_collision is None:
                        collision_finale = ""
                    else:
                        collision_finale = str(message_collision)

                    # 3. Construction du message vocal complet
                    # On ne met le message dans la file que s'il y a quelque chose à dire
                    if phrase_finale or collision_finale:
                        # On combine les deux avec un espace, en filtrant les textes vides
                        message_complet = " ".join(filter(None, [phrase_finale, collision_finale]))

                        print(f"\n Vocal: {message_complet}")

                        # Envoi à la file avec priorité 2
                        # Si collision_finale n'est pas vide, on pourrait même passer en priorité 1
                        priorite = 1 if collision_finale else 2
                        voice_queue.put((priorite, message_complet))


                    for obj in boxes:
                        x1, y1, x2, y2 = obj["bbox"]
                        label_name = id2label.get(obj["id"], "Object")
                        cv2.rectangle(seg_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        cv2.putText(seg_bgr, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                except Exception as e:
                    print(f"Erreur visuels (skel/boxes) : {e}")


def main():
    global stop_flag

    try:
        #processor = SegformerImageProcessor.from_pretrained(
            #"nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        #)
        processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
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

