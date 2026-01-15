import cv2
import time
import threading
import torch
import numpy as np
import pyttsx3
import queue
from collections import deque
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


print(torch.cuda.is_available())
print(torch.version.cuda)


# --- IMPORTS DE TES MODULES CYTHON ---
try:
    from medial_axis import trajectoire,vecteur_directeur,check_collisions,generer_alertes_collision
    from description_image import decode_segmap_cython
    from bounding_box import get_bounding_boxes,generer_description
    from prediction_model import predict_model

    print("✅ Modules Cython chargés avec succès.")
except ImportError as e:
    print(f"❌ Erreur d'importation : {e}. Vérifie tes fichiers .pyd")

palette = {
    0: (128, 64, 128),    # Road (Violet Cityscapes)
    1: (244, 35, 232),    # Sidewalk (Rose Cityscapes) - Surface principale
    2: (110, 110, 110),   # Alley (Gris sombre) - Pour différencier de la route
    3: (200, 0, 0),       # Bike Lane (Rouge Foncé) - Zone de danger
    4: (0, 255, 255),     # Braille Blocks (CYAN ELECTRIQUE) - Doit péter sur le rose !
    5: (255, 165, 0),     # Caution Zone (Orange Vif) - Attention danger
    6: (70, 70, 70),      # Building (Gris foncé Cityscapes)
    7: (190, 153, 153),   # Fence (Beige Cityscapes)
    8: (220, 20, 60),     # Pedestrian (Rouge Cityscapes)
    9: (153, 153, 153),   # Pole (Gris clair Cityscapes)
    10: (107, 142, 35),   # Vegetation (Vert Cityscapes)
    11: (0, 0, 142),      # Vehicle (Bleu foncé Cityscapes)
    12: (102, 102, 156),  # Wall (Bleu gris Cityscapes)
    13: (220, 220, 0)     # Traffic Sign (Jaune Cityscapes)
}
id2label = {
    4:"Braille Blocks",
    5:"Caution Zone",
    8:"Pedestrian",
    9:"Pole",
    11:"Vehicle",
    13:"Traffic Sign"
}

palette_np = np.zeros((14, 3), dtype=np.uint8)
for class_id, color in palette.items():
    palette_np[class_id] = color

# --- PARAMÈTRES ---
CAM_INDEX = 0
WIDTH, HEIGHT = 512, 512  # Taille pour l'affichage et SegFormer
MODEL_DIR = r"C:\Users\rasou\Desktop\PER\Modele_Complet"
TARGET_CLASSES = np.array([1, 2, 3, 4, 5, 8], dtype=np.int32) # À adapter à ton modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- GLOBAL SYNC ---
stop_flag = False
frame_queue = deque(maxlen=1)       # Toujours la frame la plus fraîche
voice_queue = queue.PriorityQueue() # Priorité aux alertes

def voice_loop():
    # On n'initialise plus ici, on le fera dans la boucle
    print("📢 Thread Audio prêt (Mode Reset Auto).")

    while not stop_flag:
        try:
            # On attend un message
            priority, message = voice_queue.get(timeout=1.0)
            print(f"🔊 Tentative de lecture : {message}")

            # --- INITIALISATION ÉPHÉMÈRE ---
            # On recrée le moteur pour "réveiller" le driver audio Windows
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)

            engine.say(message)
            engine.runAndWait()

            # On détruit/libère proprement le moteur
            del engine

            voice_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ Erreur thread audio : {e}")


def capture_loop():
    global stop_flag
    # Utilisation de DirectShow
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la caméra {CAM_INDEX}")
        stop_flag = True
        return

    # On définit les paramètres APRÈS l'ouverture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 30) # Optionnel, évite si ça crash encore

    while not stop_flag:
        success, frame = cap.read()
        if success:
            frame_queue.append(frame)
        else:
            # Petit délai si la lecture échoue temporairement
            time.sleep(0.01)

    cap.release()

# --- 3. THREAD MODÈLE
def model_loop(model, processor):
    # Pour accèlerer le processus

    model = model.to(device)
    model.eval()

    global stop_flag
    last_voice_time = 0

    print("🚀 Boucle de traitement lancée.")

    while not stop_flag:
        if not frame_queue:
            time.sleep(0.01)
            continue

        # A. Récupération de l'image
        frame_bgr = frame_queue.pop()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        prediction_model = predict_model(model=model,processor=processor,img=frame_rgb,device=device)
        prediction_short = prediction_model.astype(np.int16)

        pred_mask_small = cv2.resize(prediction_short, (128, 128), interpolation=cv2.INTER_NEAREST)
        img_rgb = decode_segmap_cython(pred_mask_small,palette_np)

        if time.time() - last_voice_time > 5.0:
            try:
                # 1. Toujours calculer les données
                skel = trajectoire(img_rgb, 5)
                trajectoire_float = skel.astype(np.float32)
                v_dir = vecteur_directeur(trajectoire_float)
                boxes = get_bounding_boxes(pred_mask_small.astype(np.int16), TARGET_CLASSES)

                # 2. Générer les messages
                phrases = generer_description(boxes, v_dir, id2label)
                collisions = check_collisions(trajectoire_float, boxes)
                alertes_collisions = generer_alertes_collision(collisions, id2label)

                # 3. Envoyer les alertes de collision (Urgent)
                # On filtre pour ne pas envoyer "Trajectoire libre" en boucle
                alertes_urgentes = [a for a in alertes_collisions if "Attention" in a]

                for msg in alertes_urgentes:
                    voice_queue.put((1, msg)) # Priorité 1

                # 4. Envoyer les descriptions (S'il n'y a pas d'alerte urgente)
                if not alertes_urgentes and phrases:
                    voice_queue.put((2, phrases[0])) # On envoie juste la phrase principale

                # 5. SI RIEN n'est détecté du tout (optionnel)
                if not alertes_urgentes and not phrases:
                    voice_queue.put((3, "Chemin dégagé"))

                # --- IMPORTANT : On met à jour le temps quoi qu'il arrive ---
                last_voice_time = time.time()

            except Exception as e:
                print(f"Erreur dans le calcul Cython: {e}")
                last_voice_time = time.time() # Évite de boucler sur l'erreur


        seg_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        seg_resized = cv2.resize(seg_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        affichage_final = np.hstack((frame_bgr, seg_resized))

        # 4. Afficher le résultat combiné
        cv2.imshow("Vision Assistance - Polytech", affichage_final)
        #cv2.imshow("Vision Assistance - Polytech", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag = True



# --- 4. MAIN ---
def main():
    global stop_flag

    print("⏳ Chargement de SegFormer...")
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR).to(device)

    model.eval()

    # Création des threads
    t_audio = threading.Thread(target=voice_loop, daemon=True)
    t_video = threading.Thread(target=capture_loop, daemon=True)
    t_model = threading.Thread(target=model_loop, args=(model, processor), daemon=True)

    # Démarrage
    t_model.start()
    t_video.start()
    t_audio.start()

    try:
        while not stop_flag:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_flag = True

    print("Cleanup...")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()