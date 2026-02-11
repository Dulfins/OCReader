
from inference import model2annotations, load_text_detector
from sugoi_translator.translator import translate_ja_to_en
from manga_ocr import MangaOcr
import threading
import queue
import numpy as np
import cv2
from mss import mss
# ocr = MangaOcr()

model_path = r'data/comictextdetector.pt'
model = load_text_detector(model_path)

# Live Screen OCR 
frame_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

latest_boxes = []
boxes_lock = threading.Lock()

bounding_box = {'top': 0, 'left': 0, 'width': 1980, 'height': 1080}

def screen_capture():
    sct = mss()
    while not stop_event.is_set():
        frame = np.array(sct.grab(bounding_box))

        # keep only latest frame
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

        frame_queue.put(frame)

def detection_worker():
    global latest_boxes
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        img = frame[:, :, :3] 

        _, _, yolo_boxes = model2annotations(model, img)

        with boxes_lock:
            latest_boxes = [tuple(map(int, box)) for box in yolo_boxes]


        # ocr_texts = []

        # for box in yolo_boxes:
        #     x1, y1, x2, y2 = box
        #     cropped_image = img.crop((x1, y1, x2, y2))
        #     text = ocr(cropped_image)

        #     norm = normalize_text(text)
        #     if norm:
        #         ocr_texts.append(norm)


        # # Send text to translator
        # new_texts = [t for t in ocr_texts if t not in translation_cache]

        # if new_texts:
        #     translation = translate_ja_to_en(new_texts)
        #     for src, tgt in zip(new_texts, translation):
        #         translation_cache[src] = tgt

def display_loop():
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.queue[-1].copy()
            
            with boxes_lock:
                boxes = list(latest_boxes)

            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("screen", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            cv2.destroyAllWindows()
            break

threading.Thread(target=screen_capture, daemon=True).start()
threading.Thread(target=detection_worker, daemon=True).start()

display_loop()


