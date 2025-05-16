import cv2
import numpy as np
import os

def crop_white_margins(image):
    """
    Przycina białe marginesy z obrazu tabeli.
    """
    # Konwertuj na skale szarości i binaryzuj (inwersja: czarne = zawartość)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Znajdź kontur zawartości
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return image  # nic nie znaleziono

    x, y, w, h = cv2.boundingRect(coords)
    cropped = image[y:y+h, x:x+w]
    return cropped

def crop_all_tables_in_folder(folder):
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, fname)
            image = cv2.imread(path)
            if image is None:
                print(f"⚠️ Nie można odczytać {fname}")
                continue

            cropped = crop_white_margins(image)
            cv2.imwrite(path, cropped)
            print(f"✅ Przycięto: {fname}")

if __name__ == "__main__":
    folder_path = "data/processed_images"  # <-- podaj swoją ścieżkę
    crop_all_tables_in_folder(folder_path)
