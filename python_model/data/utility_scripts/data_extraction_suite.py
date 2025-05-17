import cv2
import numpy as np
import os
from glob import glob


def preprocess_for_grid_detection(gray):
    """
    Przygotowanie obrazu do wykrywania linii tabeli.
    Zwraca dwa osobne obrazy: z liniami pionowymi i poziomymi.
    """
    # 1. Binaryzacja
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY_INV, 15, 10
    )

    # 2a. Pionowe linie
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    vertical_temp = cv2.erode(thresh, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_temp, vertical_kernel, iterations=1)

    # 2b. Poziome linie
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    horizontal_temp = cv2.erode(thresh, horizontal_kernel, iterations=1)
    horizontal_lines = cv2.dilate(horizontal_temp, horizontal_kernel, iterations=1)

    return vertical_lines, horizontal_lines


# === Główna funkcja ===
def snap_lines(detected_lines, expected_positions, tolerance=10 ):
    snapped = []
    for expected in expected_positions:
        near = [l for l in detected_lines  if abs(l - expected) <= tolerance]
        if near:
            snapped.append(int(np.median(near)))
        else:
            snapped.append(expected)
    return snapped

def extract_cells_with_snapping(image_path, output_dir, col_labels, cell_size=(64, 64), margin=2):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vertical_img, horizontal_img = preprocess_for_grid_detection(gray)
   

    # Krawędzie
    edges_v = cv2.Canny(vertical_img, 50, 200, apertureSize=3)
    edges_h = cv2.Canny(horizontal_img, 50, 200, apertureSize=3)

    # Linie pionowe i poziome osobno
    lines_v = cv2.HoughLinesP(edges_v, 1, np.pi / 180 *2, threshold=100, minLineLength=100, maxLineGap=10)
    lines_h = cv2.HoughLinesP(edges_h, 1, np.pi / 180 *2, threshold=100, minLineLength=100, maxLineGap=10)

    # Przygotowanie list
    verticals = []
    horizontals = []

    if lines_v is not None:
        for line in lines_v:
            x1, y1, x2, y2 = line[0]
            verticals.append((x1 + x2) // 2)

    if lines_h is not None:
        for line in lines_h:
            x1, y1, x2, y2 = line[0]
            horizontals.append((y1 + y2) // 2)


    h, w = gray.shape
    rows = 10
    cols = len(col_labels)
    #print(cols)
    expected_rows = np.linspace(0, h, rows + 1, dtype=int)
    expected_cols = np.linspace(0, w, cols + 1, dtype=int)

    snapped_rows = snap_lines(horizontals, expected_rows)
    snapped_cols = snap_lines(verticals, expected_cols)

    #snapped_rows = horizontals
    print("snapped cols len", len(snapped_cols))
    #snapped_cols = verticals

    # Zapisz siatkę podglądową
    vis = img.copy()
    for y in snapped_rows:
        cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
    for x in snapped_cols:
        cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)
    grid_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_grid.png")
    #cv2.imshow("Grid", vis)
    #cv2.waitKey(0)
    cv2.imwrite(grid_path, vis)

    for i in range(rows):
        for j in range(cols):
            y1, y2 = snapped_rows[i], snapped_rows[i + 1]
            x1, x2 = snapped_cols[j], snapped_cols[j + 1]

           # if (y2 - y1 < 10) or (x2 - x1 < 10):
            #    continue

            cell = gray[y1 + margin:y2 - margin, x1 + margin:x2 - margin]
            if cell.shape[0] == 0 or cell.shape[1] == 0:
                continue

            cell_resized = cv2.resize(cell, cell_size)
            label_dir = os.path.join(output_dir, col_labels[j])
            os.makedirs(label_dir, exist_ok=True)
            filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_r{i}_c{j}.png"
            cv2.imwrite(os.path.join(label_dir, filename), cell_resized)

# === Główna pętla dla wielu plików ===

# Kolumny w cyklu: 7,7,7,7,6 → razem 34 kolumn (po 5 obrazów w cyklu)
full_column_list = [
    'A', 'A_PL', 'B', 'C', 'C_PL', 'D', 'E', 
    'E_PL','F', 'G', 'H', 'I', 'J', 'K', 'L',
    'L_PL', 'M', 'N', 'N_PL', 'O', 'O_PL', 'P',
    'R', 'S', 'S_PL', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z', 'Z_PL0', 'Z_PL1'
]

column_cycle = [7, 7, 7, 7, 6]

input_folder = "../data/processed_images"  # <- podmień na swoją
output_folder = "../data/output_letters"

import re

def extract_number(filename):
    match = re.search(r'table_(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

image_files = sorted(
    glob(os.path.join(input_folder, "table_*.png")),
    key=extract_number
)
total_files = len(image_files)

for idx, img_path in enumerate(image_files):
    cycle_index = idx % len(column_cycle)
    col_count = column_cycle[cycle_index]

    # Wycinamy zakres kolumn dla tego pliku
    col_start = sum(column_cycle[:cycle_index])  # indeks pierwszej kolumny w tym pliku
    col_labels = full_column_list[col_start:col_start + col_count]
    print(col_start, col_labels)

    print(f"▶️ Przetwarzam {os.path.basename(img_path)} ({col_count} kolumn: {col_labels})")

    try:
        extract_cells_with_snapping(
            img_path,
            output_folder,
            col_labels=col_labels
        )
    except Exception as e:
        print(f"⛔ Błąd przetwarzania {img_path}: {e}")
