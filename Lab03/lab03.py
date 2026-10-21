import os
import cv2
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

# Wczytanie obrazu w skali szarości.
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'lena.jpg')

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Nie można wczytać obrazu: {image_path}")

img_float = np.float32(img) / 255.0


# korelacja i konwolucja
def correlation(image, kernel):
    kernel_rows, kernel_cols = kernel.shape
    pad_rows, pad_cols = kernel_rows // 2, kernel_cols // 2
    padded_image = np.pad(
        image, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode="constant"
    )
    output = np.zeros(image.shape, dtype=np.float32)

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            region = padded_image[
                row : row + kernel_rows, col : col + kernel_cols
            ]
            output[row, col] = np.sum(region * kernel)

    return output

def convolution(image, kernel):
    return correlation(image, np.flip(kernel))

# Sobel - detekcja krawędzi
kernel = np.array(
    [[1, 2, 1], 
     [0, 0, 0], 
     [-1, -2, -1]], dtype=np.float32
)
correlation_result = correlation(img_float, kernel)
convolution_result = convolution(img_float, kernel)

display_limit = max(
    float(np.max(np.abs(correlation_result))),
    float(np.max(np.abs(convolution_result))),
)
if display_limit == 0:
    display_limit = 1.0

plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
plt.imshow(img_float, cmap="gray", vmin=0, vmax=1)
plt.title("Obraz oryginalny")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(correlation_result, cmap="gray", vmin=-display_limit, vmax=display_limit)
plt.title("Korelacja")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(convolution_result, cmap="gray", vmin=-display_limit, vmax=display_limit)
plt.title("Konwolucja")
plt.axis("off")

plt.tight_layout()
plt.show()

# wczytanie emoji
def load_emojis(folder_name="emojis"):
    folder_path = os.path.join(script_dir, folder_name)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Nie znaleziono folderu z emoji: {folder_path}")

    emojis = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(".png"):
            emoji_name = os.path.splitext(filename)[0]
            emoji_path = os.path.join(folder_path, filename)
            emoji_image = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
            if emoji_image is None:
                raise OSError(f"Nie można wczytać obrazu emoji: {emoji_path}")

            emoji_gray = cv2.cvtColor(
                            emoji_image[:, :, :3], cv2.COLOR_BGR2GRAY
                        )
            alpha = emoji_image[:, :, 3].astype(np.float32) / 255.0

            emoji_gray = emoji_gray.astype(np.float32) / 255.0
            emojis[emoji_name] = (
                emoji_gray * alpha + (1.0 - alpha)
            )

    if not emojis:
        raise ValueError(f"Folder nie zawiera plików PNG z emoji: {folder_path}")

    return emojis

# znormalizowaną korelacja
def normalized_correlation(first, second):
    if first.shape != second.shape:
        raise ValueError("Porównywane obrazy muszą mieć taki sam rozmiar.")
    if first.ndim != 2:
        raise ValueError("Obrazy emoji muszą być w skali szarości.")

    first_centered = first.astype(np.float32) - np.mean(first)
    second_centered = second.astype(np.float32) - np.mean(second)
    denominator = np.sqrt(
        np.sum(first_centered**2) * np.sum(second_centered**2)
    )
    if denominator == 0:
        return 0.0

    return float(np.sum(first_centered * second_centered) / denominator)


emojis = load_emojis()
query_name = "1F602"
if query_name not in emojis:
    raise KeyError(f"Nie znaleziono emoji wzorcowego: {query_name}")

query_emoji = emojis[query_name]
similarities = {
    name: normalized_correlation(query_emoji, emoji)
    for name, emoji in emojis.items()
    if name != query_name
}
if not similarities:
    raise ValueError("Do wyszukiwania podobnego emoji potrzebne są co najmniej dwa obrazy.")

best_match_name = max(similarities, key=similarities.get)
best_match_score = similarities[best_match_name]
print(
    f"Emoji najbardziej podobne do {query_name}: {best_match_name} "
    f"(korelacja: {best_match_score:.3f})"
)

# przedstawienie wyników
columns = 4
rows = (len(emojis) + columns - 1) // columns
plt.figure(figsize=(10, 3 * rows))
for index, (name, emoji) in enumerate(emojis.items(), start=1):
    plt.subplot(rows, columns, index)
    plt.imshow(emoji, cmap="gray", vmin=0, vmax=1)
    if name == query_name:
        title = f"{name} — wzorzec"
    elif name == best_match_name:
        title = f"{name} — najlepsze ({best_match_score:.2f})"
    else:
        title = f"{name} ({similarities[name]:.2f})"
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()