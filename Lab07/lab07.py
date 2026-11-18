import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

from keras import layers, models

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IMG_SIZE = 16

def load_emojis(folder_name="emojis"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(script_dir, "..")
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

            emoji_gray = cv2.cvtColor(emoji_image[:, :, :3], cv2.COLOR_BGR2GRAY)

            alpha = emoji_image[:, :, 3].astype(np.float32) / 255.0
            emoji_gray = emoji_gray.astype(np.float32) / 255.0
            emojis[emoji_name] = (emoji_gray * alpha + (1.0 - alpha))

    if not emojis:
        raise ValueError(f"Folder nie zawiera plików PNG z emoji: {folder_path}")

    return emojis

emojis = load_emojis()

def augment_dataset(emojis_dict, num_augmentations=30):
    X_augmented = []
    names = list(emojis_dict.keys())
    h, w = IMG_SIZE, IMG_SIZE

    for name in names:
        base_img = cv2.resize(emojis_dict[name], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        X_augmented.append(base_img)  # oryginał też dodajemy do zbioru

        for _ in range(num_augmentations):
            angle = np.random.uniform(-10, 10)
            scale = np.random.uniform(0.9, 1.1)
            M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)

            M_rot[0, 2] += np.random.uniform(-1.0, 1.0)
            M_rot[1, 2] += np.random.uniform(-1.0, 1.0)

            aug_img = cv2.warpAffine(
                base_img, M_rot, (w, h),
                borderMode=cv2.BORDER_CONSTANT, borderValue=1.0
            )

            noise = np.random.normal(0, 0.02, aug_img.shape).astype(np.float32)
            aug_img = np.clip(aug_img + noise, 0.0, 1.0)

            X_augmented.append(aug_img)

    X_augmented = np.array(X_augmented, dtype=np.float32)
    X_augmented = np.expand_dims(X_augmented, axis=-1)
    return X_augmented

np.random.seed(42)
X_train = augment_dataset(emojis, num_augmentations=40)
np.random.shuffle(X_train)
print(f"Rozmiar rozszerzonego zbioru treningowego: {X_train.shape}")


def build_autoencoder(img_size=IMG_SIZE):
    input_img = layers.Input(shape=(img_size, img_size, 1))

    # ENKODER
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # redukcja do mapy 8x8

    encoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='encoded_map')(x)  # mapa 8x8x1

    # DEKODER
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)  # powrót do 16x16
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # powrót do 16x16
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded, name="autoencoder")
    encoder = models.Model(input_img, encoded, name="encoder")

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder(IMG_SIZE)
autoencoder.summary()

history = autoencoder.fit(
    X_train, X_train,
    epochs=200,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

plt.figure(figsize=(6, 3))
plt.plot(history.history['loss'], label='Train MSE')
plt.plot(history.history['val_loss'], label='Val MSE')
plt.title('Przebieg funkcji kosztu (MSE)')
plt.xlabel('Epoka')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


names = list(emojis.keys())
X_resized = []
for n in names:
    X_resized.append(cv2.resize(emojis[n], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
X_base_resized = np.expand_dims(np.array(X_resized), axis=-1)

encoded_maps = encoder.predict(X_base_resized, verbose=0)
reconstructed_imgs = autoencoder.predict(X_base_resized, verbose=0)

n_show = min(5, len(names))
plt.figure(figsize=(10, 5))
for i in range(n_show):
    plt.subplot(3, n_show, i + 1)
    plt.imshow(emojis[names[i]], cmap='gray', vmin=0, vmax=1)
    plt.title(f"Oryginał\n{names[i]}")
    plt.axis('off')

    plt.subplot(3, n_show, i + 1 + n_show)
    plt.imshow(encoded_maps[i].squeeze(), cmap='viridis')
    plt.title("Enkoder (8x8)")
    plt.axis('off')

    plt.subplot(3, n_show, i + 1 + 2 * n_show)
    plt.imshow(reconstructed_imgs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.title("Wyjście sieci")
    plt.axis('off')

plt.tight_layout()
plt.show()

def prepare_all_features_ae(emojis_dict, encoder_model):
    names = list(emojis_dict.keys())
    
    resized_imgs = []
    for n in names:
        img_resized = cv2.resize(emojis_dict[n], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        resized_imgs.append(img_resized)
        
    imgs = np.expand_dims(np.array(resized_imgs), axis=-1)
    
    encoded = encoder_model.predict(imgs, verbose=0)
    
    X_features = encoded.reshape(len(names), -1)
    return names, X_features

names, X_ae = prepare_all_features_ae(emojis, encoder)

perplexity_val = min(15, max(2, len(names) - 1))
tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_ae)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.0)

for i, name in enumerate(names):
    img = emojis[name]
    imagebox = OffsetImage(img, cmap='gray', zoom=1.8)
    ab = AnnotationBbox(imagebox, (X_tsne[i, 0], X_tsne[i, 1]), frameon=False)
    ax.add_artist(ab)
    ax.annotate(name, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=7, alpha=0.7, xytext=(0, -14), textcoords='offset points', ha='center')

plt.title("Wizualizacja t-SNE cech wydzielonych przez enkoder")
plt.xlabel("t-SNE wymiar 1")
plt.ylabel("t-SNE wymiar 2")
plt.grid(True, alpha=0.3)
plt.show()

def retrieve_best_match_ae(query_name, emojis_dict, encoder_model):
    names, X_ae = prepare_all_features_ae(emojis_dict, encoder_model)
    query_idx = names.index(query_name)
    
    db_names = [n for i, n in enumerate(names) if i != query_idx]
    X_db = np.delete(X_ae, query_idx, axis=0)
    X_query = X_ae[query_idx].reshape(1, -1)
    
    knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    knn.fit(X_db)
    distances, indices = knn.kneighbors(X_query)
    
    best_match = db_names[indices[0][0]]
    best_dist = distances[0][0]
    
    return best_match, best_dist

def show_results(query_name, emojis_dict, encoder_model, no_print=False):
    if not no_print:
        print(f"Szukany wzorzec: {query_name}")
        print("-" * 50)

    best_name, best_dist = retrieve_best_match_ae(query_name, emojis_dict, encoder_model)

    if not no_print:
        print(f"Metoda: Autoenkoder | Najlepsze dopasowanie: {best_name} (Dystans: {best_dist:.4f})")

    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(emojis_dict[query_name], cmap="gray", vmin=0, vmax=1)
    plt.title("Query\n(Wzorzec)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(emojis_dict[best_name], cmap="gray", vmin=0, vmax=1)
    plt.title(f"Autoencoder (1-NN)\n{best_name}\nDist: {best_dist:.2f}")
    plt.axis("off")

    plt.tight_layout()
    plt.suptitle("Image Retrieval - Cechy z Autoenkodera", fontsize=14, y=1.1)
    plt.show()

show_results("1F602", emojis, encoder)
show_results("1F60D", emojis, encoder)
show_results("1F630", emojis, encoder)