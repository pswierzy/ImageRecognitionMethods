import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

os.environ['LOKY_MAX_CPU_COUNT'] = '4'

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

def build_gabor_filters():
    filters = []
    ksize = 15
    
    for theta in np.arange(0, np.pi, np.pi / 4): 
        for lamda in [np.pi/4, np.pi/2, np.pi]: 
            sigma = 1.0 
            gamma = 0.5
            
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
            
    return filters

def extract_gabor_energy_signatures(img, filters):
    features = []
    
    for kern in filters:
        filtered = cv2.filter2D(img, cv2.CV_32F, kern)
        energy_mean = np.mean(filtered ** 2)
        energy_std = np.std(filtered ** 2)
        features.extend([energy_mean, energy_std])
        
    return np.array(features)

def prepare_all_features(emojis_dict, filters):
    names = list(emojis_dict.keys())
    X_gabor = []
    
    for name in names:
        img = emojis_dict[name]
        gabor_feat = extract_gabor_energy_signatures(img, filters)
        
        norm = np.linalg.norm(gabor_feat)
        if norm > 0:
            gabor_feat = gabor_feat / norm
            
        X_gabor.append(gabor_feat)
        
    return names, np.array(X_gabor)

def retrieve_best_match_gabor(query_name, emojis_dict, filters):
    names, X_gabor = prepare_all_features(emojis_dict, filters)
    query_idx = names.index(query_name)
    
    db_names = [n for i, n in enumerate(names) if i != query_idx]
    X_db = np.delete(X_gabor, query_idx, axis=0)
    
    X_query = X_gabor[query_idx].reshape(1, -1)
    
    knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    knn.fit(X_db)
    distances, indices = knn.kneighbors(X_query)
    
    best_match = db_names[indices[0][0]]
    best_dist = distances[0][0]
    
    return best_match, best_dist

def show_results(query_name, emojis, filters, no_print=False):
    if not no_print:
        print(f"Szukany wzorzec: {query_name}")
        print("-" * 50)

    best_name, best_dist = retrieve_best_match_gabor(query_name, emojis, filters)

    if not no_print:
        print(f"Metoda: Sygnatury Energii Gabora | Najlepsze dopasowanie: {best_name} (Dystans: {best_dist:.4f})")

    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(emojis[query_name], cmap="gray", vmin=0, vmax=1)
    plt.title("Query\n(Wzorzec)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(emojis[best_name], cmap="gray", vmin=0, vmax=1)
    plt.title(f"Gabor Energy (1-NN)\n{best_name}\nDist: {best_dist:.2f}")
    plt.axis("off")

    plt.tight_layout()
    plt.suptitle("Image Retrieval - Cechy Energii Gabora", fontsize=14, y=1.1)
    plt.show()

if __name__ == "__main__":
    emojis = load_emojis()
    gabor_filters = build_gabor_filters()
    
    show_results("1F602", emojis, gabor_filters)
    show_results("1F60D", emojis, gabor_filters, no_print=False)
    show_results("1F630", emojis, gabor_filters, no_print=False)