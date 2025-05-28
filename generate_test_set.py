import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from pathlib import Path

# carrega o MNIST
mnist = fetch_openml('mnist_784', as_frame=False, parser='liac-arff')
X = mnist['data']
y = mnist['target'].astype(int)

# cria a pasta onde as imagens serão salvas
output_dir = Path("test")
output_dir.mkdir(exist_ok=True)

# gera e salva 10 imagens
for idx in range(10):
    image = X[idx].reshape(28, 28) / 255.0
    label = y[idx]
    filename = output_dir / f"mnist_{label}_{idx}.png"
    plt.imsave(filename, image, cmap="gray")
    print(f"Imagem salva: {filename}")
