
import numpy as np
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def relu(x):
    return np.maximum(0, x)

def softmax(z):
    z_stable = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def image_to_vector(path):
    img = Image.open(path).convert('L')
    img = img.resize((28, 28))
    img_arr = np.asarray(img)  # Substituído aqui
    img_vector = img_arr.reshape(-1)       # shape (784,)

    return img_vector

def predict(image_vector, model_path="model_mnist.npz"):
    model = np.load(model_path)
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']

    x = image_vector.reshape(-1, 1) / 255.0

    Z1 = W1 @ x + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)

    prediction = int(np.argmax(A2))
    confidence = float(np.max(A2))
    return prediction, confidence

if __name__ == "__main__":
    image_path = input("Digite o caminho da imagem (ex: numero.png): ")
    vetor = image_to_vector(image_path)
    pred, conf = predict(vetor)

    print(f"\n🔢 Predição: {pred}")
    print(f"📊 Confiança: {conf*100:.2f}%")
