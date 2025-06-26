import numpy as np
from sklearn.datasets import fetch_openml

# ============================
# CARREGAMENTO DOS DADOS

# função de perda cross-entropy para classificação multiclasse
def cross_entropy_loss(Y, A2):
    m = Y.shape[1]  # número de amostras
    # Pequeno valor 1e-9 adicionado para evitar log(0)
    loss = -np.sum(Y * np.log(A2 + 1e-9)) / m
    return loss

# codificação one-hot dos rótulos (ex: label 3 vira [0,0,0,1,0,0,0,0,0,0])
def one_hot_encode(y, num_classes=10):
    m = y.shape[0]
    one_hot = np.zeros((num_classes, m))
    for idx, label in enumerate(y):
        one_hot[label, idx] = 1
    return one_hot

# função softmax para converter logits em probabilidades (camada de saída)
def softmax(z):
    z_stable = z - np.max(z, axis=0, keepdims=True)  # estabilidade numérica
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# função de ativação ReLU (Retifica valores negativos para zero)
def relu(x):
    return np.maximum(0, x)


# carrega o dataset MNIST (784 pixels por imagem, 70.000 imagens)
mnist = fetch_openml('mnist_784', as_frame=False, parser='liac-arff')

# normaliza os dados (0-255 → 0-1) e transpõe para shape [features, amostras]
X = mnist['data'].T / 255.0  

# converte os rótulos para inteiros e depois para one-hot
y = mnist['target'].astype(int)
Y = one_hot_encode(y)

lr = 0.2  # taxa de aprendizado
epochs = 100  # número de epochs
m = X.shape[1]  # número de amostras

# ============================
# INICIALIZAÇÃO DE PESOS

# camada oculta com 128 neurônios, entrada com 784 pixels
W1 = np.random.randn(128, 784) * 0.01  # pesos pequenos aleatórios
b1 = np.zeros((128, 1))                # bias inicial zero

# camada de saída com 10 neurônios (0–9)
W2 = np.random.randn(10, 128) * 0.01
b2 = np.zeros((10, 1))

# ============================
# TREINAMENTO 

for epoch in range(epochs):
    # FORWARD PASS

    # Z1 = W1·X + b1 → saída linear da camada oculta
    Z1 = W1 @ X + b1

    # A1 = ReLU(Z1) → ativação da camada oculta
    A1 = relu(Z1)

    # Z2 = W2·A1 + b2 → saída linear da camada de saída
    Z2 = W2 @ A1 + b2

    # A2 = softmax(Z2) → probabilidades das classes
    A2 = softmax(Z2)

    # CÁLCULO DA PERDA

    loss = cross_entropy_loss(Y, A2)
    print(f'Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}')

    # BACKPROPAGATION

    # derivada da perda em relação à saída da rede
    dZ2 = A2 - Y

    # gradiente da função de perda em relação a W2 e b2
    dW2 = (1/m) * dZ2 @ A1.T
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    # gradiente da perda em relação à saída da camada oculta
    dA1 = W2.T @ dZ2

    # gradiente da ReLU (1 para valores positivos, 0 para negativos)
    dZ1 = dA1 * (Z1 > 0)

    # gradientes para W1 e b1
    dW1 = (1/m) * dZ1 @ X.T
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    # ATUALIZAÇÃO DOS PESOS (Gradiente Descendente) 
    
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

# ============================
# SALVAR MODELO TREINADO

# salva os pesos e bias em um arquivo npz
np.savez('model_mnist.npz', W1=W1, b1=b1, W2=W2, b2=b2)
print("✅ Modelo salvo em 'model_mnist.npz'")
