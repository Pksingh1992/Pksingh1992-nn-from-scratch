
# src/numpy_nn.py
import numpy as np

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-9)

def one_hot(y, k):
    Y = np.zeros((len(y), k), dtype=np.float32)
    Y[np.arange(len(y)), y] = 1.0
    return Y

class DenseNet:
    def __init__(self, layers, lr=0.1, reg=1e-4):
        # layers e.g. [784, 128, 64, 10]
        self.lr = lr
        self.reg = reg
        self.W, self.b = [], []
        rng = np.random.default_rng(42)
        for i in range(len(layers)-1):
            fan_in = layers[i]
            # He init for ReLU layers
            w = rng.normal(0.0, np.sqrt(2.0/fan_in), size=(layers[i], layers[i+1])).astype(np.float32)
            b = np.zeros(layers[i+1], dtype=np.float32)
            self.W.append(w); self.b.append(b)

    def forward(self, X):
        # cache activations for backprop
        self.a = [X.astype(np.float32)]
        self.z = []
        for i in range(len(self.W)):
            z = self.a[-1] @ self.W[i] + self.b[i]
            self.z.append(z)
            if i < len(self.W)-1:
                a = np.maximum(0.0, z)   # ReLU
            else:
                a = softmax(z)           # final softmax
            self.a.append(a)
        return self.a[-1]

    def loss_ce(self, Y_true, P):
        # cross-entropy + L2 regularization
        ce = -np.mean(np.sum(Y_true * np.log(P + 1e-9), axis=1))
        l2 = 0.5 * self.reg * sum((w*w).sum() for w in self.W)
        return ce + l2

    def backward(self, Y_true):
        grads_W = [None]*len(self.W)
        grads_B = [None]*len(self.b)
        m = Y_true.shape[0]

        # derivative at softmax-crossentropy: dL/dz_L = (P - Y)/m
        delta = (self.a[-1] - Y_true) / m

        for i in reversed(range(len(self.W))):
            a_prev = self.a[i]
            grads_W[i] = (a_prev.T @ delta) + self.reg * self.W[i]
            grads_B[i] = delta.sum(axis=0)
            if i > 0:
                # backprop through ReLU
                delta = (delta @ self.W[i].T) * (a_prev > 0)

        return grads_W, grads_B

    def step(self, grads_W, grads_B):
        for i in range(len(self.W)):
            self.W[i] -= self.lr * grads_W[i]
            self.b[i] -= self.lr * grads_B[i]
