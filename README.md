# Neural Network from Scratch (NumPy) — Fashion-MNIST

A compact portfolio project built after completing Andrew Ng’s 'Neural Networks and Deep Learning'.
It implements a fully-connected classifier from scratch in NumPy (forward pass, backprop, mini-batch SGD, L2 regularization),
trains on Fashion-MNIST, and logs results.

## Highlights
- Pure NumPy implementation (no autograd): ReLU, Softmax, Cross-Entropy, mini-batch SGD, L2 regularization.
- Clean training loop with loss/accuracy plots and saved metrics.
- Ready to run in Google Colab or locally.

## Results
Final test accuracy: 0.8633

<p float="left">
  <img src="results/loss.png" alt="loss curve" width="46%"/>
  <img src="results/accuracy.png" alt="accuracy curve" width="46%"/>
</p>

A few predictions:
<br>
<img src="results/predictions.png" alt="sample predictions" width="70%"/>

## Project Structure
```
nn-from-scratch/
├── src/
│   └── numpy_nn.py            # NumPy neural net (forward/backprop/SGD)
├── results/                   # Plots & metrics (generated)
│   ├── loss.png
│   ├── accuracy.png
│   ├── predictions.png
│   └── metrics.txt
└── notebooks/
```

## Run in Colab (recommended)
- Open a new Colab notebook and run the training cell from the README (or this repo’s notebook if provided).
- Plots and metrics will be saved to `results/`.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
# open your own notebook and use fashion_mnist from tf.keras.datasets, or adapt a small loader
```

> If TensorFlow is not available locally, you can load Fashion-MNIST via torchvision.datasets.FashionMNIST and
> reshape to (N, 784) before calling the NumPy model. The core model stays the same.

## Next steps
- Add momentum / learning-rate decay and compare curves.
- Try a tiny 1-hidden-layer ConvNet in Keras for a reference point.
- Export this project to GitHub.

## License
MIT