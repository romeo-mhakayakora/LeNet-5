# LeNet-5 on MNIST

Minimal LeNet-5 implementation trained on MNIST.

## Setup

Create/activate a virtualenv, then:

```bash
pip install -r requirements.txt
pip install -e .
```

## Train

```bash
python scripts/train.py --epochs 20 --batch-size 32 --model-out models/best_model.keras
```

## Notes

- Input images are padded from 28x28 to 32x32.
- Uses `tanh` activations and average pooling, matching classic LeNet-5 style.


## Theory

- See docs/lenet5_theory.md for a layer-by-layer explanation of how LeNet-5 works. 

