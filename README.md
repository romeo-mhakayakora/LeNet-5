# LeNet-5 on MNIST

Minimal LeNet-5 implementation trained on MNIST.

## Setup

Create/activate a virtualenv, then:

`ash
pip install -r requirements.txt
`

## Train

`ash
python scripts/train.py --epochs 20 --batch-size 32 --model-out models/best_model.keras
`

## Notes

- Input images are padded from 28x28 to 32x32.
- Uses 	anh activations and average pooling, matching classic LeNet-5 style.

