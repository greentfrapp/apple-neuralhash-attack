# apple-neuralhash-attack

Demonstrates iterative FGSM on Apple's NeuralHash model.

TL;DR: It is possible to apply noise to CSAM images and make them to look like regular images to the model. The noise does degrade the CSAM image (see samples), but I'm certain better attacks that cause less degradation are possible too.

## Instructions

### Get ONNX model
Obtain the onnx model from [AsuharietYgvar/AppleNeuralHash2ONNX](https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX). You should have a path to a `model.onnx` file.

### Convert ONNX model to TF model
Then convert the ONNX model to a Tensorflow model by first installing the `onnx_tf` library via [onnx/onnx-tensorflow](https://github.com/onnx/onnx-tensorflow). Then run the following:

```bash
python3 convert.py -o /path/to/model.onnx
```

This will save a Tensorflow model to the current directory as `model.pb`.

### Run adversarial attack
Finally, run the adversarial attack with the following:

```bash
python3 nnhash_attack.py --seed /path/to/neuralhash_128x96_seed1.dat
```

Other arguments:

```bash
-m           Path to Tensorflow model (defaults to "model.pb")
--good       Path to good image (defaults to "samples/doge.png")
--bad        Path to bad image (defaults to "samples/grumpy_cat.png")
--lr         Learning rate (defaults to 3e-1)
--save_every Save every interval (defaults to 2000)
```

This will save generated images to `samples/iteration_{i}.png`.

Output:
```bash
# Some Tensorflow boilerplate...
Iteration #2000: L2-loss=134688, Hash Similarity=0.2916666666666667
Good Hash: 11d9b097ac960bd2c6c131fa
Bad Hash : 20f1089728150af2ca2de49a
Saving image to samples/iteration2000.png...
Iteration #4000: L2-loss=32605, Hash Similarity=0.41666666666666677
Good Hash: 11d9b097ac960bd2c6c131fa
Bad Hash : 20d9b097ac170ad2cfe170da
Saving image to samples/iteration4000.png...
Iteration #6000: L2-loss=18547, Hash Similarity=0.4166666666666667
Good Hash: 11d9b097ac960bd2c6c131fa
Bad Hash : 20d9b097ac170ad2c7c1f0de
Saving image to samples/iteration6000.png...
```

## Credit

- [KhaosT/nhcalc](https://github.com/KhaosT/nhcalc) for uncovering NeuralHash private API.
- [Tencent/TNN](https://github.com/Tencent/TNN) for compiled Core ML to ONNX script.
- [AsuharietYgvar/AppleNeuralHash2ONNX](https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX) for instructions to extract ONNX model
- [onnx/onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) for converting ONNX model to Tensorflow model
- [Gist by unrealwill](https://gist.github.com/unrealwill/c480371c3a4bf3abb29856c29197c0be)  that suggested a similar attack
