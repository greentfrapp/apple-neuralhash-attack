import argparse

import numpy as np
from PIL import Image
import tensorflow as tf


parser = argparse.ArgumentParser(description="Adversarial attack on Apple NeuralHash.")
parser.add_argument(
    "-m", "--model", type=str, default="model.pb", help="path to tensorflow model"
)
parser.add_argument(
    "--good", type=str, default="samples/doge.png", help="path to good image"
)
parser.add_argument(
    "--bad", type=str, default="samples/grumpy_cat.png", help="path to bad image"
)
parser.add_argument("--seed", type=str, required=True, help="path to seed file")
parser.add_argument("--lr", type=float, default=3e-1, help="learning rate")
parser.add_argument("--save_every", type=int, default=2000, help="save every interval")
args = parser.parse_args()


def load_image(filepath):
    """Open image and load as array.

    Arguments:
    ----------
    filepath (str): path to image file

    Returns:
    --------
    arr (np.array): image array
    """
    image = Image.open(filepath).convert("RGB").resize([360, 360])
    arr = (
        np.array(image).astype(np.float32).transpose(2, 0, 1).reshape([1, 3, 360, 360])
    )
    return arr


def preprocess(arr):
    """Preprocess image array with simple normalization.

    Arguments:
    ----------
    arr (np.array): image array

    Returns:
    --------
    arr (np.array): preprocessed image array
    """
    arr = arr / 255.0
    arr = arr * 2.0 - 1.0
    return arr


def get_gradient(model, x, target_logits):
    """Compute gradients that encourage model outputs to be the
    same as target outputs.

    Arguments:
    ----------
    model (tf model): model
    x (np.array): image array after preprocessing
    target_logits (tf.Tensor): target logits to optimize towards

    Returns:
    --------
    gradient (np.array): array of gradients
    """
    x_t = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_t)
        logits = model(image=preprocess(x_t))["leaf/logits"]
        loss = tf.nn.l2_loss(logits - target_logits)
    return t.gradient(loss, x_t).numpy()


def get_hash_sim(logits, target, seed):
    """Get hash similarity between two model outputs.

    Arguments:
    ----------
    logits (tf.Tensor): model output of adversarial image
    target (tf.Tensor): model output of target image

    Returns:
    --------
    sim (float): Float from 0 to 1 representing hash similarity
    """
    target_hash = seed.dot(target.numpy().flatten())
    logits_hash = seed.dot(logits.numpy().flatten())
    sim = np.mean((target_hash > 0) * (logits_hash > 0))
    return sim


def get_hash(logits, seed):
    """Get hash of model output.

    Arguments:
    ----------
    logits (tf.Tensor): model output

    Returns:
    --------
    hash_hex (str): Hexadecimal hash as string
    """
    hash_output = seed.dot(logits.numpy().flatten())
    hash_bits = "".join(["1" if it >= 0 else "0" for it in hash_output])
    hash_hex = "{:0{}x}".format(int(hash_bits, 2), len(hash_bits) // 4)
    return hash_hex


def main():
    model = tf.saved_model.load(args.model)

    seed = open(args.seed, "rb").read()[128:]
    seed = np.frombuffer(seed, dtype=np.float32)
    seed = seed.reshape([96, 128])

    # Load good image and compute good hash
    good_arr = load_image(args.good)
    good_logits = model(image=preprocess(good_arr))["leaf/logits"]
    good_hash = get_hash(good_logits, seed)

    # Load bad image
    bad_arr = load_image(args.bad)
    adv_arr = bad_arr.copy()

    lr = args.lr
    i = 0
    update = None
    while True:
        i += 1

        grad = get_gradient(model, adv_arr, good_logits)
        adv_arr -= lr * grad
        # Clip array so that we are still within bounds of RGB values
        adv_arr = np.clip(adv_arr, 0, 255)

        adv_logits = model(image=preprocess(adv_arr))["leaf/logits"]
        loss = tf.nn.l2_loss(adv_logits - good_logits).numpy()

        hash_sim = get_hash_sim(adv_logits, good_logits, seed)

        update = f"Iteration #{i}: L2-loss={int(loss)}, Hash Similarity={hash_sim}"
        print(update, end="\r")

        if i % args.save_every == 0 or hash_sim == 1:
            print(update)
            update = f"""Good Hash: {good_hash}
Bad Hash : {get_hash(adv_logits, seed)}
Saving image to samples/iteration{i}.png..."""
            print(update)
            Image.fromarray(adv_arr[0].transpose(1, 2, 0).astype(np.uint8)).save(
                f"samples/iteration_{i}.png"
            )
            if hash_sim == 1:
                print(f"Identical hash achieved at iteration {i}")
                break


if __name__ == "__main__":
    main()
