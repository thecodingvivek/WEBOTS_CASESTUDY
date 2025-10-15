# E-Puck Robot Neural Network Controller

This project integrates **Webots E-Puck robot simulation** with a **custom-built neural network trained on MNIST digits** (from scratch without any external ML packages) to enable autonomous wall detection and decision-making behavior.
The robot uses its **camera** to capture an image, processes it through a **feedforward neural network**, and decides how to move based on the predicted digit So that we can simulate a maze game  which the bot can solve.
---

## Features

* **Custom Neural Network (from scratch)**
  Implements feedforward, backpropagation, and SGD (Stochastic Gradient Descent) without using TensorFlow/Keras layers.

*  **MNIST Digit Recognition**
  Trains on MNIST handwritten digits to predict numbers using sigmoid activation.

* **Model Persistence**
  Trained weights and biases are saved to `weights_biases.json` and reloaded automatically for reuse.

* **Dynamic Behavior**

  * When the robot detects a wall, it captures an image.
  * If the predicted number is **even**, the robot performs a **rotation** and continues forward.
  * If the predicted number is **odd**, it performs a different maneuver (stops or rotates).

###  Prerequisites

Make sure you have the following installed:

* **Webots** (R2023a or later)
* **Python 3.8+**
* **PIP packages:**

  ```bash
  pip install numpy pillow keras tensorflow
  ```

---

###  Running the Controller

When you run the simulation:
* The program will **train the neural network** if no saved model exists.
  This can take a few seconds.
* Once trained, it will save `weights_biases.json` for future runs.
* On next runs, the weights will load automatically.

You’ll see console output like:

```
Found saved model — loading weights and biases...
Training started...
Epoch 14: 9350 / 10000
Model saved successfully!
robot detected 4
```

---
### Neural Network Architecture

| Layer  | Size        | Description                         |
| :----- | :---------- | :---------------------------------- |
| Input  | 784 neurons | Flattened 28×28 pixel image         |
| Hidden | 30 neurons  | Dense layer with sigmoid activation |
| Output | 10 neurons  | Digit prediction (0–9)              |

###  Robot Logic Flow

1. The robot continuously reads **IR sensors**.
2. If the **front sensor** detects a wall (`value > 80`):

   * It **captures a camera image**.
   * Converts RGB → **grayscale**.
   * Passes the flattened pixel array to the trained neural network.
   * Based on the **predicted digit parity**:

     * **Even →**  Rotate Left
     * **Odd →**  Rotate Right

---



