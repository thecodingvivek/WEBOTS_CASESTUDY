from controller import Robot
from PIL import Image
import numpy
import random
import os
import json
from keras.src.datasets import mnist

TIME_STEP = 65

print("Please wait e-puck bot load weights and biases")

(train_x, train_y), (testd_x, testd_y) = mnist.load_data()
train_x = train_x.reshape(60000, 784) / 255.0
testd_x = testd_x.reshape(10000, 784) / 255.0


def one_hot_encode(labels, num_classes=10):
    return numpy.eye(num_classes)[labels]

train_y = one_hot_encode(train_y)
testd_y = one_hot_encode(testd_y)

def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [numpy.random.randn(y) for y in sizes[1:]]
        self.weights = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        if os.path.exists("weights_biases.json"):
            print("Found saved model — loading weights and biases...")
            self.load("weights_biases.json")
        else:
            print("No saved model found — training from scratch.")
            self.SGD(list(zip(train_x, train_y)), epochs=15, batch_size=10, lr=3.0, test_data=list(zip(testd_x, testd_y)))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(numpy.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, batch_size, lr, test_data=None):
        print("Training started...")
        n = len(training_data)
        n_test = len(test_data) if test_data else None

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                self.update_batch(mini_batch, lr)

            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

        # Save weights and biases after training
        self.save("weights_biases.json")
        print("Model saved successfully!")

    def update_batch(self, batch, lr):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_b, delta_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_w)]

        self.weights = [w - (lr / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (lr / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backpropagation
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.outer(delta, activations[-2])

        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = numpy.dot(self.weights[-l + 1].T, delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.outer(delta, activations[-l - 1])

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        results = [(numpy.argmax(self.feedforward(x)), numpy.argmax(y)) for x, y in test_data]
        return sum(int(pred == y) for pred, y in results)
    

    def get_number(self,a):
        arr = self.feedforward(a)
        highest_index = numpy.argmax(arr)
        
        return highest_index

    def save(self, filename):
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        with open(filename, "w") as f:
            json.dump(data, f)

    def load(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        self.sizes = data["sizes"]
        self.weights = [numpy.array(w) for w in data["weights"]]
        self.biases = [numpy.array(b) for b in data["biases"]]

# ---------------- Run ----------------
net = Network([784, 30, 10])
# Create Robot instance
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Enable camera
camera = robot.getDevice('camera')
camera.enable(timestep)

# Wheel motors
rightWheel = robot.getDevice("right wheel motor")
leftWheel = robot.getDevice("left wheel motor")
rightWheel.setPosition(float('inf'))
leftWheel.setPosition(float('inf'))

# IR sensors
ir_sensors = []
for i in range(8):
    sensor = robot.getDevice(f"ps{i}")
    sensor.enable(timestep)
    ir_sensors.append(sensor)

# Speeds
leftSpeed = 2.0
rightSpeed = 2.0

WALL_THRESHOLD = 80
image_counter = 0
detected = 0

def rotateRight():
    print("started")
    duration = 3.141# seconds (tuned for 90°)
    steps = int(duration * 1000 / TIME_STEP)
    leftWheel.setVelocity(0.5)
    rightWheel.setVelocity(-0.5)

    for _ in range(steps):
        robot.step(TIME_STEP)

def rotateLeft():
    print("started")
    duration = 3.141# seconds (tuned for 90°)
    steps = int(duration * 1000 / TIME_STEP)
    leftWheel.setVelocity(-0.5)
    rightWheel.setVelocity(0.5)

    for _ in range(steps):
        robot.step(TIME_STEP)



while robot.step(timestep) != -1:
    ir_values = [sensor.getValue() for sensor in ir_sensors]
    front_center = ir_values[0]

    if front_center > WALL_THRESHOLD:
        leftSpeed = rightSpeed = 0.0

        if image_counter == 0:
            width = camera.getWidth()
            height = camera.getHeight()
            img_data = camera.getImage()

            if img_data is not None:
                flat_pixels = []
                for y in range(height):
                    for x in range(width):
                        r = camera.imageGetRed(img_data, width, x, y)
                        g = camera.imageGetGreen(img_data, width, x, y)
                        b = camera.imageGetBlue(img_data, width, x, y)
                        flat_pixels.extend([r, g, b])

                gray_pixels = [int(0.299*r + 0.587*g + 0.114*b) 
                               for r,g,b in zip(flat_pixels[0::3],
                                                flat_pixels[1::3],
                                                flat_pixels[2::3])]
                pred = net.get_number(gray_pixels)

                print(f"robot detected {pred}")
                if pred % 2 == 0:
                    rotate180()
                    leftSpeed = rightSpeed = 2.0
                else:
                    detected = 1
                    
            image_counter += 1
      

    leftWheel.setVelocity(leftSpeed)
    rightWheel.setVelocity(rightSpeed)

