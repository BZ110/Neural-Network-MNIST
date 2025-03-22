# Neural-Network-MNIST
A basic neural network built from scratch trained by the MNIST dataset.
## What is it?
This is a basic challenge I set myself to train a neural network to accurately guess numbers from the MNIST dataset.
### Constraints
1. I was not allowed to use any unnecessary imports.
2. I was not allowed to use any machine-learning libraries.
3. I had to create the forward pass on my own, with resources for the backpropagation.
## Results
The information below shows an example of a test case after the AI was trained.
```
--------------------------------------------------
Test Example 99
Expected label: 9
Predicted label: 9
Neuron percentages: 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 100.00% 
Raw logits: [-10.510929, -3.5772233, -8.770323, -1.4052339, 6.819688, -3.8787365, -13.801637, 8.478824, -0.16238567, 21.321339]
Average pixel value: 0.0944
Input sample (first 20 values): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
--------------------------------------------------
```
## What were some issues encountered on the way?
### Exploding Gradients
The random numbers I used messed up backpropagation, causing a phenomenon in neural networks called exploding gradients, where the gradients become unstable.
To fix this, I searched on the internet for a proper initialization for neurons using ReLU (Rectified Linear Unit), and came up with this:
```java
// Xavier/Glorot initialization: limit = sqrt(6 / (numInputs + 1))
// (Here, we assume the neuron has one output; adjust if needed.)
float limit = (float) Math.sqrt(6.0 / (numInputs + 1));

for (int i = 0; i < numInputs; i++) {
    // Initialize weights uniformly between -limit and limit.
    weights[i] = ThreadLocalRandom.current().nextFloat() * 2 * limit - limit;
}
        
// Initialize bias uniformly between -limit and limit.
bias = ThreadLocalRandom.current().nextFloat() * 2 * limit - limit;
```
### Testing Final Always On 9
I was training the neural network **in order**, which meant that the last number inside the network would be the most favoured, as it was the most recently backpropagated and trained to fit that number.
To fix this, I added randomization inside of the training function per each epoch, so the network wouldn't bias the most recently trained number as much.
```java
// Shuffle indices each epoch
for (int i = sampleCount - 1; i > 0; i--) {
  int j = (int)(Math.random() * (i + 1));
  int temp = indices[i];
  indices[i] = indices[j];
  indices[j] = temp;
}
```
## How Can I Test This Out?
1. Get a canvas, 28x28, grayscale, with the background being black.
2. Draw any sort of number, from 0 to 9.
3. Using the Guess class, (which will take trained_brain.ser, and image.png) my neural network will try to guess what number you drew.
### Results From My Test (Image Drawn On https://www.pixilart.com/): ✔️ Successful
```
--------------------------------------------------
Image.png Result
Predicted label: 3
Neuron percentages: 0.00% 0.00% 0.00% 100.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 
Raw logits: [-12.480954, -14.50602, 6.8725805, 25.332253, -5.301191, -8.171614, -15.852834, -1.4873123, 2.8422132, 5.3088803]
Average pixel value: 0.1811
Input sample (first 20 values): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
--------------------------------------------------
```
