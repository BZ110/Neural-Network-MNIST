import java.util.concurrent.ThreadLocalRandom;
import java.io.Serializable;
/**
 * Simple neuron application.
 * 
 * @author Bashar Zain
 * @version 1.0
 */
public class Neuron implements Serializable {
    private float[] weights;
    private float bias;

    private float lastOutput;     // Store last output (for backprop)
    private float lastWeightedSum; // Store last weighted input sum
    private float[] lastInput;    // Store last input

    public Neuron(int numInputs) {
        weights = new float[numInputs];
        // Xavier/Glorot initialization: limit = sqrt(6 / (numInputs + 1))
        // (Here, we assume the neuron has one output; adjust if needed.)
        float limit = (float) Math.sqrt(6.0 / (numInputs + 1));
        
        for (int i = 0; i < numInputs; i++) {
            // Initialize weights uniformly between -limit and limit.
            weights[i] = ThreadLocalRandom.current().nextFloat() * 2 * limit - limit;
        }
        
        // Initialize bias uniformly between -limit and limit.
        bias = ThreadLocalRandom.current().nextFloat() * 2 * limit - limit;
    }


    // Compute the sum of weighted inputs + bias
    public float compute(float[] input) {
        if (input.length != weights.length) {
            throw new IllegalArgumentException("Input size must match weight size!");
        }

        lastInput = input.clone(); // Save input for backprop
        float sum = bias;

        for (int i = 0; i < input.length; i++) {
            sum += input[i] * weights[i]; // the magic.
        }

        lastWeightedSum = sum;
        lastOutput = ReLU(sum);
        return lastOutput;
    }
    
    public float computeRaw(float[] input) {
        if (input.length != weights.length) {
            throw new IllegalArgumentException("Input size must match weight size!");
        }
        float sum = bias;
        for (int i = 0; i < input.length; i++) {
            sum += input[i] * weights[i];
        }
        return sum;
    }


    // ReLU Activation Function
    private float ReLU(float x) {
        return Math.max(0, x);
    }

    // Derivative of ReLU
    public float ReLUPrime() {
        return lastWeightedSum > 0 ? 1f : 0f;
    }

    // Getters and setters
    public float[] getWeights() {
        return weights;
    }

    public float getBias() {
        return bias;
    }

    public void setBias(float newBias) {
        this.bias = newBias;
    }

    public float getLastOutput() {
        return lastOutput;
    }

    public float[] getLastInput() {
        return lastInput;
    }

    public float getLastWeightedSum() {
        return lastWeightedSum;
    }
}
