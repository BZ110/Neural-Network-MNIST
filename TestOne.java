import java.util.Arrays;

public class TestOne {
    public static void main(String[] args) {
        int[] architecture = {50, 75, 75, 10};

        // Initialize the neural network (Brain)
        Brain brain = new Brain(architecture);

        // Create training data: 10 samples (one for each output neuron), each with 50 features.
        int numSamples = 10;  // Must equal the number of output neurons.
        int numInputs = 50;
        float[][] trainingData = new float[numSamples][numInputs];
        float[][] trainingTargets = new float[numSamples][10]; // One-hot target for 10 classes

        // Fill each training sample with random values plus a small offset to differentiate each sample.
        // Also, create a one-hot target vector for each sample (sample i => class i)
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numInputs; j++) {
                trainingData[i][j] = (float) Math.random() + i * 0.1f;
            }
            // Create one-hot target for sample i
            trainingTargets[i] = new float[10];
            trainingTargets[i][i] = 1.0f;
        }

        // Set learning parameters
        float learningRate = 0.01f;
        int epochs = 50; // Increase if necessary

        // Start training
        System.out.println("Starting training...");
        brain.train(trainingData, trainingTargets, learningRate, epochs);
        System.out.println("Training finished.\n");

        // Test the network on each training sample
        for (int i = 0; i < numSamples; i++) {
            float[] input = trainingData[i];
            float[] output = brain.pass(input);
            int predictedClass = argmax(output);
            System.out.println("Sample " + i + " predicted class: " + predictedClass 
                               + ", output: " + Arrays.toString(output));
        }
    }

    // Helper method to find the index of the highest value (argmax)
    private static int argmax(float[] array) {
        int bestIndex = 0;
        float bestValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > bestValue) {
                bestValue = array[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }
}
