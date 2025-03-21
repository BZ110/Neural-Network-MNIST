import java.io.Serializable;

/**
 * Simple layer application. Usually the "hidden layer".
 * 
 * @author Bashar
 * @version 1.1
 */
public class Layer implements Serializable {
    private Neuron[] neurons;
    private float[] lastOutputs; // Store the output of this layer

    public Layer(int numInputs, int numNeurons) {
        this.neurons = new Neuron[numNeurons];
        for (int i = 0; i < numNeurons; i++) {
            neurons[i] = new Neuron(numInputs);
        }
    }
    
    // Process data and store the outputs.
    public float[] compute(float[] arr) {
        lastOutputs = new float[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            lastOutputs[i] = neurons[i].compute(arr);
        }
        return lastOutputs;
    }
    
    public float[] computeRaw(float[] arr) {
        float[] result = new float[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            result[i] = neurons[i].computeRaw(arr);
        }
        return result;
    }

    
    public int neuronCount() {
        return neurons.length;
    }
  
    public Neuron[] getNeurons() {
        return neurons;
    }
    
    // Optionally, expose last outputs
    public float[] getLastOutputs() {
        return lastOutputs;
    }
}
