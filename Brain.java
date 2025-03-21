import java.io.Serializable;

/**
 * Neural Network: Manages multiple layers.
 * 
 * @author Bashar Zain
 * @version 1.1
 */
public class Brain implements Serializable {
    
    private Layer[] layers;
    
    public Brain(int[] nums) {
        layers = new Layer[nums.length - 1];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(nums[i], nums[i + 1]);
        }
    }
    
    // Pass forward data through layers.
    // Note: For inference, we return softmax outputs.
    // Pass forward data through layers.
    public float[] pass(float[] arr) {
        float[] res = arr;
        // Process all but the final layer normally
        for (int i = 0; i < layers.length - 1; i++) {
            res = layers[i].compute(res);
        }
        // For the final layer, use raw outputs (no activation)
        res = layers[layers.length - 1].computeRaw(res);
        return softmax(res);
    }

    
    // Forward pass without applying softmax (for training use)
    public float[] forward(float[] arr) {
        float[] res = arr;
        for (int i = 0; i < layers.length - 1; i++) {
            res = layers[i].compute(res);
        }
        res = layers[layers.length - 1].computeRaw(res);
        return res; // raw outputs for training
    }

    
    // Standard softmax function.
    float[] softmax(float[] arr) {
        float sumExp = 0;
        float[] expValues = new float[arr.length];
        for (int i = 0; i < arr.length; i++) {
            expValues[i] = (float) Math.exp(arr[i]);
            sumExp += expValues[i];
        }
        for (int i = 0; i < arr.length; i++) {
            expValues[i] /= sumExp;
        }
        return expValues;
    }
    
    void train(float[][] trainingInputs, float[][] trainingTargets, float learningRate, int epochs) {
        // Use a clip threshold (you can adjust this as needed)
        float clipThreshold = 0.1f;
        int sampleCount = trainingInputs.length;
        int[] indices = new int[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            indices[i] = i;
        }
        
        for (int epoch = 1; epoch <= epochs; epoch++) {
            // Shuffle indices each epoch
            for (int i = sampleCount - 1; i > 0; i--) {
                int j = (int)(Math.random() * (i + 1));
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
            
            double epochLoss = 0.0;
            
            // Process training samples in shuffled order
            for (int idx = 0; idx < sampleCount; idx++) {
                int sampleIdx = indices[idx];
                float[] input = trainingInputs[sampleIdx];
                float[] target = trainingTargets[sampleIdx];
                
                // --- Forward Pass ---
                float[][] layerInputs = new float[layers.length][];
                float[] activations = input; // initial input
                
                // Process hidden layers with activation.
                for (int i = 0; i < layers.length - 1; i++) {
                    layerInputs[i] = activations;
                    activations = layers[i].compute(activations);
                }
                // For the final layer, store its input and compute raw outputs.
                layerInputs[layers.length - 1] = activations;
                activations = layers[layers.length - 1].computeRaw(activations);
                
                float[] finalOutput = activations;
                float[] softmaxOutput = softmax(finalOutput);
                
                // --- Loss Computation (Cross-Entropy) ---
                double sampleLoss = 0.0;
                for (int i = 0; i < softmaxOutput.length; i++) {
                    sampleLoss -= target[i] * Math.log(softmaxOutput[i] + 1e-8);
                }
                epochLoss += sampleLoss;
                
                // --- Backward Pass ---
                float[][] deltas = new float[layers.length][];
                int L = layers.length;
                
                // Output layer delta: for cross-entropy with softmax, delta = softmax - target.
                Neuron[] outputNeurons = layers[L - 1].getNeurons();
                deltas[L - 1] = new float[outputNeurons.length];
                for (int i = 0; i < outputNeurons.length; i++) {
                    deltas[L - 1][i] = softmaxOutput[i] - target[i];
                }
                
                // Hidden layers: propagate error backward.
                for (int layerIdx = L - 2; layerIdx >= 0; layerIdx--) {
                    Neuron[] currentNeurons = layers[layerIdx].getNeurons();
                    deltas[layerIdx] = new float[currentNeurons.length];
                    Neuron[] nextNeurons = layers[layerIdx + 1].getNeurons();
                    for (int i = 0; i < currentNeurons.length; i++) {
                        float errorSum = 0;
                        for (int j = 0; j < nextNeurons.length; j++) {
                            float weight_ij = nextNeurons[j].getWeights()[i];
                            errorSum += deltas[layerIdx + 1][j] * weight_ij;
                        }
                        deltas[layerIdx][i] = currentNeurons[i].ReLUPrime() * errorSum;
                    }
                }
                
                // --- Weight and Bias Updates (with gradient clipping) ---
                for (int layerIdx = 0; layerIdx < L; layerIdx++) {
                    Neuron[] neurons = layers[layerIdx].getNeurons();
                    float[] layerInput = layerInputs[layerIdx];
                    for (int i = 0; i < neurons.length; i++) {
                        float[] neuronWeights = neurons[i].getWeights();
                        for (int w = 0; w < neuronWeights.length; w++) {
                            float grad = learningRate * deltas[layerIdx][i] * layerInput[w];
                            if (grad > clipThreshold) {
                                grad = clipThreshold;
                            } else if (grad < -clipThreshold) {
                                grad = -clipThreshold;
                            }
                            neuronWeights[w] -= grad;
                        }
                        float biasGrad = learningRate * deltas[layerIdx][i];
                        if (biasGrad > clipThreshold) {
                            biasGrad = clipThreshold;
                        } else if (biasGrad < -clipThreshold) {
                            biasGrad = -clipThreshold;
                        }
                        float newBias = neurons[i].getBias() - biasGrad;
                        neurons[i].setBias(newBias);
                    }
                }
            }
            System.out.println("Epoch " + epoch + " average loss: " + (epochLoss / sampleCount));
        }
    }


}
