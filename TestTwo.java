import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;

public class TestTwo {
    public static final int IMAGE_WIDTH = 28;
    public static final int IMAGE_HEIGHT = 28;
    public static final int INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
    public static final int NUM_CLASSES = 10;

    public static void main(String[] args) {
        // Load training data from the "train" folder (expects subfolders 0-9)
        String trainPath = "train";
        ArrayList<TrainingExample> trainingExamples = new ArrayList<>();
        
        for (int label = 0; label < NUM_CLASSES; label++) {
            File labelFolder = new File(trainPath + File.separator + label);
            if (!labelFolder.exists() || !labelFolder.isDirectory()) {
                System.err.println("Folder not found: " + labelFolder.getAbsolutePath());
                continue;
            }
            File[] imageFiles = labelFolder.listFiles();
            if (imageFiles == null) continue;
            for (File imgFile : imageFiles) {
                try {
                    BufferedImage img = ImageIO.read(imgFile);
                    if (img == null) continue;
                    float[] inputVector = imageToFloatArray(img);
                    trainingExamples.add(new TrainingExample(inputVector, label));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        
        System.out.println("Loaded " + trainingExamples.size() + " training examples.");
        
        // Convert training examples to arrays
        int numSamples = trainingExamples.size();
        float[][] trainingInputs = new float[numSamples][INPUT_SIZE];
        float[][] trainingTargets = new float[numSamples][NUM_CLASSES];
        for (int i = 0; i < numSamples; i++) {
            TrainingExample ex = trainingExamples.get(i);
            trainingInputs[i] = ex.input;
            trainingTargets[i] = new float[NUM_CLASSES];
            trainingTargets[i][ex.label] = 1.0f; // one-hot encoding
        }
        
        // Create a neural network with architecture: {784, 128, 64, 10}
        int[] architecture = {INPUT_SIZE, 128, 64, NUM_CLASSES};
        Brain brain = new Brain(architecture);
        
        // Train network (adjust learning rate and epochs as needed)
        float learningRate = 0.01f;
        int epochs = 10;  // Increase as needed
        
        System.out.println("Starting training...");
        brain.train(trainingInputs, trainingTargets, learningRate, epochs);
        System.out.println("Training finished.");
        
        // Serialize the trained network to "trained_brain.ser"
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("trained_brain.ser"))) {
            oos.writeObject(brain);
            System.out.println("Trained network serialized to trained_brain.ser");
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        // Optionally, test on the training set and print accuracy
        int correct = 0;
        for (int i = 0; i < numSamples; i++) {
            float[] output = brain.pass(trainingInputs[i]);  // inference uses softmax
            int predicted = argmax(output);
            if (predicted == trainingExamples.get(i).label) {
                correct++;
            }
        }
        System.out.println("Training Accuracy: " + ((float)correct / numSamples * 100) + "%");
    }
    
    // Convert a BufferedImage (assumed 28x28 grayscale) to a normalized float array
    public static float[] imageToFloatArray(BufferedImage img) {
        float[] result = new float[INPUT_SIZE];
        int index = 0;
        for (int y = 0; y < IMAGE_HEIGHT; y++) {
            for (int x = 0; x < IMAGE_WIDTH; x++) {
                int rgb = img.getRGB(x, y);
                int r = (rgb >> 16) & 0xff;
                int g = (rgb >> 8) & 0xff;
                int b = rgb & 0xff;
                // Average the channels to get a grayscale value
                int gray = (r + g + b) / 3;
                result[index++] = gray / 255.0f;
            }
        }
        return result;
    }

    
    // Helper method: returns index of maximum value
    public static int argmax(float[] arr) {
        int index = 0;
        float max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                index = i;
            }
        }
        return index;
    }
}

// Simple helper class to bundle training examples
class TrainingExample {
    public float[] input;
    public int label;
    
    public TrainingExample(float[] input, int label) {
        this.input = input;
        this.label = label;
    }
}
