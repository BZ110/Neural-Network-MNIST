import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Arrays;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.ObjectInputStream;

public class TestNetwork {
    public static final int IMAGE_WIDTH = 28;
    public static final int IMAGE_HEIGHT = 28;
    public static final int INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
    public static final int NUM_CLASSES = 10;
    
    public static void main(String[] args) {
        // Load test data from the "train" folder (expects subfolders 0-9)
        // For each digit, pick 10 random files.
        String testPath = "train";
        ArrayList<TrainingExample> testExamples = new ArrayList<>();
        
        for (int label = 0; label < NUM_CLASSES; label++) {
            File labelFolder = new File(testPath + File.separator + label);
            if (!labelFolder.exists() || !labelFolder.isDirectory()) {
                System.err.println("Folder not found: " + labelFolder.getAbsolutePath());
                continue;
            }
            File[] imageFiles = labelFolder.listFiles();
            if (imageFiles == null || imageFiles.length == 0) continue;
            
            // Shuffle files and pick 10 random files per label
            ArrayList<File> filesList = new ArrayList<>(Arrays.asList(imageFiles));
            Collections.shuffle(filesList);
            int count = Math.min(10, filesList.size());
            
            for (int i = 0; i < count; i++) {
                File imgFile = filesList.get(i);
                try {
                    BufferedImage img = ImageIO.read(imgFile);
                    if (img == null) continue;
                    float[] inputVector = imageToFloatArray(img);
                    testExamples.add(new TrainingExample(inputVector, label));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        
        System.out.println("Loaded " + testExamples.size() + " test examples.");
        
        // Deserialize the trained network from "trained_brain.ser"
        Brain brain = null;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream("trained_brain.ser"))) {
            brain = (Brain) ois.readObject();
            System.out.println("Trained network loaded from trained_brain.ser");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return;
        }
        
        // Evaluate the network on the test data, printing detailed information
        int correct = 0;
        for (int i = 0; i < testExamples.size(); i++) {
            TrainingExample ex = testExamples.get(i);
            
            // Get inference output (softmax probabilities)
            float[] output = brain.pass(ex.input);
            int predicted = argmax(output);
            
            // Also get raw logits (before softmax)
            float[] rawLogits = brain.forward(ex.input);
            
            if (predicted == ex.label) {
                correct++;
            }
            
            // Compute average pixel value for debugging
            float sumPixels = 0;
            for (float v : ex.input) {
                sumPixels += v;
            }
            float avgPixel = sumPixels / ex.input.length;
            
            System.out.println("Test Example " + i);
            System.out.println("Expected label: " + ex.label);
            System.out.println("Predicted label: " + predicted);
            System.out.print("Neuron percentages: ");
            for (int j = 0; j < output.length; j++) {
                System.out.printf("%.2f%% ", output[j] * 100);
            }
            System.out.println();
            System.out.println("Raw logits: " + Arrays.toString(rawLogits));
            System.out.printf("Average pixel value: %.4f%n", avgPixel);
            // Print a snippet of the input vector (first 20 values)
            System.out.println("Input sample (first 20 values): " + Arrays.toString(Arrays.copyOf(ex.input, 20)));
            System.out.println("--------------------------------------------------");
        }
        float accuracy = (float) correct / testExamples.size() * 100;
        System.out.println("Test Accuracy: " + accuracy + "%");
    }
    
    // Convert a BufferedImage (assumed 28x28 grayscale) to a normalized float array.
    // This method averages the R, G, and B channels.
    public static float[] imageToFloatArray(BufferedImage img) {
        float[] result = new float[INPUT_SIZE];
        int index = 0;
        for (int y = 0; y < IMAGE_HEIGHT; y++) {
            for (int x = 0; x < IMAGE_WIDTH; x++) {
                int rgb = img.getRGB(x, y);
                int r = (rgb >> 16) & 0xff;
                int g = (rgb >> 8) & 0xff;
                int b = rgb & 0xff;
                int gray = (r + g + b) / 3;
                result[index++] = gray / 255.0f;
            }
        }
        return result;
    }
    
    // Helper method: returns index of maximum value in an array.
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
