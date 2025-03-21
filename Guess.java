import java.io.File;
import java.io.ObjectInputStream;
import java.io.FileInputStream;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.util.Arrays;
/**
 * A simple Guess class that allows you to use the neural network.
 * 
 * @author Bashar Zain
 * @version 1.0
 */
public class Guess {
    public static final int IMAGE_WIDTH = 28;
    public static final int IMAGE_HEIGHT = 28;
    public static final int INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
    public static final int NUM_CLASSES = 10;

    public static void main(String[] args) {
        try {
            // Load the trained network
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream("trained_brain.ser"));
            Brain brain = (Brain) ois.readObject();
            ois.close();
            System.out.println("Trained network loaded from trained_brain.ser");
            
            // Load the image "image.png"
            File imgFile = new File("image.png");
            BufferedImage img = ImageIO.read(imgFile);
            if (img == null) {
                System.err.println("Could not load image.png");
                return;
            }
            
            // Convert image to float array
            float[] input = imageToFloatArray(img);
            
            // Compute average pixel value
            float sum = 0;
            for (float v : input) {
                sum += v;
            }
            float avgPixel = sum / input.length;
            
            // Get raw logits (using forward pass without softmax) and softmax output (for inference)
            float[] rawLogits = brain.forward(input);
            float[] softmaxOutput = brain.pass(input);
            
            // Determine predicted label
            int predicted = argmax(softmaxOutput);
            
            // Print the results
            System.out.println("--------------------------------------------------");
            System.out.println("Image.png Result");
            System.out.println("Predicted label: " + predicted);
            System.out.print("Neuron percentages: ");
            for (int i = 0; i < softmaxOutput.length; i++) {
                System.out.printf("%.2f%% ", softmaxOutput[i] * 100);
            }
            System.out.println();
            System.out.println("Raw logits: " + Arrays.toString(rawLogits));
            System.out.printf("Average pixel value: %.4f%n", avgPixel);
            System.out.println("Input sample (first 20 values): " + Arrays.toString(Arrays.copyOf(input, 20)));
            System.out.println("--------------------------------------------------");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    // Converts a BufferedImage (assumed 28x28) to a normalized float array.
    // Uses the average of R, G, and B channels.
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
    
    // Returns the index of the maximum value in the array.
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
