import activationfunction.ActivationFunctions;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class Driver {

    static double[][] X = {
            {0, 0},
            {1, 0},
            {0, 1},
            {1, 1}
    };
    static double[][] Y = {
            {0}, {1}, {1}, {0}
    };

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(new int[]{2, 255, 1}, 0.01, ActivationFunctions.RELU);

        List<Double> output;
        double[] loss = nn.fit(X, Y, 50000);
        for (double[] d : X) {
            output = nn.predict(d);
            System.out.println(output.toString());
        }

        // write loss to file
        File file = new File("loss.txt");
        StringBuilder sb = new StringBuilder();
        for (double d : loss) {
            sb.append(d).append(" ");
        }

        try {
            FileOutputStream fos = new FileOutputStream(file);
            fos.write(sb.toString().getBytes(StandardCharsets.UTF_8));
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}