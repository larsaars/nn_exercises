import activationfunction.ActivationFunction;
import activationfunction.ActivationFunctions;

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
        NeuralNetwork nn = new NeuralNetwork(2, 10, 2, 0.01, ActivationFunctions.RELU);

        List<Double> output;
        nn.fit(X, Y, 50000);
        for (double[] d : X) {
            output = nn.predict(d);
            System.out.println(output.toString());
        }
    }
}