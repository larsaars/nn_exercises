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
            {Matrix.NEAR_ZERO}, {1}, {1}, {Matrix.NEAR_ZERO}
    };

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(new int[]{2, 10, 10, 10, 1}, 0.01, ActivationFunctions.RELU);

        List<Double> output;
        nn.fit(X, Y, 500000);
        for (double[] d : X) {
            output = nn.predict(d);
            System.out.println(output.toString());
        }
    }
}