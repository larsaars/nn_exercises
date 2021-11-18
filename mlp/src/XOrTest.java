import mlp.utils.NNUtils;
import mlp.NeuralNetwork;
import mlp.activationfunction.ActivationFunctions;

import java.util.List;

public class XOrTest {

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
        double[] loss = nn.fit(X, Y, 0.01, 20);
        for (double[] d : X) {
            output = nn.predict(d);
            System.out.println(output.toString());
        }

        NNUtils.save(nn, loss);
    }
}