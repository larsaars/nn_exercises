import java.util.List;

public class Driver {

    static double[][] X = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    static double[][] Y = {{0}, {1}, {1}, {0}};

    public static void main(String[] args) {

        NeuralNetwork nn = new NeuralNetwork(2, 10, 1, 0.01, false);

        List<Double> output;
        nn.fit(X, Y, 500000, 1);
        for (double[] d : X) {
            output = nn.predict(d);
            System.out.println(output.toString());
        }

    }

}

