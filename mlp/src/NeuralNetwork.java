import activationfunction.ActivationFunction;

import java.util.List;
import java.util.Random;


public class NeuralNetwork {

    public Matrix[] weights, biases;
    public ActivationFunction[] activationFunctions;
    public int[] layers;

    public double learningRate;

    private Random random = new Random();

    public NeuralNetwork(int[] layers, double learningRate, ActivationFunction activationFunction) {
        this.learningRate = learningRate;
        this.layers = layers;
        int layersSize = layers.length;

        weights = new Matrix[layersSize - 1];
        biases = new Matrix[layersSize - 1];
        activationFunctions = new ActivationFunction[layersSize - 1];

        for (int i = 0; i < 2; i++) {
            activationFunctions[i] = activationFunction;
            weights[i] = new Matrix(layers[i + 1], layers[i], true);
            biases[i] = new Matrix(layers[i + 1], 1, true);
        }
    }

    public List<Double> predict(double[] X) {
        Matrix input = Matrix.fromArray(X);

        for (int i = 0; i < weights.length; i++)
            input = Matrix.dot(weights[i], input)
                    .add(biases[i])
                    .apply(activationFunctions[i], false);

        return input.toArray();
    }

    public void fit(double[][] X, double[][] Y, int epochs) {
        // train in n epochs with stochastic gradient descent
        for (int i = 0; i < epochs; i++) {
            int sampleN = random.nextInt(X.length);
            backprop(X[sampleN], Y[sampleN]);
        }
    }

    public void backprop(double[] X, double[] Y) {
        Matrix input = Matrix.fromArray(X);
        Matrix hidden = Matrix.dot(weights[0], input)
                .add(biases[0])
                .apply(activationFunctions[0], false);

        Matrix output = Matrix.dot(weights[1], hidden)
                .add(biases[1])
                .apply(activationFunctions[0], false);

        Matrix error = Matrix.fromArray(Y).subtract(output);

        Matrix gradient = Matrix.c(output)
                .apply(activationFunctions[0], true)
                .multiply(error)
                .multiply(learningRate);

        Matrix hidden_T = Matrix.transpose(hidden);
        Matrix who_delta = Matrix.dot(gradient, hidden_T); // 1x10

        weights[1].add(who_delta);
        biases[1].add(gradient);


        Matrix who_T = Matrix.transpose(weights[1]); // 10x1
        Matrix hidden_error = Matrix.dot(who_T, error); // 10x1

        Matrix h_gradient = Matrix.c(hidden) // 10x1
                .apply(activationFunctions[0], true)
                .multiply(hidden_error)
                .multiply(learningRate);

        Matrix i_T = Matrix.transpose(input); // 1x2
        Matrix wih_delta = Matrix.dot(h_gradient, i_T); // 10x2

        weights[0].add(wih_delta);
        biases[0].add(h_gradient);

        for(int i = layers.length - 1; i >= 0; i--) {
            Matrix who_T1 = Matrix.transpose(weights[1]);
            Matrix hidden_error = Matrix.dot(who_T, error);

            Matrix gradient = Matrix.c(hidden)
                    .apply(activationFunctions[0], true)
                    .multiply(hidden_error)
                    .multiply(learningRate);

            Matrix i_T = Matrix.transpose(input);
            Matrix wih_delta = Matrix.dot(h_gradient, i_T);

            weights[i].add(wih_delta);
            biases[i].add(h_gradient);
        }
    }
}

