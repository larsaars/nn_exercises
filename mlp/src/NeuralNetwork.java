import activationfunction.ActivationFunction;

import java.util.List;
import java.util.Random;


public class NeuralNetwork {

    public Matrix[] weights, biases;
    public ActivationFunction[] activationFunctions;
    public int[] layers;

    public double learningRate;

    private final Random random = new Random();

    public NeuralNetwork(int[] layers, double learningRate, ActivationFunction activationFunction) {
        this.learningRate = learningRate;
        this.layers = layers;
        int layersSize = layers.length;

        weights = new Matrix[layersSize - 1];
        biases = new Matrix[layersSize - 1];
        activationFunctions = new ActivationFunction[layersSize - 1];

        for (int i = 0; i < layersSize - 1; i++) {
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
        Matrix[] processing = new Matrix[layers.length];
        processing[0] = Matrix.fromArray(X);

        for(int i = 0; i < processing.length - 1; i++)
            processing[i + 1] = Matrix.dot(weights[i], processing[i])
                    .add(biases[i])
                    .apply(activationFunctions[i], false);

        Matrix inputWeightsBefore = Matrix.fromArray(X);
        Matrix errorBefore = Matrix.fromArray(Y)
                .subtract(processing[processing.length - 1]);

        for (int i = layers.length - 2; i >= 0; i--) {
            Matrix error = i == layers.length - 2 ? errorBefore : Matrix.dot(inputWeightsBefore, errorBefore);

            Matrix gradient = Matrix.c(processing[i + 1])
                    .apply(activationFunctions[i], true)
                    .multiply(error)
                    .multiply(learningRate);

            Matrix l_T = Matrix.transpose(processing[i]);
            Matrix l_delta = Matrix.dot(gradient, l_T);

            weights[i].add(l_delta);
            biases[i].add(gradient);

            if (i != 0) {
                inputWeightsBefore = Matrix.transpose(weights[i]);
                errorBefore = error;
            }
        }
    }
}

