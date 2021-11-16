import activationfunction.ActivationFunction;
import activationfunction.ActivationFunctions;

import java.util.List;
import java.util.Random;


public class NeuralNetwork {

    Matrix weights_ih, weights_ho, bias_h, bias_o;
    double l_rate = 0.01;
    ActivationFunction activationFunction;

    Random random = new Random();

    public NeuralNetwork(int i, int h, int o, double l_rate, ActivationFunction activationFunction) {
        weights_ih = new Matrix(h, i, false);
        weights_ho = new Matrix(o, h, false);

        bias_h = new Matrix(h, 1, false);
        bias_o = new Matrix(o, 1, false);

        this.l_rate = l_rate;
        this.activationFunction = activationFunction;
    }

    public List<Double> predict(double[] X) {
        Matrix input = Matrix.fromArray(X);
        Matrix hidden = Matrix.dot(weights_ih, input);
        hidden.add(bias_h);
        hidden.apply(activationFunction, false);

        Matrix output = Matrix.dot(weights_ho, hidden);
        output.add(bias_o);
        output.apply(activationFunction, false);

        return output.toArray();
    }

    public void fit(double[][] X, double[][] Y, int epochs) {
        for (int i = 0; i < epochs; i++) {
            int sampleN = (int) (Math.random() * X.length);
            this.train(X[sampleN], Y[sampleN], false);
        }
    }

    public void fit(double[][] X, double[][] Y, int epochs, int verbose) {
        switch (verbose) {

            case 0: {
                System.out.println("Staring training with " + epochs + " epochs");
                long start = System.currentTimeMillis();
                for (int i = 0; i < epochs; i++) {
                    int sampleN = random.nextInt(X.length);
                    this.train(X[sampleN], Y[sampleN], i + 1 == epochs);
                }
                long end = System.currentTimeMillis();
                long elapsedTime = end - start;
                System.out.println("Training took : " + (elapsedTime / 1000) + "s");

                break;
            }

            case 1: {
                System.out.println("Staring training with " + epochs + " epochs");
                long start = System.currentTimeMillis();
                for (int i = 0; i < epochs; i++) {
                    System.out.println("Epoch: " + (i + 1));
                    int sampleN = random.nextInt(X.length);
                    this.train(X[sampleN], Y[sampleN], true);
                }
                long end = System.currentTimeMillis();
                long elapsedTime = end - start;
                System.out.println("Training took : " + (elapsedTime / 1000) + "s");

                break;
            }
        }

    }

    public void train(double[] X, double[] Y, boolean showLoss) {
        Matrix input = Matrix.fromArray(X);
        Matrix hidden = Matrix.dot(weights_ih, input);
        hidden.add(bias_h);
        hidden.apply(activationFunction, false);

        Matrix output = Matrix.dot(weights_ho, hidden);
        output.add(bias_o);
        output.apply(activationFunction, false);

        Matrix target = Matrix.fromArray(Y);

        Matrix error = Matrix.c(target).subtract(output);
        Matrix gradient = Matrix.c(output)
                .apply(activationFunction, true)
                .multiply(error)
                .multiply(l_rate);

        if (showLoss)
            printLoss(error);

        Matrix hidden_T = Matrix.transpose(hidden);
        Matrix who_delta = Matrix.dot(gradient, hidden_T);

        weights_ho.add(who_delta);
        bias_o.add(gradient);

        Matrix who_T = Matrix.transpose(weights_ho);
        Matrix hidden_errors = Matrix.dot(who_T, error);

        Matrix h_gradient = Matrix.c(hidden)
                .apply(activationFunction, true)
                .multiply(hidden_errors)
                .multiply(l_rate);

        Matrix i_T = Matrix.transpose(input);
        Matrix wih_delta = Matrix.dot(h_gradient, i_T);

        weights_ih.add(wih_delta);
        bias_h.add(h_gradient);

    }

    private void printLoss(Matrix error) {
        double avg = 0.0;

        for (int i = 0; i < error.rows; i++) {
            for (int j = 0; j < error.cols; j++) {
                avg += error.data[i][j];
            }
        }

        System.out.print("Average Error: " + avg + "\n");
    }

}

