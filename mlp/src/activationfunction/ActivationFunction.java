package activationfunction;

public interface ActivationFunction {
    double activate(double input);

    double derive(double input);
}