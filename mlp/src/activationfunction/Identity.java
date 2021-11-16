package activationfunction;

import activationfunction.ActivationFunction;

public class Identity implements ActivationFunction {

    @Override
    public double activate(double input) {
        return input;
    }

    @Override
    public double derive(double input) {
        return 1;
    }
}
