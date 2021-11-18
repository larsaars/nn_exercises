package mlp.matrix;

public class ArrayUtils {
    public static double[] flatten(double[][] array) {
        double[] flat = new double[array.length * array[0].length];
        for(int i = 0; i < array.length; i++)
            System.arraycopy(array[i], 0, flat, i * array[0].length, array[i].length);
        return flat;
    }
}
