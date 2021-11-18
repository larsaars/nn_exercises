package mlp.matrix;

import java.util.List;

public class ArrayUtils {
    public static double[] flatten(double[][] array) {
        double[] flat = new double[array.length * array[0].length];
        for(int i = 0; i < array.length; i++)
            System.arraycopy(array[i], 0, flat, i * array[0].length, array[i].length);
        return flat;
    }

    public static double[][] unflatten(double[] flat, int rows, int cols) {
        double[][] array = new double[rows][cols];
        for(int i = 0; i < rows; i++)
            System.arraycopy(flat, i * cols, array[i], 0, cols);
        return array;
    }

    public static double[][] fromList(List<Double[]> list) {
        double [][] array = new double[list.size()][];

        for(int i = 0; i < list.size(); i++)
            array[i] = toPrimitive(list.get(i));

        return array;
    }

    public static double[] toPrimitive(Double[] array) {
        double[] result = new double[array.length];
        for(int i = 0; i < array.length; i++)
            result[i] = array[i];
        return result;
    }

    public static Double[] toObject(double[] array) {
        Double[] result = new Double[array.length];
        for(int i = 0; i < array.length; i++)
            result[i] = array[i];
        return result;
    }

    public static Double[] oneHotEncoding(int size, int index) {
        Double[] result = new Double[size];
        result[index] = 1.;
        return result;
    }
}
