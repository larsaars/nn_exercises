import activationfunction.ActivationFunction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class Matrix {
    double[][] data;
    int rows, cols;

    public Matrix(int rows, int cols, boolean random) {
        this.rows = rows;
        this.cols = cols;
        data = new double[rows][cols];

        if (random) {
            // assign random values between -1 and 1 to matrix
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    data[i][j] = Math.random() * 2 - 1;
        }
    }

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Matrix(Matrix matrix) {
        this.rows = matrix.rows;
        this.cols = matrix.cols;

        data = Arrays.stream(matrix.data).map(double[]::clone).toArray(double[][]::new);
    }

    public Matrix add(double scalar) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] += scalar;

        return this;
    }

    public Matrix add(Matrix m) {
        if (cols != m.cols || rows != m.rows) {
            System.out.println("shape mismatch");
            return null;
        }

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] += m.data[i][j];

        return this;
    }

    public Matrix subtract(Matrix m) {
        if (cols != m.cols || rows != m.rows) {
            System.out.println("shape mismatch");
            return null;
        }

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] -= m.data[i][j];

        return this;
    }


    public Matrix multiply(double scalar) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] *= scalar;

        return this;
    }

    public Matrix squared() {
        return multiply(this);
    }

    public Matrix abs() {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = Math.abs(data[i][j]);

        return this;
    }

    // elementwise multiplication
    public Matrix multiply(Matrix m) {
        if (cols != m.cols || rows != m.rows) {
            System.out.println("shape mismatch");
            return null;
        }

        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                data[i][j] *= m.data[i][j];
            }
        }

        return this;
    }


    public Matrix apply(ActivationFunction activationFunction, boolean derive) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = derive ? activationFunction.derive(data[i][j]) : activationFunction.activate(data[i][j]);

        return this;
    }

    /*
     * copy a matrix
     */
    public static Matrix c(Matrix m) {
        return new Matrix(m);
    }


    /*
     * static matrix options return copies
     */

    public static Matrix transpose(Matrix m) {
        Matrix temp = new Matrix(m.cols, m.rows);
        for (int i = 0; i < m.rows; i++)
            for (int j = 0; j < m.cols; j++)
                temp.data[j][i] = m.data[i][j];

        return temp;
    }

    public static Matrix dot(Matrix a, Matrix b) {
        if (a.cols != b.rows) {
            System.out.println("shape mismatch");
            return null;
        }

        Matrix temp = new Matrix(a.rows, b.cols);
        for (int i = 0; i < temp.rows; i++)
            for (int j = 0; j < temp.cols; j++)
                for (int k = 0; k < a.cols; k++)
                    temp.data[i][j] += a.data[i][k] * b.data[k][j];

        return temp;
    }

    public int[] shape() {
        return new int[]{rows, cols};
    }

    public void printShape() {
        System.out.println(Arrays.toString(shape()));
    }

    /*
     * functions for creating from array
     */
    public static Matrix fromArray(double[] x) {
        Matrix temp = new Matrix(x.length, 1);
        for (int i = 0; i < x.length; i++)
            temp.data[i][0] = x[i];
        return temp;

    }

    public List<Double> toArray() {
        List<Double> temp = new ArrayList<>();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                temp.add(data[i][j]);
            }
        }
        return temp;
    }

    /*
    * helpful utility functions
    */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sb.append(data[i][j]).append(" ");
            }
            sb.append('\n');
        }

        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Matrix)) return false;
        Matrix matrix = (Matrix) o;
        return rows == matrix.rows && cols == matrix.cols && Arrays.deepEquals(data, matrix.data);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(rows, cols);
        result = 31 * result + Arrays.deepHashCode(data);
        return result;
    }

    public void print() {
        System.out.println(this);
    }
}
