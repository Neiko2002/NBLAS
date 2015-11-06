package org.nblas.generic;

/**
 * Eine Native Matrix alloziert Ressourcen die wieder frei gegeben werden müssen.
 * 
 * TODO das zu einen Interface machen und dafür ANativeCLMatrix und ANativeCUDAMatrix inder die internen rows gespeichert sind
 * 
 * @author Nico
 *
 */
public abstract class ANativeFloatMatrix extends ANativeMatrix {
   
	protected abstract void getColumnWiseOn(float[] values);

    // JAVA GETTER

    public float[] toArray() {
        float[] values = new float[getRows() * getColumns()];
        getColumnWiseOn(values);
        return values;
    }

	public float[][] toArray2() {
        float[][] matrix = new float[rows][columns];
        float[] array = toArray();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = array[i + j * rows];
            }
        }
        return matrix;
    }

    // PRINT

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        float[][] matrix = toArray2();
        builder.append("[");
        for (int i = 0; i < rows - 1; i++) {
            for (int j = 0; j < columns - 1; j++) {
                builder.append(String.format("%.6f", matrix[i][j]));
                builder.append(", ");
            }

            builder.append(String.format("%.6f", matrix[i][columns - 1]));
            builder.append("; ");
        }
        for (int j = 0; j < columns - 1; j++) {
            builder.append(String.format("%.6f", matrix[rows - 1][j]));
            builder.append(", ");
        }
        builder.append(String.format("%.6f", matrix[rows - 1][columns - 1]));
        builder.append("]");

        return builder.toString();
    }

    public String toString2() {
        StringBuilder builder = new StringBuilder();
        float[][] matrix = toArray2();
        for (int i = 0; i < rows - 1; i++) {
            builder.append("[");
            for (int j = 0; j < columns - 1; j++) {
                builder.append(String.format("%.6f", matrix[i][j]));
                builder.append(", ");
            }

            builder.append(String.format("%.6f", matrix[i][columns - 1]));
            builder.append("]\n");
        }

        builder.append("[");
        for (int j = 0; j < columns - 1; j++) {
            builder.append(String.format("%.6f", matrix[rows - 1][j]));
            builder.append(", ");
        }
        builder.append(String.format("%.6f", matrix[rows - 1][columns - 1]));
        builder.append("]\n");

        return builder.toString();
    }
}
