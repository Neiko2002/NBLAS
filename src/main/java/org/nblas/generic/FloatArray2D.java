package org.nblas.generic;

/**
 * @author Nico
 *
 */
public interface FloatArray2D {
   
	void getColumnWiseOn(float[] values);
	int getRows();
	int getColumns();
	String toString();
	
    // JAVA GETTER

    public default float[] toArray() {
        float[] values = new float[getRows() * getColumns()];
        getColumnWiseOn(values);
        return values;
    }

	public default float[][] toArray2() {
        float[][] matrix = new float[getRows()][getColumns()];
        float[] array = toArray();
        for (int i = 0; i < getRows(); i++) {
            for (int j = 0; j < getColumns(); j++) {
                matrix[i][j] = array[i + j * getRows()];
            }
        }
        return matrix;
    }

    // PRINT

    public default String toString1D() {
        StringBuilder builder = new StringBuilder();
        float[][] matrix = toArray2();
        builder.append("[");
        for (int i = 0; i < getRows() - 1; i++) {
            for (int j = 0; j < getColumns() - 1; j++) {
                builder.append(String.format("%.6f", matrix[i][j]));
                builder.append(", ");
            }

            builder.append(String.format("%.6f", matrix[i][getColumns() - 1]));
            builder.append("; ");
        }
        for (int j = 0; j < getColumns() - 1; j++) {
            builder.append(String.format("%.6f", matrix[getRows() - 1][j]));
            builder.append(", ");
        }
        builder.append(String.format("%.6f", matrix[getRows() - 1][getColumns() - 1]));
        builder.append("]");

        return builder.toString();
    }

    public default String toString2D() {
        StringBuilder builder = new StringBuilder();
        float[][] matrix = toArray2();
        for (int i = 0; i < getRows() - 1; i++) {
            builder.append("[");
            for (int j = 0; j < getColumns() - 1; j++) {
                builder.append(String.format("%.6f", matrix[i][j]));
                builder.append(", ");
            }

            builder.append(String.format("%.6f", matrix[i][getColumns() - 1]));
            builder.append("]\n");
        }

        builder.append("[");
        for (int j = 0; j < getColumns() - 1; j++) {
            builder.append(String.format("%.6f", matrix[getRows() - 1][j]));
            builder.append(", ");
        }
        builder.append(String.format("%.6f", matrix[getRows() - 1][getColumns() - 1]));
        builder.append("]\n");

        return builder.toString();
    }
}
