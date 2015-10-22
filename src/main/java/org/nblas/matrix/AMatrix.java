package org.nblas.matrix;


abstract class AMatrix {
    private static final String abRowsMessage = "a Matrix rows is not b Matrix rows: ";
    private static final String abColumnsMesage = "a Matrix columns is not b Matrix columns: ";
    private static final String aResultRowsMessage = "a Matrix rows is not result Matrix rows: ";
    private static final String aResultColumnsMessage = "a Matrix columns is not result Matrix columns: ";
    protected int rows;
    protected int columns;
    protected int length;
    protected boolean notReleased = true;

    public boolean isColumnVector() {
        return columns == 1;
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public int getLength() {
        return length;
    }

    public boolean isRowVector() {
        return rows == 1;
    }

    public boolean isScalar() {
        return isRowVector() && isColumnVector();
    }

    public boolean isMatrix() {
        return !(isColumnVector() || isRowVector());
    }

    private static boolean isNotSameColumnSize(AMatrix a, AMatrix b) {
        return a.columns != b.columns;
    }

    private static boolean isNotSameRowSize(AMatrix a, AMatrix b) {
        return a.rows != b.rows;
    }

    protected static void isTransposeable(AMatrix a, AMatrix b) {
        if (!(a.rows == b.columns && a.columns == b.rows))
            throw new IllegalArgumentException("b Matrix does not fit transposed a in its size!");
    }

    protected static void isMmulable(AMatrix a, AMatrix b, AMatrix result) {
        if (a.columns != b.rows)
            throw new IllegalArgumentException("a Matrix columns is not b Matrix rows: " + a.columns + " != " + b.rows);
        checkRowSize(a, result, " a Matrix rows is not result Matrix rows: " + a.rows + " !=  " + result.rows);
        checkColumnSize(b, result, " b Matrix columns is not result Matrix columns: " + b.columns + " !=  " + result.columns);
    }


    protected static void checkSameSize(AMatrix a, AMatrix b, AMatrix result) {
        checkSizes(a, b, abRowsMessage + a.rows + " !=  " + b.rows, abColumnsMesage + a.columns + " !=  " + b.columns);
        checkSizes(a, result, aResultRowsMessage + a.rows + " !=  " + result.rows, aResultColumnsMessage + a.columns + " !=  " + result.columns);
    }


    protected static void checkSameSize(AMatrix a, AMatrix result) {
        checkSizes(a, result, aResultRowsMessage + a.rows + " !=  " + result.rows, aResultColumnsMessage + a.columns + " !=  " + result.columns);
    }

    protected static void checkColumnVectorSize(AMatrix a, AMatrix b, AMatrix result) {
        if (!b.isColumnVector()) throw new IllegalArgumentException("b Matrix is not a column Vector");
        checkRowSize(a, b, abRowsMessage + a.rows + " !=  " + b.rows);
        checkSizes(a, result, aResultRowsMessage + a.rows + " !=  " + result.rows, aResultColumnsMessage + a.columns + " !=  " + result.columns);
    }

    protected static void checkRowVectorSize(AMatrix a, AMatrix b, AMatrix result) {
        if (!b.isRowVector()) throw new IllegalArgumentException("b Matrix is not a row Vector");
        checkColumnSize(a, b, abColumnsMesage + a.columns + " !=  " + b.columns);
        checkSizes(a, result, aResultRowsMessage + a.rows + " !=  " + result.rows, aResultColumnsMessage + a.columns + " !=  " + result.columns);
    }

    private static void checkSizes(AMatrix a, AMatrix result, String s, String s2) {
        checkRowSize(a, result, s);
        checkColumnSize(a, result, s2);
    }

    private static void checkColumnSize(AMatrix a, AMatrix result, String s2) {
        if (isNotSameColumnSize(a, result))
            throw new IllegalArgumentException(s2);
    }

    private static void checkRowSize(AMatrix a, AMatrix result, String s) {
        if (isNotSameRowSize(a, result))
            throw new IllegalArgumentException(s);
    }

    // JAVA GETTER

    public float[] toArray() {
        float[] values = new float[getRows() * getColumns()];
        getColumnWiseOn(values);
        return values;
    }

    public abstract void getColumnWiseOn(float[] values);

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
