package org.nblas.generic;

/**
 * Eine Native Matrix alloziert Ressourcen die wieder frei gegeben werden müssen.
 * 
 * @author Nico
 *
 */
public abstract class NativeMatrix {
    private static final String abRowsMessage = "a Matrix rows is not b Matrix rows: ";
    private static final String abColumnsMesage = "a Matrix columns is not b Matrix columns: ";
    private static final String aResultRowsMessage = "a Matrix rows is not result Matrix rows: ";
    private static final String aResultColumnsMessage = "a Matrix columns is not result Matrix columns: ";
    
    protected int rows;
    protected int columns;
    protected int length;
    protected boolean released = true;

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public int getLength() {
        return length;
    }

    public boolean isColumnVector() {
        return columns == 1;
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
    
    public boolean isReleased() {
        return released;
    }

    private static boolean isNotSameColumnSize(NativeMatrix a, NativeMatrix b) {
        return a.columns != b.columns;
    }

    private static boolean isNotSameRowSize(NativeMatrix a, NativeMatrix b) {
        return a.rows != b.rows;
    }

    protected static void isTransposeable(NativeMatrix a, NativeMatrix b) {
        if (!(a.rows == b.columns && a.columns == b.rows))
            throw new IllegalArgumentException("b Matrix does not fit transposed a in its size!");
    }

    protected static void isMmulable(NativeMatrix a, NativeMatrix b, NativeMatrix result) {
        if (a.columns != b.rows)
            throw new IllegalArgumentException("a Matrix columns is not b Matrix rows: " + a.columns + " != " + b.rows);
        checkRowSize(a, result, " a Matrix rows is not result Matrix rows: " + a.rows + " !=  " + result.rows);
        checkColumnSize(b, result, " b Matrix columns is not result Matrix columns: " + b.columns + " !=  " + result.columns);
    }


    protected static void checkSameSize(NativeMatrix a, NativeMatrix b, NativeMatrix result) {
        checkSizes(a, b, abRowsMessage + a.rows + " !=  " + b.rows, abColumnsMesage + a.columns + " !=  " + b.columns);
        checkSizes(a, result, aResultRowsMessage + a.rows + " !=  " + result.rows, aResultColumnsMessage + a.columns + " !=  " + result.columns);
    }


    protected static void checkSameSize(NativeMatrix a, NativeMatrix result) {
        checkSizes(a, result, aResultRowsMessage + a.rows + " !=  " + result.rows, aResultColumnsMessage + a.columns + " !=  " + result.columns);
    }

    protected static void checkColumnVectorSize(NativeMatrix a, NativeMatrix b, NativeMatrix result) {
        if (!b.isColumnVector()) throw new IllegalArgumentException("b Matrix is not a column Vector");
        checkRowSize(a, b, abRowsMessage + a.rows + " !=  " + b.rows);
        checkSizes(a, result, aResultRowsMessage + a.rows + " !=  " + result.rows, aResultColumnsMessage + a.columns + " !=  " + result.columns);
    }

    protected static void checkRowVectorSize(NativeMatrix a, NativeMatrix b, NativeMatrix result) {
        if (!b.isRowVector()) throw new IllegalArgumentException("b Matrix is not a row Vector");
        checkColumnSize(a, b, abColumnsMesage + a.columns + " !=  " + b.columns);
        checkSizes(a, result, aResultRowsMessage + a.rows + " !=  " + result.rows, aResultColumnsMessage + a.columns + " !=  " + result.columns);
    }

    private static void checkSizes(NativeMatrix a, NativeMatrix result, String s, String s2) {
        checkRowSize(a, result, s);
        checkColumnSize(a, result, s2);
    }

    private static void checkColumnSize(NativeMatrix a, NativeMatrix result, String s2) {
        if (isNotSameColumnSize(a, result))
            throw new IllegalArgumentException(s2);
    }

    private static void checkRowSize(NativeMatrix a, NativeMatrix result, String s) {
        if (isNotSameRowSize(a, result))
            throw new IllegalArgumentException(s);
    }    
}
