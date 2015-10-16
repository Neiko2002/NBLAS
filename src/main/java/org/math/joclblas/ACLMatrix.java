package org.math.joclblas;

import org.jocl.CL;
import org.jocl.cl_mem;

import java.util.Random;

/**
 * Created by Moritz on 4/12/2015.
 */
class ACLMatrix {
    protected static final Random RANDOM = new Random();
    private static final String abRowsMessage = "a Matrix rows is not b Matrix rows: ";
    ;
    public static final String abcolumnsMesage = "a Matrix columns is not b Matrix columns: ";
    public static final String aResultRowsMessage = "a Matrix rows is not result Matrix rows: ";
    public static final String aResultColumnsMessage = "a Matrix columns is not result Matrix columns: ";
    protected int rows;
    protected int columns;
    protected int length;
    protected int clRows;
    protected int clColumns;
    protected int clLength;
    protected cl_mem buffer;

    private boolean notReleased = true;

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

    private static boolean isNotSameColumnSize(CLFloatMatrix a, CLFloatMatrix b) {
        return a.columns != b.columns;
    }

    private static boolean isNotSameRowSize(CLFloatMatrix a, CLFloatMatrix b) {
        return a.rows != b.rows;
    }

    protected static void isTransposeable(CLFloatMatrix a, CLFloatMatrix b) {
        if (!(a.rows == b.columns && a.columns == b.rows))
            throw new IllegalArgumentException("b Matrix does not fit transposed a in its size!");
    }

    protected static void isMmulable(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        if (a.columns != b.rows)
            throw new IllegalArgumentException("a Matrix columns is not b Matrix rows: " + a.columns + " != " + b.rows);
        checkRowSize(a, result, " a Matrix rows is not result Matrix rows: " + a.rows + " !=  " + result.rows);
        checkColumnSize(b, result, " b Matrix columns is not result Matrix columns: " + b.columns + " !=  " + result.columns);
    }


    protected static void checkSameSize(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkSizes(a, b, abRowsMessage + a.rows + " !=  " + b.rows, abcolumnsMesage + a.columns + " !=  " + b.columns);
        checkSizes(a, result, aResultRowsMessage + a.rows + " !=  " + result.rows, aResultColumnsMessage + a.columns + " !=  " + result.columns);
    }


    protected static void checkSameSize(CLFloatMatrix a, CLFloatMatrix result) {
        checkSizes(a, result, aResultRowsMessage + a.rows + " !=  " + result.rows, aResultColumnsMessage + a.columns + " !=  " + result.columns);
    }

    protected static void checkColumnVectorSize(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        if (!b.isColumnVector()) throw new IllegalArgumentException("b Matrix is not a column Vector");
        checkRowSize(a, b, abRowsMessage + a.rows + " !=  " + b.rows);
        checkSizes(a, result, aResultRowsMessage + a.rows + " !=  " + result.rows, aResultColumnsMessage + a.columns + " !=  " + result.columns);
    }

    protected static void checkRowVectorSize(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        if (!b.isRowVector()) throw new IllegalArgumentException("b Matrix is not a row Vector");
        checkColumnSize(a, b, abcolumnsMesage + a.columns + " !=  " + b.columns);
        checkSizes(a, result, aResultRowsMessage + a.rows + " !=  " + result.rows, aResultColumnsMessage + a.columns + " !=  " + result.columns);
    }

    private static void checkSizes(CLFloatMatrix a, CLFloatMatrix result, String s, String s2) {
        checkRowSize(a, result, s);
        checkColumnSize(a, result, s2);
    }

    private static void checkColumnSize(CLFloatMatrix a, CLFloatMatrix result, String s2) {
        if (isNotSameColumnSize(a, result))
            throw new IllegalArgumentException(s2);
    }

    private static void checkRowSize(CLFloatMatrix a, CLFloatMatrix result, String s) {
        if (isNotSameRowSize(a, result))
            throw new IllegalArgumentException(s);
    }

    public void release() {
        if (notReleased) {
            CL.clReleaseMemObject(buffer);
            notReleased = false;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        release();
        super.finalize();
    }
}
