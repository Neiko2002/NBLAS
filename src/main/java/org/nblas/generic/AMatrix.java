package org.nblas.generic;

import org.nblas.cl.CLMatrix;

public abstract class AMatrix {
	
	private static final String aNotScalar = "matrix 'a' is not scalar: ";
    private static final String abRowsMessage = "matrix 'a' rows is not matrix 'b' rows: ";
    private static final String abColumnsMesage = "matrix 'a' columns is not matrix 'b' columns: ";
    private static final String aResultRowsMessage = "matrix 'a' rows is not matrix 'result' rows: ";
    private static final String aResultColumnsMessage = "matrix 'a' columns is not matrix 'result' columns: ";
    
    protected int rows;
    protected int columns;
    protected int length;
	protected boolean released = false;

    public AMatrix(int rows, int columns) {
    	this.columns = columns;
		this.rows = rows;
		this.length = columns*rows;
	}
    
    public int getLength() {
		return length;
	}
    
    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }
    public boolean isColumnVector() {
        return getColumns() == 1;
    }

    public boolean isRowVector() {
        return getRows() == 1;
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
    
    public abstract void free();  
    

    private static boolean isNotSameColumnSize(AMatrix a, AMatrix b) {
        return a.getColumns() != b.getColumns();
    }

    private static boolean isNotSameRowSize(AMatrix a, AMatrix b) {
        return a.getRows() != b.getRows();
    }

    protected static void isTransposeable(AMatrix a, AMatrix b) {
        if (!(a.getRows() == b.getColumns() && a.getColumns() == b.getRows()))
            throw new IllegalArgumentException("b Matrix does not fit transposed a in its size!");
    }

    protected static void isMmulable(AMatrix a, AMatrix b, AMatrix result) {
        if (a.getColumns() != b.getRows())
            throw new IllegalArgumentException("a Matrix columns is not b Matrix rows: " + a.getColumns() + " != " + b.getRows());
        checkRowSize(a, result, " a Matrix rows is not result Matrix rows: " + a.getRows() + " !=  " + result.getRows());
        checkColumnSize(b, result, " b Matrix columns is not result Matrix columns: " + b.getColumns() + " !=  " + result.getColumns());
    }

    protected static void checkSameSize(AMatrix a, AMatrix b, AMatrix result) {
        checkSizes(a, b, abRowsMessage + a.getRows() + " !=  " + b.getRows(), abColumnsMesage + a.getColumns() + " !=  " + b.getColumns());
        checkSizes(a, result, aResultRowsMessage + a.getRows() + " !=  " + result.getRows(), aResultColumnsMessage + a.getColumns() + " !=  " + result.getColumns());
    }
	
    protected static void checkScalarSize(CLMatrix scalar) {
		if (scalar.getRows() != 1 && scalar.getColumns() != 1)
			throw new IllegalArgumentException(aNotScalar + scalar.getRows() + "," + scalar.getColumns());
	}

    protected static void checkSameSize(AMatrix a, AMatrix result) {
        checkSizes(a, result, aResultRowsMessage + a.getRows() + " !=  " + result.getRows(), aResultColumnsMessage + a.getColumns() + " !=  " + result.getColumns());
    }

    protected static void checkColumnVectorSize(AMatrix a, AMatrix b, AMatrix result) {
        if (!b.isColumnVector()) throw new IllegalArgumentException("b Matrix is not a column Vector");
        checkRowSize(a, b, abRowsMessage + a.getRows() + " !=  " + b.getRows());
        checkSizes(a, result, aResultRowsMessage + a.getRows() + " !=  " + result.getRows(), aResultColumnsMessage + a.getColumns() + " !=  " + result.getColumns());
    }

    protected static void checkRowVectorSize(AMatrix a, AMatrix b, AMatrix result) {
        if (!b.isRowVector()) throw new IllegalArgumentException("b Matrix is not a row Vector");
        checkColumnSize(a, b, abColumnsMesage + a.getColumns() + " !=  " + b.getColumns());
        checkSizes(a, result, aResultRowsMessage + a.getRows() + " !=  " + result.getRows(), aResultColumnsMessage + a.getColumns() + " !=  " + result.getColumns());
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
}
