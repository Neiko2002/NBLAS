package org.nblas.cl;


import java.util.Optional;

import org.jblas.util.Random;
import org.jocl.cl_kernel;
import org.nblas.cl.blas.CLLevel1;
import org.nblas.generic.FloatArray2D;
import org.nblas.generic.Subprogram;

/**
 * 
 * @author Nico
 *
 */
public class CLFloatMatrix extends ANativeCLMatrix implements FloatArray2D {


    public CLFloatMatrix(int rows, int columns, float... values) {
       super(rows, columns, values);
       
		if (values.length == 0) {
			this.dataPointer = CORE.mallocSinglePrecision(this.clLength);
			setZero();
		} else {
			if (rows * columns != values.length)
				throw new IllegalArgumentException(
						"rows times columns " + (rows * columns) + " != " + "data length = " + values.length);

			float[] clValues = getCLMatrix(rows, columns, values);
			this.dataPointer = CORE.malloc(clValues);
		}
    }
    
    public CLFloatMatrix(float value) {
        this(1, 1, value);
    }

    private float[] getCLMatrix(int rows, int columns, float[] values) {
        float[] clValues = new float[clLength];
        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                clValues[i * clRows + j] = values[i * rows + j];
            }
        }
        return clValues;
    }

 

    public static CLFloatMatrix zeros(int rows, int columns) {
        CLFloatMatrix matrix = new CLFloatMatrix(rows, columns);
        matrix.setZero();
        return matrix;
    }

    public static CLFloatMatrix ones(int rows, int columns) {
        CLFloatMatrix matrix = new CLFloatMatrix(rows, columns);
        matrix.setOne();
        return matrix;
    }

    public static CLFloatMatrix rand(int rows, int columns) {
        CLFloatMatrix matrix = new CLFloatMatrix(rows, columns);
        matrix.nextRand();
        return matrix;
    }

    public static CLFloatMatrix randn(int rows, int columns) {
        CLFloatMatrix matrix = new CLFloatMatrix(rows, columns);
        matrix.nextRandn();
        return matrix;
    }

    private void initRandom() {
        if (!randomDataPointer.isPresent()) {
            int[] initRandom = new int[CORE.getThreadCount_Y() * CORE.getThreadCount_X() * 4];
            for (int i = 0; i < initRandom.length; i++) {
                initRandom[i] = Random.nextInt(Integer.MAX_VALUE - 1234) + 1234;
            }
            randomDataPointer = Optional.of(CORE.malloc(initRandom));
        }
    }

    public void setOne() {
        CLLevel1.setOne(this);
    }


    public void setZero() {
        CORE.execute(CLPredefined.getSubprogram("setZero"), this.clRows, this.clColumns, dataPointer);
    }
    
    public CLFloatMatrix repmat(int rowTimes, int columnTimes) {
        CLFloatMatrix result = new CLFloatMatrix(rows * rowTimes, columns * columnTimes);
        CORE.repmat(dataPointer, result.dataPointer,
                result.clRows, result.clColumns,
                result.rows, result.columns, rows, columns, clRows);
        return result;
    }

    public void setSubMatrix(CLFloatMatrix matrix, int rowOffset, int columnOffset) {
    	CORE.setSubMatrix(matrix.dataPointer, dataPointer,
                matrix.clRows, matrix.clColumns,
                matrix.rows, matrix.columns, rowOffset, columnOffset, clRows);
    }

    public CLFloatMatrix getSubMatrix(CLFloatMatrix result, int rowOffset, int columnOffset) {
        CORE.getSubMatrix(dataPointer, result.dataPointer,
                result.clRows, result.clColumns,
                result.rows, result.columns, rowOffset, columnOffset, clRows);
        return result;
    }
    
    // ADD
    
    public static void add(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	CLLevel1.add(matrixA, matrixB, result);
    }

    public static void add(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLLevel1.add(matrix, scalar, result);
    }

    public static void addColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	CLLevel1.addColumnVector(matrix, columnVector, result);
    }

    public static void addRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	CLLevel1.addRowVector(matrix, rowVector, result);
    }


    
    // MUL

    public static void mul(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	CLLevel1.mul(matrixA, matrixB, result);
    }

    public static void mul(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLLevel1.mul(matrix, scalar, result);
    }

    public static void mulColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	CLLevel1.mulColumnVector(matrix, columnVector, result);
    }

    public static void mulRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	CLLevel1.mulRowVector(matrix, rowVector, result);
    }


    
    // SUB

    public static void sub(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	CLLevel1.sub(matrixA, matrixB, result);
    }

    public static void sub(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLLevel1.sub(matrix, scalar, result);
    }

    public static void subColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	CLLevel1.subColumnVector(matrix, columnVector, result);
    }

    public static void subRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	CLLevel1.subRowVector(matrix, rowVector, result);
    }

    public static void rsub(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLLevel1.rsub(matrix, scalar, result);
    }

    public static void rsubColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	CLLevel1.rsubColumnVector(matrix, columnVector, result);
    }

    public static void rsubRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	CLLevel1.rsubRowVector(matrix, rowVector, result);
    }


    // DIV

    public static void div(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	CLLevel1.div(matrixA, matrixB, result);
    }

    public static void div(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLLevel1.div(matrix, scalar, result);
    }

    public static void divColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	CLLevel1.divColumnVector(matrix, columnVector, result);
    }

    public static void divRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	CLLevel1.divRowVector(matrix, rowVector, result);
    }

    public static void rdiv(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLLevel1.rdiv(matrix, scalar, result);
    }

    public static void rdivColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	CLLevel1.rdivColumnVector(matrix, columnVector, result);
    }

    public static void rdivRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	CLLevel1.rdivRowVector(matrix, rowVector, result);
    }


    // GREATER THAN
    
    public static void gt(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	CLLevel1.gt(matrixA, matrixB, result);
    }
    
    public static void gt(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLLevel1.gt(matrix, scalar, result);
    }

    public static void gtColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	CLLevel1.gtColumnVector(matrix, columnVector, result);
    }

    public static void gtRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	CLLevel1.gtRowVector(matrix, rowVector, result);
    }   
    
    
    // GREATER THAN OR EQUAL

    public static void ge(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	CLLevel1.ge(matrixA, matrixB, result);
    }
    
    public static void ge(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLLevel1.ge(matrix, scalar, result);
    }

    public static void geColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	CLLevel1.geColumnVector(matrix, columnVector, result);
    }

    public static void geRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	CLLevel1.geRowVector(matrix, rowVector, result);
    }   
    
    
    // LOWER THAN

    public static void lt(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	CLLevel1.lt(matrixA, matrixB, result);
    }
    
    public static void lt(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLLevel1.lt(matrix, scalar, result);
    }

    public static void ltColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	CLLevel1.ltColumnVector(matrix, columnVector, result);
    }

    public static void ltRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	CLLevel1.ltRowVector(matrix, rowVector, result);
    }   
    
    
    // LOWER THAN OR EQUAL

    public static void le(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	CLLevel1.le(matrixA, matrixB, result);
    }
    
    public static void le(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLLevel1.le(matrix, scalar, result);
    }

    public static void leColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	CLLevel1.leColumnVector(matrix, columnVector, result);
    }

    public static void leRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	CLLevel1.leRowVector(matrix, rowVector, result);
    } 
    
    
    // EQUAL

    public static void eq(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	CLLevel1.eq(matrixA, matrixB, result);
    }
    
    public static void eq(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLLevel1.eq(matrix, scalar, result);
    }

    public static void eqColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	CLLevel1.eqColumnVector(matrix, columnVector, result);
    }

    public static void eqRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	CLLevel1.eqRowVector(matrix, rowVector, result);
    }   
    
    
    // NOT EQUAL

    public static void ne(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	CLLevel1.ne(matrixA, matrixB, result);
    }
    
    public static void ne(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLLevel1.ne(matrix, scalar, result);
    }

    public static void neColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	CLLevel1.neColumnVector(matrix, columnVector, result);
    }

    public static void neRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	CLLevel1.neRowVector(matrix, rowVector, result);
    } 
 
   
    
    public static void dup(CLFloatMatrix matrix, CLFloatMatrix result) {
		CLLevel1.dup(matrix, result);
	}
	
    public static void exp(CLFloatMatrix matrix, CLFloatMatrix result) {
    	CLLevel1.exp(matrix, result);
	}

    public static void neg(CLFloatMatrix matrix, CLFloatMatrix result) {
    	CLLevel1.neg(matrix, result);
	}
	
	public static void sigmoid(CLFloatMatrix matrix, CLFloatMatrix result) {
		CLLevel1.sigmoid(matrix, result);
	}
    
    
    //// MATRIX_MULTIPLICATION

    public static void mmul(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        CORE.sgemm_nn(a.dataPointer, b.dataPointer, result.dataPointer, a.clRows, b.clColumns, a.clColumns);
    }

    public static void mmulTransposeA(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        CORE.sgemm_tn(a.dataPointer, b.dataPointer, result.dataPointer, a.clColumns, b.clColumns, a.clRows);
    }

    public static void mmulTransposeB(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        CORE.sgemm_nt(a.dataPointer, b.dataPointer, result.dataPointer, a.clRows, b.clRows, a.clColumns);
    }

    
    // TRANSPOSE

    public static void transpose(CLFloatMatrix matrix, CLFloatMatrix transposed) {
        CORE.transpose(matrix.dataPointer, transposed.dataPointer, matrix.clRows, matrix.clColumns, matrix.rows, matrix.columns);
    }

    
    
    // REDUCE

    public static float sum(CLFloatMatrix matrix) {
//        return CORE.reduce2D("sumFloats", matrix.dataPointer, matrix.rows, matrix.columns, 0);
        return CORE.reduce1D("sumFloats1D", matrix.dataPointer, matrix.clRows*matrix.clColumns);
    }

    public static CLFloatMatrix testsum(CLFloatMatrix matrix) {
        int tempSizeX = (int) Math.ceil((double) matrix.rows / 32);
        int tempSizeY = (int) Math.ceil((double) matrix.columns / 32);
        CLFloatMatrix result = new CLFloatMatrix(tempSizeX, tempSizeY);
        CORE.reduce2D("sumFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, result.rows, result.columns, 0);
        return result;
    }

    public static float mean(CLFloatMatrix matrix) {
        return sum(matrix) / matrix.length;
    }

    public static float prod(CLFloatMatrix matrix) {
        return CORE.reduce2D("productFloats", matrix.dataPointer, matrix.rows, matrix.columns, 1);
    }

    public static float max(CLFloatMatrix matrix) {
        return CORE.reduce2D("maxFloats", matrix.dataPointer, matrix.rows, matrix.columns, Float.NEGATIVE_INFINITY);
    }

    public static float min(CLFloatMatrix matrix) {
        return CORE.reduce2D("minFloats", matrix.dataPointer, matrix.rows, matrix.columns, Float.POSITIVE_INFINITY);
    }

    public static void columnSums(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceColumns("columnSumsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, 0);
    }

    public static void columnMeans(CLFloatMatrix matrix, CLFloatMatrix result) {
        columnSums(matrix, result);
        div(result, matrix.rows, result);
    }

    public static void columnProds(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceColumns("columnProductsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, 1);
    }

    public static void columnMaxs(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceColumns("columnMaxsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, Float.NEGATIVE_INFINITY);
    }

    public static void columnMins(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceColumns("columnMinsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, Float.POSITIVE_INFINITY);
    }


    // ROW REDUCTION

    public static void rowSums(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceRows("rowSumsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, 0);
    }

    public static void rowMeans(CLFloatMatrix matrix, CLFloatMatrix result) {
        rowSums(matrix, result);
        div(result, matrix.columns, result);
    }

    public static void rowProds(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceRows("rowProductsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, 1);
    }

    public static void rowMaxs(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceRows("rowMaxsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, Float.NEGATIVE_INFINITY);
    }

    public static void rowMins(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceRows("rowMinsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, Float.POSITIVE_INFINITY);
    }

    
    
    /**
     * Führe ein OpenCL Programm auf einer Matrix aus.
     * 
     * @param subprogram
     * @param a
     */
	protected static void runMatrixOperation(Subprogram<cl_kernel> subprogram, CLFloatMatrix a) {
		CORE.execute(subprogram, a.clRows, a.clColumns, a.rows, a.columns, a.dataPointer);
	}
	
    
    /**
     * Führe ein OpenCL Programm auf zwei gleich große Matrizen durch,
     * das Resultat wird in eine dritte Matrix gespeichert
     *  
     * @param programId
     * @param a
     * @param b
     * @param result
     */
	protected static void runMatrixMatrixElementWiseOperation(Subprogram<cl_kernel> subprogram, CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
		checkSameSize(a, b, result);
        CORE.execute(subprogram, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
	}
	
	
	/**
     * Führe ein OpenCL Programm auf einer Matrix durch,
     * das Resultat wird in eine zweite Matrix gespeichert
     * 
	 * @param programId
	 * @param matrix
	 * @param scalar
	 * @param result
	 */
	protected static void runMatrixElementWiseOperation(Subprogram<cl_kernel> subprogram, CLFloatMatrix a, CLFloatMatrix result) {
		checkSameSize(a, result);
        CORE.execute(subprogram, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer);
	}
	
	/**
     * Führe ein OpenCL Programm auf einer Matrix und einem Scalar durch,
     * das Resultat wird in eine zweite Matrix gespeichert
     * 
	 * @param programId
	 * @param matrix
	 * @param scalar
	 * @param result
	 */
	protected static void runMatrixScalarElementWiseOperation(Subprogram<cl_kernel> subprogram, CLFloatMatrix a, float scalar, CLFloatMatrix result) {
	    CLFloatMatrix b = new CLFloatMatrix(1, 1, scalar);
        CORE.execute(subprogram, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
        b.free();
	}
	
	/**
     * Führe ein OpenCL Programm auf einer Matrix und einem Row Vector durch,
     * das Resultat wird in eine zweite Matrix gespeichert
     * 
	 * @param programId
	 * @param matrix 
	 * @param row vector
	 * @param result
	 */
	protected static void runMatrixRowVectorElementWiseOperation(Subprogram<cl_kernel> subprogram, CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(subprogram, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
	}	
	
	
	/**
     * Führe ein OpenCL Programm auf einer Matrix und einem Column Vector durch,
     * das Resultat wird in eine zweite Matrix gespeichert
     * 
	 * @param programId
	 * @param matrix
	 * @param column vector
	 * @param result
	 */
	protected static void runMatrixColumnVectorElementWiseOperation(Subprogram<cl_kernel> subprogram, CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
		checkColumnVectorSize(a, b, result);
		CORE.execute(subprogram, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
	}	

   


    // RANDOM

    public void nextRand() {
        initRandom();
        CORE.uniform(dataPointer, randomDataPointer.get(), this.clRows, this.clColumns, this.rows, this.columns);
    }

    public void nextRandn() {
        initRandom();
        CORE.boxMuller(dataPointer, randomDataPointer.get(), this.clRows, this.clColumns, this.rows, this.columns);
    }

    public static void waitOnComplete() {
        CORE.waitOnComplete();
    }
    
    
    @Override
    public void free() {
        CORE.free(dataPointer);
        if (randomDataPointer.isPresent()) {
            CORE.free(randomDataPointer.get());
        }
    }

    @Override
    public void getColumnWiseOn(float[] values) {
        float[] clValues = new float[clLength];
        CORE.getData(dataPointer, clValues);
        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                values[i * rows + j] = clValues[i * clRows + j];
            }
        }
    }
    
    @Override
    public String toString() {
    	return toString1D();
    }
}
