package org.nblas.cl;


import java.util.Optional;

import org.jblas.util.Random;
import org.jocl.cl_kernel;
import org.nblas.Context;
import org.nblas.FloatMatrix;
import org.nblas.cl.blas.CLLevel1;
import org.nblas.generic.Subprogram;

/**
 * 
 * @author Nico
 *
 */
public class CLFloatMatrix extends CLMatrix implements FloatMatrix {

    public CLFloatMatrix(int rows, int columns) {
        super(rows, columns);
 		this.dataPointer = CORE.mallocSinglePrecision(this.clLength);
     }

    public CLFloatMatrix(int rows, int columns, float[] values) {
       super(rows, columns);

		if (rows * columns != values.length)
			throw new IllegalArgumentException("rows times columns " + (rows * columns) + " != " + "data length = " + values.length);

		this.dataPointer = CORE.malloc(values);
    }
    
    // ---------------------------------- utility methods -------------------------------------

    /**
  	 * @see FloatMatrix#getContext()
  	 */
	@Override
	public Context getContext() {
		return Context.createOpenCLSinglePrecisionContext();
	}  	
    
	/**
	  * @see FloatMatrix#getColumnWiseOn(float[])
	  */
	@Override
	public FloatMatrix getColumnWiseOn(float[] values) {
		
		float[] clValues = new float[clLength];
		CORE.getData(dataPointer, clValues);
		for (int i = 0; i < columns; i++) {
			for (int j = 0; j < rows; j++) {
				values[i * rows + j] = clValues[i * clRows + j];
			}
		}
		return this;
	}
	
	@Override
	public String toString() {
		return toString1D();
	}
	
	/**
	 * @see FloatMatrix#dup(FloatMatrix, FloatMatrix)
	 */
	@Override
    public FloatMatrix dup(FloatMatrix matrix, FloatMatrix result) {
		CLLevel1.dup((CLFloatMatrix)matrix, (CLFloatMatrix)result);
		return result;
	}
	
	/**
	 * @see FloatMatrix#transpose(FloatMatrix, FloatMatrix)
	 */
    @Override
    public FloatMatrix transpose(FloatMatrix matrix, FloatMatrix transposed) {
    	CLFloatMatrix mat = (CLFloatMatrix) matrix;
    	CLFloatMatrix result = (CLFloatMatrix) transposed;
        CORE.transpose(mat.dataPointer, result.dataPointer, mat.clRows, mat.clColumns, mat.rows, mat.columns);
        return transposed;
    }

    
    // ---------------------------------- inplace methods -------------------------------------
	
    /**
	 * @see FloatMatrix#setOne()
	 */
	@Override
    public FloatMatrix setOne() {
        CLLevel1.setOne(this);
        return this;
    }

    /**
	 * @see FloatMatrix#setZero()
	 */
	@Override
    public FloatMatrix setZero() {
        CORE.execute(CLPredefined.getSubprogram("setZero"), this.clRows, this.clColumns, dataPointer);
        return this;
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
    
    /**
	 * @see FloatMatrix#randi()
	 */
    @Override
    public FloatMatrix randi() {
        initRandom();
        CORE.uniform(dataPointer, randomDataPointer.get(), this.clRows, this.clColumns, this.rows, this.columns);
        return this;
    }

    /**
	 * @see FloatMatrix#randni()
	 */
    @Override
    public FloatMatrix randni() {
        initRandom();
        CORE.boxMuller(dataPointer, randomDataPointer.get(), this.clRows, this.clColumns, this.rows, this.columns);
        return this;
    }
    
	
	
	
    // --------------------------------------- add methods ----------------------------------------
    
    /**
	 * @see FloatMatrix#add(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
    @Override
    public FloatMatrix add(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	CLLevel1.add((CLFloatMatrix)matrixA, (CLFloatMatrix)matrixB, (CLFloatMatrix)result);
    	return result;
    }

    /**
	 * @see FloatMatrix#add(FloatMatrix, float, FloatMatrix)
	 */
    @Override
    public FloatMatrix add(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLLevel1.add((CLFloatMatrix)matrix, scalar, (CLFloatMatrix)result);
    	return result;
    }

    public static void addColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	CLLevel1.addColumnVector(matrix, columnVector, result);
    }

    public static void addRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	CLLevel1.addRowVector(matrix, rowVector, result);
    }
	 
    
    
    // --------------------------------------- sub methods ----------------------------------------
    
    /**
	 * @see FloatMatrix#sub(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
    @Override
    public FloatMatrix sub(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	CLLevel1.sub((CLFloatMatrix)matrixA, (CLFloatMatrix)matrixB, (CLFloatMatrix)result);
    	return result;
    }

    /**
	 * @see FloatMatrix#sub(FloatMatrix, float, FloatMatrix)
	 */
    @Override
    public FloatMatrix sub(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLLevel1.sub((CLFloatMatrix)matrix, scalar, (CLFloatMatrix)result);
    	return result;
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

    
    // --------------------------------------- mul methods ----------------------------------------
    
    /**
  	 * @see FloatMatrix#mul(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix mul(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	CLLevel1.mul((CLFloatMatrix)matrixA, (CLFloatMatrix)matrixB, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#mul(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix mul(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLLevel1.mul((CLFloatMatrix)matrix, scalar, (CLFloatMatrix)result);
    	return result;
    }

    public static void mulColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	CLLevel1.mulColumnVector(matrix, columnVector, result);
    }

    public static void mulRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	CLLevel1.mulRowVector(matrix, rowVector, result);
    }

    
    // --------------------------------------- div methods ----------------------------------------
    
    /**
  	 * @see FloatMatrix#div(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix div(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	CLLevel1.div((CLFloatMatrix)matrixA, (CLFloatMatrix)matrixB, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#div(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix div(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLLevel1.div((CLFloatMatrix)matrix, scalar, (CLFloatMatrix)result);
    	return result;
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

    
    
    // --------------------------------------- mathematical functions ----------------------------------------
	
    /**
  	 * @see FloatMatrix#exp(FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix exp(FloatMatrix matrix, FloatMatrix result) {
    	CLLevel1.exp((CLFloatMatrix)matrix, (CLFloatMatrix)result);
    	return result;
	}

    /**
  	 * @see FloatMatrix#neg(FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix neg(FloatMatrix matrix, FloatMatrix result) {
    	CLLevel1.neg((CLFloatMatrix)matrix, (CLFloatMatrix)result);
    	return result;
	}
	
    /**
  	 * @see FloatMatrix#sigmoid(FloatMatrix, FloatMatrix)
  	 */
    @Override
	public FloatMatrix sigmoid(FloatMatrix matrix, FloatMatrix result) {
		CLLevel1.sigmoid((CLFloatMatrix)matrix, (CLFloatMatrix)result);
		return result;
	}
	
	
    // --------------------------------------- greater than ----------------------------------------
    
    /**
  	 * @see FloatMatrix#gt(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix gt(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	CLLevel1.gt((CLFloatMatrix)matrixA, (CLFloatMatrix)matrixB, (CLFloatMatrix)result);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#gt(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix gt(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLLevel1.gt((CLFloatMatrix)matrix, scalar, (CLFloatMatrix)result);
    	return result;
    }

    public static void gtColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	CLLevel1.gtColumnVector(matrix, columnVector, result);
    }

    public static void gtRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	CLLevel1.gtRowVector(matrix, rowVector, result);
    }   
        
    
    // --------------------------------------- matrix multiplication ----------------------------------------    

    /**
  	 * @see FloatMatrix#mmul(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix mmul(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
    	CLFloatMatrix matrixA = (CLFloatMatrix) a;
    	CLFloatMatrix matrixB = (CLFloatMatrix) b;
    	CLFloatMatrix matrixR = (CLFloatMatrix) result;
        CORE.sgemm_nn(matrixA.dataPointer, matrixB.dataPointer, matrixR.dataPointer, matrixA.clRows, matrixB.clColumns, matrixA.clColumns);
        return result;
    }

    /**
  	 * @see FloatMatrix#mmulTN(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix mmulTN(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
    	CLFloatMatrix matrixA = (CLFloatMatrix) a;
    	CLFloatMatrix matrixB = (CLFloatMatrix) b;
    	CLFloatMatrix matrixR = (CLFloatMatrix) result;
        CORE.sgemm_tn(matrixA.dataPointer, matrixB.dataPointer, matrixR.dataPointer, matrixA.clColumns, matrixB.clColumns, matrixA.clRows);
        return result;
    }

    /**
  	 * @see FloatMatrix#mmulNT(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix mmulNT(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
    	CLFloatMatrix matrixA = (CLFloatMatrix) a;
    	CLFloatMatrix matrixB = (CLFloatMatrix) b;
    	CLFloatMatrix matrixR = (CLFloatMatrix) result;
        CORE.sgemm_nt(matrixA.dataPointer, matrixB.dataPointer, matrixR.dataPointer, matrixA.clRows, matrixB.clRows, matrixA.clColumns);
        return result;
    }

    
    
    // --------------------------------------- reduction methods ----------------------------------------

    /**
  	 * @see FloatMatrix#sum(FloatMatrix)
  	 */
    @Override
    public float sum(FloatMatrix matrix) {
    	CLFloatMatrix mat = (CLFloatMatrix) matrix;
//        return CORE.reduce2D("sumFloats", mat.dataPointer, mat.rows, mat.columns, 0);
        return CORE.reduce1D("sumFloats1D", mat.dataPointer, mat.clRows*mat.clColumns);
    }
    
    public static float mean(CLFloatMatrix matrix) {
        return matrix.sum(matrix) / matrix.length;
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
    
    public static CLFloatMatrix testsum(CLFloatMatrix matrix) {
        int tempSizeX = (int) Math.ceil((double) matrix.rows / 32);
        int tempSizeY = (int) Math.ceil((double) matrix.columns / 32);
        CLFloatMatrix result = new CLFloatMatrix(tempSizeX, tempSizeY);
        CORE.reduce2D("sumFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, result.rows, result.columns, 0);
        return result;
    }
    
    // --------------------------------------- column reduction methods ----------------------------------------

    public static void columnSums(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceColumns("columnSumsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, 0);
    }

    public static void columnMeans(CLFloatMatrix matrix, CLFloatMatrix result) {
        columnSums(matrix, result);
        result.div(result, matrix.rows, result);
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


    
    // --------------------------------------- row reduction methods ----------------------------------------

    public static void rowSums(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceRows("rowSumsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, 0);
    }

    public static void rowMeans(CLFloatMatrix matrix, CLFloatMatrix result) {
        rowSums(matrix, result);
        result.div(result, matrix.columns, result);
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
    
    
    
	// --------------------------------------- getter and setter methods ----------------------------------------
    
    /**
  	 * @see FloatMatrix#setSubMatrix(FloatMatrix, FloatMatrix, int, int)
  	 */
    @Override
    public FloatMatrix setSubMatrix(FloatMatrix source, FloatMatrix destination, int rowOffset, int columnOffset) {
    	CLFloatMatrix src = (CLFloatMatrix) source;
    	CLFloatMatrix dst = (CLFloatMatrix) destination;
    	CORE.setSubMatrix(src.dataPointer, dst.dataPointer, src.clRows, src.clColumns, src.rows, src.columns, rowOffset, columnOffset, dst.clRows);
    	return destination;
    }

    /**
  	 * @see FloatMatrix#getSubMatrix(FloatMatrix, FloatMatrix, int, int)
  	 */
    @Override
    public FloatMatrix getSubMatrix(FloatMatrix source, FloatMatrix destination, int rowOffset, int columnOffset) {
    	CLFloatMatrix src = (CLFloatMatrix) source;
    	CLFloatMatrix dst = (CLFloatMatrix) destination;
        CORE.getSubMatrix(dst.dataPointer, src.dataPointer, src.clRows, src.clColumns, src.rows, src.columns, rowOffset, columnOffset, dst.clRows);
        return destination;
    }
    
    
    
    
    
    
    
    
    // --------------------------------------- implementation ----------------------------------------

    
    public CLFloatMatrix repmat(int rowTimes, int columnTimes) {
        CLFloatMatrix result = new CLFloatMatrix(rows * rowTimes, columns * columnTimes);
        CORE.repmat(dataPointer, result.dataPointer,
                result.clRows, result.clColumns,
                result.rows, result.columns, rows, columns, clRows);
        return result;
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
 
   

    
    


    
    
    

    // --------------------------------------- helper methods ----------------------------------------
    
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
	    CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
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

	/**
	 * Warte so lange bis alle anstehenden Operationen auf der GPU durchgeführt wurden
	 */
    public static void waitOnComplete() {
        CORE.waitOnComplete();
    }


}
