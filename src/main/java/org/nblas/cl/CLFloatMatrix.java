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
 * TODO: addColumnVector usw. müssen noch in FloatMatrix
 * 
 * @author Nico
 *
 */
public class CLFloatMatrix extends CLMatrix implements FloatMatrix {

	protected static final CLFloatFunctionBuilder builder;
	protected static final CLLevel1 level1;
	

	static {
        
		builder = new CLFloatFunctionBuilder();
		
        // lade alle Predefined Kernels
        for (Subprogram<cl_kernel> subprogram : CLPredefined.getAllSubPrograms())
        	CORE.loadFromGeneratedSubprogram(subprogram);
	
        level1 = new CLLevel1(builder);
		
		CORE.compileMatrixFunctions();
	}
	
	
	/**
	 * dirty allocation
	 * 
	 * @param rows
	 * @param columns
	 */
    public CLFloatMatrix(int rows, int columns) {
        super(rows, columns);
 		this.dataPointer = CORE.mallocSinglePrecision(this.clLength);
     }

    public CLFloatMatrix(int rows, int columns, float[] values) {
       super(rows, columns);

		if (rows * columns != values.length)
			throw new IllegalArgumentException("rows times columns " + (rows * columns) + " != " + "data length = " + values.length);

        float[] clValues = getCLMatrix(rows, columns, values);
		this.dataPointer = CORE.malloc(clValues);
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
    
    
    // ---------------------------------- utility methods -------------------------------------

    /**
  	 * @see FloatMatrix#getContext()
  	 */
	@Override
	public Context getContext() {
		return Context.createOpenCLSinglePrecisionContext();
	}  	
    
	/**
	  * @see FloatMatrix#readRowMajor(float[])
	  */
	@Override
	public FloatMatrix readRowMajor(float[] values) {
		
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
		level1.dup((CLFloatMatrix)matrix, (CLFloatMatrix)result);
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
        level1.setOne(this);
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
    	level1.add((CLFloatMatrix)matrixA, (CLFloatMatrix)matrixB, (CLFloatMatrix)result);
    	return result;
    }

    /**
	 * @see FloatMatrix#add(FloatMatrix, float, FloatMatrix)
	 */
    @Override
    public FloatMatrix add(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.addScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.free();
    	return result;
    }

    public static void addColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	level1.addColumnVector(matrix, columnVector, result);
    }

    public static void addRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	level1.addRowVector(matrix, rowVector, result);
    }
	 
    
    
    // --------------------------------------- sub methods ----------------------------------------
    
    /**
	 * @see FloatMatrix#sub(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
    @Override
    public FloatMatrix sub(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.sub((CLFloatMatrix)matrixA, (CLFloatMatrix)matrixB, (CLFloatMatrix)result);
    	return result;
    }

    /**
	 * @see FloatMatrix#sub(FloatMatrix, float, FloatMatrix)
	 */
    @Override
    public FloatMatrix sub(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.subScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.free();
    	return result;
    }

    public static void subColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	level1.subColumnVector(matrix, columnVector, result);
    }

    public static void subRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	level1.subRowVector(matrix, rowVector, result);
    }

    public static void rsub(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.rsubScalar(matrix, b, result);
    	b.free();
    }

    public static void rsubColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	level1.rsubColumnVector(matrix, columnVector, result);
    }

    public static void rsubRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	level1.rsubRowVector(matrix, rowVector, result);
    }

    
    // --------------------------------------- mul methods ----------------------------------------
    
    /**
  	 * @see FloatMatrix#mul(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix mul(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.mul((CLFloatMatrix)matrixA, (CLFloatMatrix)matrixB, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#mul(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix mul(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.mulScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.free();
    	return result;
    }

    public static void mulColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	level1.mulColumnVector(matrix, columnVector, result);
    }

    public static void mulRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	level1.mulRowVector(matrix, rowVector, result);
    }

    
    // --------------------------------------- div methods ----------------------------------------
    
    /**
  	 * @see FloatMatrix#div(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix div(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.div((CLFloatMatrix)matrixA, (CLFloatMatrix)matrixB, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#div(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix div(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.divScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.free();
    	return result;
    }

    public static void divColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	level1.divColumnVector(matrix, columnVector, result);
    }

    public static void divRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	level1.divRowVector(matrix, rowVector, result);
    }
    
    public static void rdiv(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.rdivScalar(matrix, b, result);
    	b.free();
    }

    public static void rdivColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	level1.rdivColumnVector(matrix, columnVector, result);
    }

    public static void rdivRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	level1.rdivRowVector(matrix, rowVector, result);
    }

    
    
    // --------------------------------------- mathematical functions ----------------------------------------
	
    /**
  	 * @see FloatMatrix#exp(FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix exp(FloatMatrix matrix, FloatMatrix result) {
    	level1.exp((CLFloatMatrix)matrix, (CLFloatMatrix)result);
    	return result;
	}

    /**
  	 * @see FloatMatrix#neg(FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix neg(FloatMatrix matrix, FloatMatrix result) {
    	level1.neg((CLFloatMatrix)matrix, (CLFloatMatrix)result);
    	return result;
	}
	
    /**
  	 * @see FloatMatrix#sigmoid(FloatMatrix, FloatMatrix)
  	 */
    @Override
	public FloatMatrix sigmoid(FloatMatrix matrix, FloatMatrix result) {
		level1.sigmoid((CLFloatMatrix)matrix, (CLFloatMatrix)result);
		return result;
	}
	
	
    // --------------------------------------- greater than ----------------------------------------
    
    /**
  	 * @see FloatMatrix#gt(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix gt(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.gt((CLFloatMatrix)matrixA, (CLFloatMatrix)matrixB, (CLFloatMatrix)result);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#gt(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix gt(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.gtScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.free();
    	return result;
    }

    public static void gtColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	level1.gtColumnVector(matrix, columnVector, result);
    }

    public static void gtRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	level1.gtRowVector(matrix, rowVector, result);
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
        CORE.reduceColumns("columnSumsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, result.clRows, 0);
    }

    public static void columnMeans(CLFloatMatrix matrix, CLFloatMatrix result) {
        columnSums(matrix, result);
        result.div(result, matrix.rows, result);
    }

    public static void columnProds(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceColumns("columnProductsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, result.clRows, 1);
    }

    public static void columnMaxs(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceColumns("columnMaxsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, result.clRows, Float.NEGATIVE_INFINITY);
    }

    public static void columnMins(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceColumns("columnMinsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, result.clRows, Float.POSITIVE_INFINITY);
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
        CORE.getSubMatrix(src.dataPointer, dst.dataPointer, dst.clRows, dst.clColumns, dst.rows, dst.columns, rowOffset, columnOffset, src.clRows);
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
    	level1.ge(matrixA, matrixB, result);
    }
    
    public static void ge(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.geScalar(matrix, b, result);
    	b.free();
    }

    public static void geColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	level1.geColumnVector(matrix, columnVector, result);
    }

    public static void geRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	level1.geRowVector(matrix, rowVector, result);
    }   
    
    
    // LOWER THAN

    public static void lt(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	level1.lt(matrixA, matrixB, result);
    }
    
    public static void lt(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.ltScalar(matrix, b, result);
    	b.free();
    }

    public static void ltColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	level1.ltColumnVector(matrix, columnVector, result);
    }

    public static void ltRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	level1.ltRowVector(matrix, rowVector, result);
    }   
    
    
    // LOWER THAN OR EQUAL

    public static void le(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	level1.le(matrixA, matrixB, result);
    }
    
    public static void le(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.leScalar(matrix, b, result);
    	b.free();
    }

    public static void leColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	level1.leColumnVector(matrix, columnVector, result);
    }

    public static void leRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	level1.leRowVector(matrix, rowVector, result);
    } 
    
    
    // EQUAL

    public static void eq(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	level1.eq(matrixA, matrixB, result);
    }
    
    public static void eq(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.eqScalar(matrix, b, result);
    	b.free();
    }

    public static void eqColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	level1.eqColumnVector(matrix, columnVector, result);
    }

    public static void eqRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	level1.eqRowVector(matrix, rowVector, result);
    }   
    
    
    // NOT EQUAL

    public static void ne(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	level1.ne(matrixA, matrixB, result);
    }
    
    public static void ne(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.neScalar(matrix, b, result);
    	b.free();
    }

    public static void neColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	level1.neColumnVector(matrix, columnVector, result);
    }

    public static void neRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	level1.neRowVector(matrix, rowVector, result);
    } 
 
}
