package org.nblas.cuda;


import org.nblas.Context;
import org.nblas.FloatMatrix;
import org.nblas.cuda.blas.CudaLevel1;
import org.nblas.generic.Subprogram;

import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * TODO hier sollten keine CORE zugriffe statt finden
 * TODO hier sollten keine subprogram gebaut werden
 * 
 * @author Nico
 *
 */
public class CudaFloatMatrix extends CudaMatrix implements FloatMatrix  {
   
    protected static final CudaLevel1 level1;
    protected static final Context context;

    static {
    	context = Context.createCudaSinglePrecisionContext();
    	
        CudaFloatFunctionBuilder builder = new CudaFloatFunctionBuilder();
               
        for (Subprogram<CUfunction> subprogram : CudaPredefined.kernels.values()) {
            CORE.loadFromGeneratedSubprogram(subprogram);
        }
        
        level1 = new CudaLevel1(builder);
    }


    /**
     * dirty allocation
     * 
     * @param rows
     * @param columns
     */
    public CudaFloatMatrix(int rows, int columns) {
        super(rows, columns);
        this.dataPointer = CORE.malloc(this.length, Sizeof.FLOAT);
     }

    public CudaFloatMatrix(int rows, int columns, float[] values) {
       super(rows, columns);
		
		if (rows * columns != values.length)
			throw new IllegalArgumentException("rows times columns " + (rows * columns) + " != " + "data length = " + values.length);

		this.dataPointer = CORE.malloc(values, Sizeof.FLOAT);
    }    
    
    // ---------------------------------- utility methods -------------------------------------

	@Override
	public Context getContext() {
		return context;
	}
	
    @Override
    public FloatMatrix readRowMajor(float[] values) {
        if (getRows() * getColumns() != values.length)
            throw new IllegalArgumentException("Array's length is not the size of rows times columns.");
        CORE.getData(dataPointer, values);
        return this;
    }
    
	@Override
	public String toString() {
		return toString1D();
	}
    
	@Override
	public FloatMatrix dup(FloatMatrix source, FloatMatrix destination) {
		level1.dup((CudaMatrix)source, (CudaMatrix)destination);
	    return destination;
	}

    public FloatMatrix transpose(FloatMatrix matrix, FloatMatrix transposed) {
	   	CudaMatrix mat = (CudaMatrix)matrix;
    	CudaMatrix result = (CudaMatrix)transposed;
        CORE.transpose(mat.dataPointer, result.dataPointer, mat.getRows(), mat.getColumns());
        return transposed;
    }
	
	// ---------------------------------- inplace methods -------------------------------------
	
	@Override
    public FloatMatrix setOne() {
		level1.setOne(this);
        return this;
    }

	@Override
    public FloatMatrix setZero() {
		level1.setZero(this);
        return this;
    }

	@Override
    public FloatMatrix randi() {
        CORE.rand(dataPointer, length);
        return this;
    }

	@Override
    public FloatMatrix randni() {
        CORE.randn(dataPointer, length);
        return this;
    }
	
	
    // --------------------------------------- add methods ----------------------------------------
    
    /**
	 * @see FloatMatrix#add(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
    @Override
    public FloatMatrix add(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.add((CudaFloatMatrix)matrixA, (CudaFloatMatrix)matrixB, (CudaFloatMatrix)result);
    	return result;
    }

    /**
	 * @see FloatMatrix#add(FloatMatrix, float, FloatMatrix)
	 */
    @Override
    public FloatMatrix add(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CudaMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	level1.addScalar((CudaFloatMatrix)matrix, b, (CudaFloatMatrix)result);
    	b.free();
    	return result;
    }

    /**
 	 * @see FloatMatrix#addColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
 	 */
    @Override
    public FloatMatrix addColumnVector(FloatMatrix a, FloatMatrix columnVector, FloatMatrix result) {
    	level1.addColumnVector((CudaFloatMatrix)a, (CudaFloatMatrix)columnVector, (CudaFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#addRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix addRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result) {
    	level1.addRowVector((CudaFloatMatrix)a, (CudaFloatMatrix)rowVector, (CudaFloatMatrix)result);
    	return result;
    } 	
    
    
    // --------------------------------------- sub methods ----------------------------------------
    
    /**
	 * @see FloatMatrix#sub(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
    @Override
    public FloatMatrix sub(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.sub((CudaFloatMatrix)matrixA, (CudaFloatMatrix)matrixB, (CudaFloatMatrix)result);
    	return result;
    }

    /**
	 * @see FloatMatrix#sub(FloatMatrix, float, FloatMatrix)
	 */
    @Override
    public FloatMatrix sub(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CudaMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	level1.subScalar((CudaFloatMatrix)matrix, b, (CudaFloatMatrix)result);
    	b.free();
    	return result;
    }

    /**
 	 * @see FloatMatrix#subColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
 	 */
    @Override
    public FloatMatrix subColumnVector(FloatMatrix a, FloatMatrix columnVector, FloatMatrix result) {
    	level1.subColumnVector((CudaFloatMatrix)a, (CudaFloatMatrix)columnVector, (CudaFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#subRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix subRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result) {
    	level1.subRowVector((CudaFloatMatrix)a, (CudaFloatMatrix)rowVector, (CudaFloatMatrix)result);
    	return result;
    } 
    
    /**
  	 * @see FloatMatrix#rsub(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix rsub(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CudaMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	level1.rsubScalar((CudaFloatMatrix)matrix, b, (CudaFloatMatrix)result);
    	b.free();
    	return result;
    }

    /**
  	 * @see FloatMatrix#rsubColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix rsubColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.rsubColumnVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)columnVector, (CudaFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#rsubRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix rsubRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.rsubRowVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)rowVector, (CudaFloatMatrix)result);
    	return result;
    }
    
    
    // --------------------------------------- mul methods ----------------------------------------
    
    /**
  	 * @see FloatMatrix#mul(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix mul(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.mul((CudaFloatMatrix)matrixA, (CudaFloatMatrix)matrixB, (CudaFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#mul(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix mul(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CudaMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	level1.mulScalar((CudaFloatMatrix)matrix, b, (CudaFloatMatrix)result);
    	b.free();
    	return result;
    }

    /**
 	 * @see FloatMatrix#mulColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
 	 */
    @Override
    public FloatMatrix mulColumnVector(FloatMatrix a, FloatMatrix columnVector, FloatMatrix result) {
    	level1.mulColumnVector((CudaFloatMatrix)a, (CudaFloatMatrix)columnVector, (CudaFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#mulRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix mulRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result) {
    	level1.mulRowVector((CudaFloatMatrix)a, (CudaFloatMatrix)rowVector, (CudaFloatMatrix)result);
    	return result;
    } 
    
    
    // --------------------------------------- div methods ----------------------------------------
    
    /**
  	 * @see FloatMatrix#div(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix div(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.div((CudaFloatMatrix)matrixA, (CudaFloatMatrix)matrixB, (CudaFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#div(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix div(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CudaMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	level1.divScalar((CudaFloatMatrix)matrix, b, (CudaFloatMatrix)result);
    	b.free();
    	return result;
    }
    
    /**
 	 * @see FloatMatrix#divColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
 	 */
    @Override
    public FloatMatrix divColumnVector(FloatMatrix a, FloatMatrix columnVector, FloatMatrix result) {
    	level1.divColumnVector((CudaFloatMatrix)a, (CudaFloatMatrix)columnVector, (CudaFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#divRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix divRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result) {
    	level1.divRowVector((CudaFloatMatrix)a, (CudaFloatMatrix)rowVector, (CudaFloatMatrix)result);
    	return result;
    } 
	
    public static void rdiv(CudaFloatMatrix matrix, float scalar, CudaFloatMatrix result) {
    	CudaMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	level1.rdivScalar(matrix, b, result);
    	b.free();
    }

    public static void rdivColumnVector(CudaFloatMatrix matrix, CudaFloatMatrix columnVector, CudaFloatMatrix result) {
    	level1.rdivColumnVector(matrix, columnVector, result);
    }

    public static void rdivRowVector(CudaFloatMatrix matrix, CudaFloatMatrix rowVector, CudaFloatMatrix result) {
    	level1.rdivRowVector(matrix, rowVector, result);
    }
    
    /**
  	 * @see FloatMatrix#rdiv(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix rdiv(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CudaMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	level1.rdivScalar((CudaFloatMatrix)matrix, b, (CudaFloatMatrix)result);
    	b.free();
    	return result;
    }

    /**
  	 * @see FloatMatrix#rdivColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix rdivColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.rdivColumnVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)columnVector, (CudaFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#rdivRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix rdivRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.rdivRowVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)rowVector, (CudaFloatMatrix)result);
    	return result;
    }
    
    
    // --------------------------------------- mathematical functions ----------------------------------------
	
    @Override
    public FloatMatrix exp(FloatMatrix matrix, FloatMatrix result) {
    	throw new NotImplementedException();
	}

    @Override
    public FloatMatrix neg(FloatMatrix matrix, FloatMatrix result) {
    	throw new NotImplementedException();
	}
	
    @Override
	public FloatMatrix sigmoid(FloatMatrix matrix, FloatMatrix result) {
    	throw new NotImplementedException();
	}
    
    // --------------------------------------- greater than ----------------------------------------
    
    @Override
    public FloatMatrix gt(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	throw new NotImplementedException();
    }
    
    @Override
    public FloatMatrix gt(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	throw new NotImplementedException();
    }
    
    
    
    
    // --------------------------------------- matrix multiplication ----------------------------------------    

    @Override
    public FloatMatrix mmul(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
    	CudaMatrix matrixA = (CudaMatrix) a;
    	CudaMatrix matrixB = (CudaMatrix) b;
    	CudaMatrix matrixR = (CudaMatrix) result;
    	CORE.sgemmNN(matrixA.dataPointer, matrixA.getRows(), matrixA.getColumns(), matrixB.dataPointer, matrixB.getColumns(), matrixR.dataPointer);
        return result;
    }

    @Override
    public FloatMatrix mmulTN(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
    	CudaMatrix matrixA = (CudaMatrix) a;
    	CudaMatrix matrixB = (CudaMatrix) b;
    	CudaMatrix matrixR = (CudaMatrix) result;
    	CORE.sgemmTN(matrixA.dataPointer, matrixA.getRows(), matrixA.getColumns(), matrixB.dataPointer, matrixB.getColumns(), matrixR.dataPointer);
        return result;
    }

    @Override
    public FloatMatrix mmulNT(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
    	CudaMatrix matrixA = (CudaMatrix) a;
    	CudaMatrix matrixB = (CudaMatrix) b;
    	CudaMatrix matrixR = (CudaMatrix) result;
    	CORE.sgemmNT(matrixA.dataPointer, matrixA.getRows(), matrixA.getColumns(), matrixB.dataPointer, matrixB.getRows(), matrixR.dataPointer);
        return result;
    }
    
    // --------------------------------------- reduction methods ----------------------------------------

    @Override
    public float sum(FloatMatrix matrix) {
    	CudaMatrix mat = (CudaMatrix) matrix;
        return CORE.reduce("sumFloats", mat.dataPointer, mat.getLength(), 0);
    }

    @Override
    public float mean(FloatMatrix matrix) {
    	CudaMatrix mat = (CudaMatrix) matrix;
        return sum(matrix) / mat.getLength();
    }

    @Override
    public float prod(FloatMatrix matrix) {
    	CudaMatrix mat = (CudaMatrix) matrix;
        return CORE.reduce("productFloats", mat.dataPointer, mat.getLength(), 1);
    }

    @Override
    public float max(FloatMatrix matrix) {
    	CudaMatrix mat = (CudaMatrix) matrix;
        return CORE.reduce("maxFloats", mat.dataPointer, mat.getLength(), Float.NEGATIVE_INFINITY);
    }

    @Override
    public float min(FloatMatrix matrix) {
    	CudaMatrix mat = (CudaMatrix) matrix;
        return CORE.reduce("minFloats", mat.dataPointer, mat.getLength(), Float.POSITIVE_INFINITY);
    }


    
    // --------------------------------------- row reduction methods ----------------------------------------

    public static void rowSums(CudaMatrix matrix, CudaMatrix result) {
        CORE.reduceRows("rowSumsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), 0);
    }

    public static void rowMeans(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        rowSums(matrix, result);
        result.div(result, matrix.getColumns(), result);
    }

    public static void rowProds(CudaMatrix matrix, CudaMatrix result) {
        CORE.reduceRows("rowProductsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), 1);
    }

    public static void rowMaxs(CudaMatrix matrix, CudaMatrix result) {
        CORE.reduceRows("rowMaxsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), Float.NEGATIVE_INFINITY);
    }

    public static void rowMins(CudaMatrix matrix, CudaMatrix result) {
        CORE.reduceRows("rowMinsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), Float.POSITIVE_INFINITY);

    }

    

    // --------------------------------------- column reduction methods ----------------------------------------

    public static void columnSums(CudaMatrix matrix, CudaMatrix result) {
        CORE.reduceColumns("columnSumsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), 0);
    }

    public static void columnMeans(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        columnSums(matrix, result);
        result.div(result, matrix.getRows(), result);
    }

    public static void columnProds(CudaMatrix matrix, CudaMatrix result) {
        CORE.reduceColumns("columnProductsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), 1);
    }

    public static void columnMaxs(CudaMatrix matrix, CudaMatrix result) {
        CORE.reduceColumns("columnMaxsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), Float.NEGATIVE_INFINITY);
    }

    public static void columnMins(CudaMatrix matrix, CudaMatrix result) {
        CORE.reduceColumns("columnMinsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), Float.POSITIVE_INFINITY);
    }
    
    
    // --------------------------------------- getter and setter methods ----------------------------------------

    public FloatMatrix getSubMatrix(FloatMatrix source, FloatMatrix destination, int rowOffset, int columnOffset) {
    	CudaMatrix src = (CudaMatrix) source;
    	CudaMatrix dst = (CudaMatrix) destination;    	
        CORE.getSubMatrix(src.dataPointer, dst.dataPointer, dst.getRows(), dst.getColumns(), src.getRows(), rowOffset, columnOffset);
        return destination;
    }
    
    public FloatMatrix setSubMatrix(FloatMatrix source, FloatMatrix destination, int offsetRow, int offsetColumn) {
    	CudaMatrix src = (CudaMatrix) source;
    	CudaMatrix dst = (CudaMatrix) destination;   
        CORE.setSubMatrix(dst.dataPointer, src.dataPointer, src.getRows(), src.getRows(), dst.getRows(), offsetRow, offsetColumn);
        return destination;
    }
    
    

}
