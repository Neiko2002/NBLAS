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
		level1.dup((CudaFloatMatrix)source, (CudaFloatMatrix)destination);
	    return destination;
	}

    public FloatMatrix transpose(FloatMatrix matrix, FloatMatrix transposed) {
	   	CudaFloatMatrix mat = (CudaFloatMatrix)matrix;
    	CudaFloatMatrix result = (CudaFloatMatrix)transposed;
        CORE.transpose(mat.dataPointer, result.dataPointer, mat.getRows(), mat.getColumns());
        return transposed;
    }
    
    @Override
    public FloatMatrix repmat(FloatMatrix source, FloatMatrix destination, int rowMultiplicator, int columnMultiplicator) {
    	throw new NotImplementedException();
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
    	CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
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
    	CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
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
    	CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
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
    	CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
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
    	CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
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
    	CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
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
    	CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
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
	
    /**
  	 * @see FloatMatrix#exp(FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix exp(FloatMatrix matrix, FloatMatrix result) {
    	level1.exp((CudaFloatMatrix)matrix, (CudaFloatMatrix)result);
    	return result;
	}

    /**
  	 * @see FloatMatrix#neg(FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix neg(FloatMatrix matrix, FloatMatrix result) {
    	level1.neg((CudaFloatMatrix)matrix, (CudaFloatMatrix)result);
    	return result;
	}
	
    /**
  	 * @see FloatMatrix#sigmoid(FloatMatrix, FloatMatrix)
  	 */
    @Override
	public FloatMatrix sigmoid(FloatMatrix matrix, FloatMatrix result) {
		level1.sigmoid((CudaFloatMatrix)matrix, (CudaFloatMatrix)result);
		return result;
	}
    
	
    // --------------------------------------- greater than ----------------------------------------
    
    /**
  	 * @see FloatMatrix#gt(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix gt(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.gt((CudaFloatMatrix)matrixA, (CudaFloatMatrix)matrixB, (CudaFloatMatrix)result);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#gt(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix gt(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	level1.gtScalar((CudaFloatMatrix)matrix, b, (CudaFloatMatrix)result);
    	b.free();
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#gtColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix gtColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.gtColumnVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)columnVector, (CudaFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#gtRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix gtRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.gtRowVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)rowVector, (CudaFloatMatrix)result);
    	return result;
    }   
        
    
    // --------------------------------------- greater or equal than ----------------------------------------
    
    /**
  	 * @see FloatMatrix#ge(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ge(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.ge((CudaFloatMatrix)matrixA, (CudaFloatMatrix)matrixB, (CudaFloatMatrix)result);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#ge(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ge(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	level1.geScalar((CudaFloatMatrix)matrix, b, (CudaFloatMatrix)result);
    	b.free();
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#geColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix geColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.geColumnVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)columnVector, (CudaFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#geRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix geRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.geRowVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)rowVector, (CudaFloatMatrix)result);
    	return result;
    }  
    
    // --------------------------------------- less than ----------------------------------------
    
    /**
  	 * @see FloatMatrix#lt(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix lt(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.lt((CudaFloatMatrix)matrixA, (CudaFloatMatrix)matrixB, (CudaFloatMatrix)result);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#lt(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix lt(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	level1.ltScalar((CudaFloatMatrix)matrix, b, (CudaFloatMatrix)result);
    	b.free();
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#ltColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ltColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.ltColumnVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)columnVector, (CudaFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#ltRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ltRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.ltRowVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)rowVector, (CudaFloatMatrix)result);
    	return result;
    }   
    
    
    // --------------------------------------- less or equal than ----------------------------------------
    
    /**
  	 * @see FloatMatrix#le(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix le(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.le((CudaFloatMatrix)matrixA, (CudaFloatMatrix)matrixB, (CudaFloatMatrix)result);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#le(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix le(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	level1.leScalar((CudaFloatMatrix)matrix, b, (CudaFloatMatrix)result);
    	b.free();
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#leColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix leColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.leColumnVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)columnVector, (CudaFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#leRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix leRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.leRowVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)rowVector, (CudaFloatMatrix)result);
    	return result;
    }   
    
    
	// --------------------------------------- equal to ----------------------------------------
    
    /**
  	 * @see FloatMatrix#eq(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix eq(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.eq((CudaFloatMatrix)matrixA, (CudaFloatMatrix)matrixB, (CudaFloatMatrix)result);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#eq(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix eq(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	level1.eqScalar((CudaFloatMatrix)matrix, b, (CudaFloatMatrix)result);
    	b.free();
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#eqColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix eqColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.eqColumnVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)columnVector, (CudaFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#eqRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix eqRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.eqRowVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)rowVector, (CudaFloatMatrix)result);
    	return result;
    }   
    
    
    // --------------------------------------- not equal to ----------------------------------------
    
    /**
  	 * @see FloatMatrix#ne(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ne(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.ne((CudaFloatMatrix)matrixA, (CudaFloatMatrix)matrixB, (CudaFloatMatrix)result);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#ne(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ne(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	level1.neScalar((CudaFloatMatrix)matrix, b, (CudaFloatMatrix)result);
    	b.free();
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#neColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix neColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.neColumnVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)columnVector, (CudaFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#neRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix neRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.neRowVector((CudaFloatMatrix)matrix, (CudaFloatMatrix)rowVector, (CudaFloatMatrix)result);
    	return result;
    }    
    
    
    // --------------------------------------- matrix multiplication ----------------------------------------    

    @Override
    public FloatMatrix mmul(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
    	CudaFloatMatrix matrixA = (CudaFloatMatrix) a;
    	CudaFloatMatrix matrixB = (CudaFloatMatrix) b;
    	CudaFloatMatrix matrixR = (CudaFloatMatrix) result;
    	CORE.sgemmNN(matrixA.dataPointer, matrixA.getRows(), matrixA.getColumns(), matrixB.dataPointer, matrixB.getColumns(), matrixR.dataPointer);
        return result;
    }

    @Override
    public FloatMatrix mmulTN(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
    	CudaFloatMatrix matrixA = (CudaFloatMatrix) a;
    	CudaFloatMatrix matrixB = (CudaFloatMatrix) b;
    	CudaFloatMatrix matrixR = (CudaFloatMatrix) result;
    	CORE.sgemmTN(matrixA.dataPointer, matrixA.getRows(), matrixA.getColumns(), matrixB.dataPointer, matrixB.getColumns(), matrixR.dataPointer);
        return result;
    }

    @Override
    public FloatMatrix mmulNT(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
    	CudaFloatMatrix matrixA = (CudaFloatMatrix) a;
    	CudaFloatMatrix matrixB = (CudaFloatMatrix) b;
    	CudaFloatMatrix matrixR = (CudaFloatMatrix) result;
    	CORE.sgemmNT(matrixA.dataPointer, matrixA.getRows(), matrixA.getColumns(), matrixB.dataPointer, matrixB.getRows(), matrixR.dataPointer);
        return result;
    }
    
    // --------------------------------------- reduction methods ----------------------------------------

    @Override
    public float sum(FloatMatrix matrix) {
    	CudaFloatMatrix mat = (CudaFloatMatrix) matrix;
        return CORE.reduce("sumFloats", mat.dataPointer, mat.getLength(), 0);
    }

    @Override
    public float mean(FloatMatrix matrix) {
    	CudaFloatMatrix mat = (CudaFloatMatrix) matrix;
        return sum(matrix) / mat.getLength();
    }

    @Override
    public float prod(FloatMatrix matrix) {
    	CudaFloatMatrix mat = (CudaFloatMatrix) matrix;
        return CORE.reduce("productFloats", mat.dataPointer, mat.getLength(), 1);
    }

    @Override
    public float max(FloatMatrix matrix) {
    	CudaFloatMatrix mat = (CudaFloatMatrix) matrix;
        return CORE.reduce("maxFloats", mat.dataPointer, mat.getLength(), Float.NEGATIVE_INFINITY);
    }

    @Override
    public float min(FloatMatrix matrix) {
    	CudaFloatMatrix mat = (CudaFloatMatrix) matrix;
        return CORE.reduce("minFloats", mat.dataPointer, mat.getLength(), Float.POSITIVE_INFINITY);
    }


    
    // --------------------------------------- row reduction methods ----------------------------------------

    public static void rowSums(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceRows("rowSumsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), 0);
    }

    public static void rowMeans(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        rowSums(matrix, result);
        result.div(result, matrix.getColumns(), result);
    }

    public static void rowProds(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceRows("rowProductsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), 1);
    }

    public static void rowMaxs(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceRows("rowMaxsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), Float.NEGATIVE_INFINITY);
    }

    public static void rowMins(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceRows("rowMinsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), Float.POSITIVE_INFINITY);

    }

    

    // --------------------------------------- column reduction methods ----------------------------------------

    public static void columnSums(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceColumns("columnSumsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), 0);
    }

    public static void columnMeans(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        columnSums(matrix, result);
        result.div(result, matrix.getRows(), result);
    }

    public static void columnProds(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceColumns("columnProductsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), 1);
    }

    public static void columnMaxs(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceColumns("columnMaxsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), Float.NEGATIVE_INFINITY);
    }

    public static void columnMins(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceColumns("columnMinsFloats", matrix.dataPointer, result.dataPointer, matrix.getRows(), matrix.getColumns(), Float.POSITIVE_INFINITY);
    }
    
    
    // --------------------------------------- getter and setter methods ----------------------------------------

    public FloatMatrix getSubMatrix(FloatMatrix source, FloatMatrix destination, int rowOffset, int columnOffset) {
    	CudaFloatMatrix src = (CudaFloatMatrix) source;
    	CudaFloatMatrix dst = (CudaFloatMatrix) destination;    	
        CORE.getSubMatrix(src.dataPointer, dst.dataPointer, dst.getRows(), dst.getColumns(), src.getRows(), rowOffset, columnOffset);
        return destination;
    }
    
    public FloatMatrix setSubMatrix(FloatMatrix source, FloatMatrix destination, int offsetRow, int offsetColumn) {
    	CudaFloatMatrix src = (CudaFloatMatrix) source;
    	CudaFloatMatrix dst = (CudaFloatMatrix) destination;   
        CORE.setSubMatrix(dst.dataPointer, src.dataPointer, src.getRows(), src.getRows(), dst.getRows(), offsetRow, offsetColumn);
        return destination;
    }

//	@Override
//	public FloatMatrix getRow(FloatMatrix src, FloatMatrix row, int rowIndex) {
//		// TODO Auto-generated method stub
//		return null;
//	}
//
//	@Override
//	public FloatMatrix getColumn(FloatMatrix src, FloatMatrix column, int columnIndex) {
//		// TODO Auto-generated method stub
//		return null;
//	}
//
//	@Override
//	public FloatMatrix setRow(FloatMatrix dst, FloatMatrix row, int rowIndex) {
//		// TODO Auto-generated method stub
//		return null;
//	}
//
//	@Override
//	public FloatMatrix setColumn(FloatMatrix dst, FloatMatrix column, int columnIndex) {
//		// TODO Auto-generated method stub
//		return null;
//	}
    
    

}
