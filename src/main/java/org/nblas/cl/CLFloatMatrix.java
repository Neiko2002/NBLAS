package org.nblas.cl;


import org.jocl.cl_kernel;
import org.nblas.Context;
import org.nblas.FloatMatrix;
import org.nblas.FloatMatrixDefault;
import org.nblas.cl.blas.CLLevel1;
import org.nblas.generic.Subprogram;

/**
 * 
 * TODO: addColumnVector usw. müssen noch in FloatMatrix
 * 
 * @author Nico
 *
 */
public class CLFloatMatrix extends CLMatrix implements FloatMatrixDefault {

	protected static final CLLevel1 level1;

	static {        
		CLFloatFunctionBuilder builder = new CLFloatFunctionBuilder();
		
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
// 		this.dataPointer = CORE.malloc(new float[clLength]);
     }

    public CLFloatMatrix(int rows, int columns, float[] values) {
       super(rows, columns);

		if (rows * columns != values.length)
			throw new IllegalArgumentException("rows times columns " + (rows * columns) + " != " + "data length = " + values.length);

        float[] clValues = getFloatArray2D(values);
		this.dataPointer = CORE.malloc(clValues);
    }
    
    private float[] getFloatArray2D(float[] values) {
        float[] clValues = new float[clLength];
        for (int y = 0; y < columns; y++)
            for (int x = 0; x < rows; x++)
                clValues[y * clRows + x] = values[y * rows + x];
        return clValues;
    }
    
    
    // ---------------------------------- utility methods -------------------------------------

    /**
  	 * @see FloatMatrix#getContext()
  	 */
	@Override
	public Context getContext() {
		return Context.OpenCLSinglePrecisionContext;
	}  	
    
	/**
	  * @see FloatMatrix#readRowMajor(float[])
	  */
	@Override
	public FloatMatrix readRowMajor(float[] values) {
		
		float[] clValues = new float[clLength];
		CORE.getData(dataPointer, clValues);
		for (int y = 0; y < columns; y++)
			for (int x = 0; x < rows; x++)
				values[y * rows + x] = clValues[y * clRows + x];
		return this;
	}
	
	@Override
	public String toString() {
		return toString2D();
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
		// TODO CL.clEnqueueFillBuffer vielleicht schneller
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
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.addScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.release();
    	return result;
    }

    /**
 	 * @see FloatMatrix#addColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
 	 */
    @Override
    public FloatMatrix addColumnVector(FloatMatrix a, FloatMatrix columnVector, FloatMatrix result) {
    	level1.addColumnVector((CLFloatMatrix)a, (CLFloatMatrix)columnVector, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#addRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix addRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result) {
    	level1.addRowVector((CLFloatMatrix)a, (CLFloatMatrix)rowVector, (CLFloatMatrix)result);
    	return result;
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
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.subScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.release();
    	return result;
    }

    /**
 	 * @see FloatMatrix#subColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
 	 */
    @Override
    public FloatMatrix subColumnVector(FloatMatrix a, FloatMatrix columnVector, FloatMatrix result) {
    	level1.subColumnVector((CLFloatMatrix)a, (CLFloatMatrix)columnVector, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#subRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix subRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result) {
    	level1.subRowVector((CLFloatMatrix)a, (CLFloatMatrix)rowVector, (CLFloatMatrix)result);
    	return result;
    } 

    /**
  	 * @see FloatMatrix#rsub(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix rsub(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.rsubScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.release();
    	return result;
    }

    /**
  	 * @see FloatMatrix#rsubColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix rsubColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.rsubColumnVector((CLFloatMatrix)matrix, (CLFloatMatrix)columnVector, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#rsubRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix rsubRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.rsubRowVector((CLFloatMatrix)matrix, (CLFloatMatrix)rowVector, (CLFloatMatrix)result);
    	return result;
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
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.mulScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.release();
    	return result;
    }

    /**
 	 * @see FloatMatrix#mulColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
 	 */
    @Override
    public FloatMatrix mulColumnVector(FloatMatrix a, FloatMatrix columnVector, FloatMatrix result) {
    	level1.mulColumnVector((CLFloatMatrix)a, (CLFloatMatrix)columnVector, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#mulRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix mulRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result) {
    	level1.mulRowVector((CLFloatMatrix)a, (CLFloatMatrix)rowVector, (CLFloatMatrix)result);
    	return result;
    } 

    
    // --------------------------------------- div methods ----------------------------------------
    
    /**
  	 * @see FloatMatrix#div(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix div(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
    	level1.div((CLFloatMatrix)a, (CLFloatMatrix)b, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#div(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix div(FloatMatrix a, float scalar, FloatMatrix result) {
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.divScalar((CLFloatMatrix)a, b, (CLFloatMatrix)result);
    	b.release();
    	return result;
    }
    
    /**
 	 * @see FloatMatrix#divColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
 	 */
    @Override
    public FloatMatrix divColumnVector(FloatMatrix a, FloatMatrix columnVector, FloatMatrix result) {
    	level1.divColumnVector((CLFloatMatrix)a, (CLFloatMatrix)columnVector, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#divRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix divRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result) {
    	level1.divRowVector((CLFloatMatrix)a, (CLFloatMatrix)rowVector, (CLFloatMatrix)result);
    	return result;
    } 
    
    /**
  	 * @see FloatMatrix#rdiv(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix rdiv(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.rdivScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.release();
    	return result;
    }

    /**
  	 * @see FloatMatrix#rdivColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix rdivColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.rdivColumnVector((CLFloatMatrix)matrix, (CLFloatMatrix)columnVector, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#rdivRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix rdivRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.rdivRowVector((CLFloatMatrix)matrix, (CLFloatMatrix)rowVector, (CLFloatMatrix)result);
    	return result;
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
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.gtScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.release();
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#gtColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix gtColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.gtColumnVector((CLFloatMatrix)matrix, (CLFloatMatrix)columnVector, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#gtRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix gtRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.gtRowVector((CLFloatMatrix)matrix, (CLFloatMatrix)rowVector, (CLFloatMatrix)result);
    	return result;
    }   
        
    
    // --------------------------------------- greater or equal than ----------------------------------------
    
    /**
  	 * @see FloatMatrix#ge(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ge(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.ge((CLFloatMatrix)matrixA, (CLFloatMatrix)matrixB, (CLFloatMatrix)result);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#ge(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ge(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.geScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.release();
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#geColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix geColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.geColumnVector((CLFloatMatrix)matrix, (CLFloatMatrix)columnVector, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#geRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix geRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.geRowVector((CLFloatMatrix)matrix, (CLFloatMatrix)rowVector, (CLFloatMatrix)result);
    	return result;
    }  
    
    // --------------------------------------- less than ----------------------------------------
    
    /**
  	 * @see FloatMatrix#lt(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix lt(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.lt((CLFloatMatrix)matrixA, (CLFloatMatrix)matrixB, (CLFloatMatrix)result);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#lt(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix lt(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.ltScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.release();
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#ltColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ltColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.ltColumnVector((CLFloatMatrix)matrix, (CLFloatMatrix)columnVector, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#ltRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ltRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.ltRowVector((CLFloatMatrix)matrix, (CLFloatMatrix)rowVector, (CLFloatMatrix)result);
    	return result;
    }   
    
    
    // --------------------------------------- less or equal than ----------------------------------------
    
    /**
  	 * @see FloatMatrix#le(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix le(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.le((CLFloatMatrix)matrixA, (CLFloatMatrix)matrixB, (CLFloatMatrix)result);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#le(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix le(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.leScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.release();
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#leColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix leColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.leColumnVector((CLFloatMatrix)matrix, (CLFloatMatrix)columnVector, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#leRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix leRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.leRowVector((CLFloatMatrix)matrix, (CLFloatMatrix)rowVector, (CLFloatMatrix)result);
    	return result;
    }   
    
    
	// --------------------------------------- equal to ----------------------------------------
    
    /**
  	 * @see FloatMatrix#eq(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix eq(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.eq((CLFloatMatrix)matrixA, (CLFloatMatrix)matrixB, (CLFloatMatrix)result);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#eq(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix eq(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.eqScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.release();
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#eqColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix eqColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.eqColumnVector((CLFloatMatrix)matrix, (CLFloatMatrix)columnVector, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#eqRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix eqRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.eqRowVector((CLFloatMatrix)matrix, (CLFloatMatrix)rowVector, (CLFloatMatrix)result);
    	return result;
    }   
    
    
    // --------------------------------------- not equal to ----------------------------------------
    
    /**
  	 * @see FloatMatrix#ne(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ne(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	level1.ne((CLFloatMatrix)matrixA, (CLFloatMatrix)matrixB, (CLFloatMatrix)result);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#ne(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ne(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.neScalar((CLFloatMatrix)matrix, b, (CLFloatMatrix)result);
    	b.release();
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#neColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix neColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	level1.neColumnVector((CLFloatMatrix)matrix, (CLFloatMatrix)columnVector, (CLFloatMatrix)result);
    	return result;
    }

    /**
  	 * @see FloatMatrix#neRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix neRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	level1.neRowVector((CLFloatMatrix)matrix, (CLFloatMatrix)rowVector, (CLFloatMatrix)result);
    	return result;
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
    
     public FloatMatrix mmulCustom(cl_kernel kernel, FloatMatrix a, FloatMatrix b, FloatMatrix result) {
     	CLFloatMatrix matrixA = (CLFloatMatrix) a;
     	CLFloatMatrix matrixB = (CLFloatMatrix) b;
     	CLFloatMatrix matrixR = (CLFloatMatrix) result;
        CORE.sgemm_nn_custom(kernel, matrixA.dataPointer, matrixB.dataPointer, matrixR.dataPointer, matrixA.clRows, matrixB.clColumns, matrixA.clColumns);
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
    
    @Override
    public float mean(FloatMatrix matrix) {
    	CLFloatMatrix mat = (CLFloatMatrix) matrix;
        return mat.sum(matrix) / mat.length;
    }

    @Override
    public float prod(FloatMatrix matrix) {
    	CLFloatMatrix mat = (CLFloatMatrix) matrix;
        return CORE.reduce2D("productFloats", mat.dataPointer, mat.rows, mat.columns, 1);
    }

    @Override
    public float max(FloatMatrix matrix) {
    	CLFloatMatrix mat = (CLFloatMatrix) matrix;
        return CORE.reduce2D("maxFloats", mat.dataPointer, mat.rows, mat.columns, Float.NEGATIVE_INFINITY);
    }

    @Override
    public float min(FloatMatrix matrix) {
    	CLFloatMatrix mat = (CLFloatMatrix) matrix;
        return CORE.reduce2D("minFloats", mat.dataPointer, mat.rows, mat.columns, Float.POSITIVE_INFINITY);
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
    
        
    
    // --------------------------------------- implementation ----------------------------------------

    @Override
    public FloatMatrix repmat(FloatMatrix source, FloatMatrix destination, int rowMultiplicator, int columnMultiplicator) {
        CLFloatMatrix result = (CLFloatMatrix)destination;
        CORE.repmat(dataPointer, result.dataPointer,
                result.clRows, result.clColumns,
                result.rows, result.columns, rows, columns, clRows);
        return destination;
    }

    
    // GREATER THAN OR EQUAL

    public static void ge(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	level1.ge(matrixA, matrixB, result);
    }
    
    public static void ge(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.geScalar(matrix, b, result);
    	b.release();
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
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.ltScalar(matrix, b, result);
    	b.release();
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
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.leScalar(matrix, b, result);
    	b.release();
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
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.eqScalar(matrix, b, result);
    	b.release();
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
    	CLFloatMatrix b = new CLFloatMatrix(1, 1, new float[] { scalar });
    	level1.neScalar(matrix, b, result);
    	b.release();
    }

    public static void neColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	level1.neColumnVector(matrix, columnVector, result);
    }

    public static void neRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	level1.neRowVector(matrix, rowVector, result);
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
    

    public FloatMatrix getCustom(cl_kernel kernel, FloatMatrix source, FloatMatrix destination, int rowOffset, int columnOffset) {
    	CLFloatMatrix src = (CLFloatMatrix) source;
    	CLFloatMatrix dst = (CLFloatMatrix) destination;
        CORE.getCustom(kernel, src.dataPointer, dst.dataPointer, dst.clRows, dst.clColumns, dst.rows, dst.columns, rowOffset, columnOffset, src.clRows);
        return destination;
    }

    /**
     * @see FloatMatrix#put(FloatMatrix, int, int, float)
     */
	@Override
	public FloatMatrix put(FloatMatrix dst, int rowIndex, int columnIndex, float value) {
    	CLFloatMatrix val = new CLFloatMatrix(1, 1, new float[] { value });
		setSubMatrix(val, dst, rowIndex, columnIndex);
		val.release();
		return this;
	}

	/**
     * @see FloatMatrix#get(FloatMatrix, int, int)
     */
	@Override
	public float get(FloatMatrix src, int rowIndex, int columnIndex) {
    	CLFloatMatrix val = new CLFloatMatrix(1, 1, new float[] { 0 });
		getSubMatrix(src, val, rowIndex, columnIndex);
		float value = val.toArray()[0];
		val.release();
		return value;
	}

	/**
     * @see FloatMatrix#getRow(FloatMatrix, FloatMatrix, int)
     */
	@Override
	public FloatMatrix getRow(FloatMatrix src, FloatMatrix row, int rowIndex) {
		getSubMatrix(src, row, rowIndex, 0);
		return row;
	}

	/**
     * @see FloatMatrix#getColumn(FloatMatrix, FloatMatrix, int)
     */
	@Override
	public FloatMatrix getColumn(FloatMatrix src, FloatMatrix column, int columnIndex) {
		getSubMatrix(src, column, 0, columnIndex);
		return column;
	}
	
	/**
     * @see FloatMatrix#putRow(FloatMatrix, FloatMatrix, int)
     */
	@Override
	public FloatMatrix putRow(FloatMatrix dst, FloatMatrix row, int rowIndex) {
		setSubMatrix(row, dst, rowIndex, 0);
		return dst;
	}

	/**
     * @see FloatMatrix#putColumn(FloatMatrix, FloatMatrix, int)
     */
	@Override
	public FloatMatrix putColumn(FloatMatrix dst, FloatMatrix column, int columnIndex) {
		setSubMatrix(column, dst, 0, columnIndex);
		return dst;
	} 
 
}
