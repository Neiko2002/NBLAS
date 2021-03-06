package org.nblas.cl;


import org.jocl.cl_kernel;
import org.nblas.FloatMatrix;
import org.nblas.cl.blas.CLLevel1;
import org.nblas.cl.model.CLArray;
import org.nblas.cl.model.CLScalar;
import org.nblas.generic.Subprogram;
import org.nblas.impl.FloatMatrixDefault;

/**
 * 
 * TODO: addColumnVector usw. müssen noch in FloatMatrix
 * 
 * @author Nico
 *
 */
public class CLFloatMatrix extends CLMatrix implements FloatMatrixDefault {

	protected CLContext context;
	protected CLLevel1 level1;
	
	/**
	 * dirty allocation
	 * 
	 * @param rows
	 * @param columns
	 */
    public CLFloatMatrix(int rows, int columns, CLContext context) {
        super(rows, columns, CLCore.getCore(context.getDeviceId()));
 		this.clMemory = CORE.mallocSinglePrecision(clRows * clColumns);
 		this.context = context;
 		this.level1 = context.getLevel1();
    }

    public CLFloatMatrix(int rows, int columns, float[] values, CLContext context) {
       super(rows, columns, CLCore.getCore(context.getDeviceId()));

		if (rows * columns != values.length)
			throw new IllegalArgumentException("rows times columns " + (rows * columns) + " != " + "data length = " + values.length);

        float[] clValues = getFloatArray2D(values);
		this.clMemory = CORE.malloc(clValues);
 		this.context = context;
 		this.level1 = context.getLevel1();
    }
    
    private float[] getFloatArray2D(float[] values) {
        float[] clValues = new float[clRows * clColumns];
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
	public CLContext getContext() {
		return context;
	}  	
	
	/**
	  * @see FloatMatrix#readRowMajor(float[])
	  */
	@Override
	public FloatMatrix readRowMajor(float[] values) {
		
		float[] clValues = new float[clRows * clColumns];
		CORE.getData(this.clMemory, clValues);
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
    	CLFloatMatrix src = (CLFloatMatrix) matrix;
    	CLFloatMatrix dst = (CLFloatMatrix) transposed;
    	
    	Subprogram<cl_kernel> subprogram = CLPredefined.getSubprogram("transpose");
      	CORE.execute(subprogram, src.clRows, src.clColumns, src, dst, CLScalar.of(src.rows), CLScalar.of(src.columns));
      	
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
        CORE.execute(CLPredefined.getSubprogram("setZero"), this.clRows, this.clColumns, this);
        return this;
    }
    
    /**
	 * @see FloatMatrix#randi()
	 */
    @Override
    public FloatMatrix randi() {
        initRandom();
        Subprogram<cl_kernel> subprogram = CLPredefined.getSubprogram("auniform");
      	CORE.execute(subprogram, CORE.getThreadCountY(), CORE.getThreadCountX(), randomDataPointer.get(), this,
    			CLScalar.of(this.clRows), CLScalar.of(this.rows), CLScalar.of(this.columns));

        return this;
    }

    /**
	 * @see FloatMatrix#randni()
	 */
    @Override
    public FloatMatrix randni() {
        initRandom();        
        Subprogram<cl_kernel> subprogram = CLPredefined.getSubprogram("boxmuller");
      	CORE.execute(subprogram, CORE.getThreadCountY(), CORE.getThreadCountX(), randomDataPointer.get(), this,
    			CLScalar.of(this.clRows), CLScalar.of(this.rows), CLScalar.of(this.columns));
      	
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
    	level1.addScalar((CLFloatMatrix)matrix, CLScalar.of(scalar), (CLFloatMatrix)result);
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
    	level1.subScalar((CLFloatMatrix)matrix, CLScalar.of(scalar), (CLFloatMatrix)result);
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
    	level1.rsubScalar((CLFloatMatrix)matrix, CLScalar.of(scalar), (CLFloatMatrix)result);
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
    	level1.mulScalar((CLFloatMatrix)matrix, CLScalar.of(scalar), (CLFloatMatrix)result);
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
    	level1.divScalar((CLFloatMatrix)a, CLScalar.of(scalar), (CLFloatMatrix)result);
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
    	level1.rdivScalar((CLFloatMatrix)matrix, CLScalar.of(scalar), (CLFloatMatrix)result);
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
    	level1.gtScalar((CLFloatMatrix)matrix, CLScalar.of(scalar), (CLFloatMatrix)result);
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
    	level1.geScalar((CLFloatMatrix)matrix, CLScalar.of(scalar), (CLFloatMatrix)result);
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
    	level1.ltScalar((CLFloatMatrix)matrix, CLScalar.of(scalar), (CLFloatMatrix)result);
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
    	level1.leScalar((CLFloatMatrix)matrix, CLScalar.of(scalar), (CLFloatMatrix)result);
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
    	level1.eqScalar((CLFloatMatrix)matrix, CLScalar.of(scalar), (CLFloatMatrix)result);
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
    	level1.neScalar((CLFloatMatrix)matrix, CLScalar.of(scalar), (CLFloatMatrix)result);
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

    
    public FloatMatrix mmulCustom(Subprogram<cl_kernel> subprogram, FloatMatrix a, FloatMatrix b, FloatMatrix result) {
    	CLFloatMatrix matrixA = (CLFloatMatrix) a;
    	CLFloatMatrix matrixB = (CLFloatMatrix) b;
    	CLFloatMatrix matrixR = (CLFloatMatrix) result;    	
    	
    	CORE.execute(subprogram, matrixA.clRows, matrixB.clColumns, matrixA, matrixB, matrixR, 
      			CLArray.ofFloat(CORE.getThreadCount()), CLArray.ofFloat(CORE.getThreadCount()),
    			CLScalar.of(matrixA.clRows), CLScalar.of(matrixB.clColumns), CLScalar.of(matrixA.clColumns));
           
       return result;
    }
    
    /**
  	 * @see FloatMatrix#mmul(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix mmul(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
    	CLFloatMatrix matrixA = (CLFloatMatrix) a;
    	CLFloatMatrix matrixB = (CLFloatMatrix) b;
    	CLFloatMatrix matrixR = (CLFloatMatrix) result;
    	
    	Subprogram<cl_kernel> subprogram = CLPredefined.getSubprogram("sgemm_nn");
      	CORE.execute(subprogram, matrixA.clRows, matrixB.clColumns, matrixA, matrixB, matrixR, 
      			CLArray.ofFloat(CORE.getThreadCount()), CLArray.ofFloat(CORE.getThreadCount()),
    			CLScalar.of(matrixA.clRows), CLScalar.of(matrixB.clColumns), CLScalar.of(matrixA.clColumns));
      	
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
    	
     	Subprogram<cl_kernel> subprogram = CLPredefined.getSubprogram("sgemm_tn");
      	CORE.execute(subprogram, matrixA.clColumns, matrixB.clColumns, matrixA, matrixB, matrixR, 
      			CLArray.ofFloat(CORE.getThreadCount()), CLArray.ofFloat(CORE.getThreadCount()),
    			CLScalar.of(matrixA.clColumns), CLScalar.of(matrixB.clColumns), CLScalar.of(matrixA.clRows));
      	
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
    	
     	Subprogram<cl_kernel> subprogram = CLPredefined.getSubprogram("sgemm_nt");
      	CORE.execute(subprogram, matrixA.clRows, matrixB.clRows, matrixA, matrixB, matrixR, 
      			CLArray.ofFloat(CORE.getThreadCount()), CLArray.ofFloat(CORE.getThreadCount()),
    			CLScalar.of(matrixA.clRows), CLScalar.of(matrixB.clRows), CLScalar.of(matrixA.clColumns));

        return result;
    }

    
    
    // --------------------------------------- reduction methods ----------------------------------------

    /**
  	 * @see FloatMatrix#sum(FloatMatrix)
  	 */
    @Override
    public float sum(FloatMatrix matrix) {
    	CLFloatMatrix mat = (CLFloatMatrix) matrix;
//        return CORE.reduce2D("sumFloats", mat, mat.rows, mat.columns, 0);
        return CORE.reduce1D("sumFloats1D", mat, mat.clRows*mat.clColumns);
    }
    
    @Override
    public float mean(FloatMatrix matrix) {
    	CLFloatMatrix mat = (CLFloatMatrix) matrix;
        return mat.sum(matrix) / mat.getLength();
    }

    @Override
    public float prod(FloatMatrix matrix) {
    	CLFloatMatrix mat = (CLFloatMatrix) matrix;
        return CORE.reduce2D("productFloats", mat, mat.rows, mat.columns, 1);
    }

    @Override
    public float max(FloatMatrix matrix) {
    	CLFloatMatrix mat = (CLFloatMatrix) matrix;
        return CORE.reduce2D("maxFloats", mat, mat.rows, mat.columns, Float.NEGATIVE_INFINITY);
    }

    @Override
    public float min(FloatMatrix matrix) {
    	CLFloatMatrix mat = (CLFloatMatrix) matrix;
        return CORE.reduce2D("minFloats", mat, mat.rows, mat.columns, Float.POSITIVE_INFINITY);
    }
    
//    public static CLFloatMatrix testsum(CLFloatMatrix matrix) {
//        int tempSizeX = (int) Math.ceil((double) matrix.rows / 32);
//        int tempSizeY = (int) Math.ceil((double) matrix.columns / 32);
//        CLFloatMatrix result = new CLFloatMatrix(tempSizeX, tempSizeY);
//        CORE.reduce2D("sumFloats", matrix, result, matrix.rows, matrix.columns, result.rows, result.columns, 0);
//        return result;
//    }
//    
//    // --------------------------------------- column reduction methods ----------------------------------------
//
//    public static void columnSums(CLFloatMatrix matrix, CLFloatMatrix result) {
//        CORE.reduceColumns("columnSumsFloats", matrix, result, matrix.rows, matrix.columns, result.clRows, 0);
//    }
//
//    public static void columnMeans(CLFloatMatrix matrix, CLFloatMatrix result) {
//        columnSums(matrix, result);
//        result.div(result, matrix.rows, result);
//    }
//
//    public static void columnProds(CLFloatMatrix matrix, CLFloatMatrix result) {
//        CORE.reduceColumns("columnProductsFloats", matrix, result, matrix.rows, matrix.columns, result.clRows, 1);
//    }
//
//    public static void columnMaxs(CLFloatMatrix matrix, CLFloatMatrix result) {
//        CORE.reduceColumns("columnMaxsFloats", matrix, result, matrix.rows, matrix.columns, result.clRows, Float.NEGATIVE_INFINITY);
//    }
//
//    public static void columnMins(CLFloatMatrix matrix, CLFloatMatrix result) {
//        CORE.reduceColumns("columnMinsFloats", matrix, result, matrix.rows, matrix.columns, result.clRows, Float.POSITIVE_INFINITY);
//    }
//
//
//    
//    // --------------------------------------- row reduction methods ----------------------------------------
//
//    public static void rowSums(CLFloatMatrix matrix, CLFloatMatrix result) {
//        CORE.reduceRows("rowSumsFloats", matrix, result, matrix.rows, matrix.columns, 0);
//    }
//
//    public static void rowMeans(CLFloatMatrix matrix, CLFloatMatrix result) {
//        rowSums(matrix, result);
//        result.div(result, matrix.columns, result);
//    }
//
//    public static void rowProds(CLFloatMatrix matrix, CLFloatMatrix result) {
//        CORE.reduceRows("rowProductsFloats", matrix, result, matrix.rows, matrix.columns, 1);
//    }
//
//    public static void rowMaxs(CLFloatMatrix matrix, CLFloatMatrix result) {
//        CORE.reduceRows("rowMaxsFloats", matrix, result, matrix.rows, matrix.columns, Float.NEGATIVE_INFINITY);
//    }
//
//    public static void rowMins(CLFloatMatrix matrix, CLFloatMatrix result) {
//        CORE.reduceRows("rowMinsFloats", matrix, result, matrix.rows, matrix.columns, Float.POSITIVE_INFINITY);
//    }
    

	// --------------------------------------- getter and setter methods ----------------------------------------
    
    /**
  	 * @see FloatMatrix#repmat(FloatMatrix, FloatMatrix, int, int)
  	 */
    @Override
    public FloatMatrix repmat(FloatMatrix source, FloatMatrix destination, int rowMultiplicator, int columnMultiplicator) {
    	CLFloatMatrix src = (CLFloatMatrix) source;
    	CLFloatMatrix dst = (CLFloatMatrix)destination;
        
        Subprogram<cl_kernel> subprogram = CLPredefined.getSubprogram("repmat");
    	CORE.execute(subprogram, dst.clRows, dst.clColumns, src, dst, CLScalar.of(rows), CLScalar.of(columns), CLScalar.of(clRows));
    	
        return destination;
    }
    
    /**
  	 * @see FloatMatrix#setSubMatrix(FloatMatrix, FloatMatrix, int, int)
  	 */
    @Override
    public FloatMatrix setSubMatrix(FloatMatrix source, FloatMatrix destination, int rowOffset, int columnOffset) {
    	CLFloatMatrix src = (CLFloatMatrix) source;
    	CLFloatMatrix dst = (CLFloatMatrix) destination;
    	
    	Subprogram<cl_kernel> subprogram = CLPredefined.getSubprogram("setSubMatrix");
    	CORE.execute(subprogram, src.clRows, src.clColumns, src, dst,
    			CLScalar.of(src.rows), CLScalar.of(src.columns), 
    			CLScalar.of(rowOffset), CLScalar.of(columnOffset), 
    			CLScalar.of(dst.clRows));
    	
    	return destination;
    }

    /**
  	 * @see FloatMatrix#getSubMatrix(FloatMatrix, FloatMatrix, int, int)
  	 */
    @Override
    public FloatMatrix getSubMatrix(FloatMatrix source, FloatMatrix destination, int rowOffset, int columnOffset) {
    	CLFloatMatrix src = (CLFloatMatrix) source;
    	CLFloatMatrix dst = (CLFloatMatrix) destination;
    		
    	Subprogram<cl_kernel> subprogram = CLPredefined.getSubprogram("getSubMatrix");
      	CORE.execute(subprogram, dst.clRows, dst.clColumns, src, dst,
    			CLScalar.of(rowOffset+dst.rows), CLScalar.of(columnOffset+dst.columns), 
    			CLScalar.of(rowOffset), CLScalar.of(columnOffset), 
    			CLScalar.of(src.clRows));
      	
        return destination;
    }
    

    public FloatMatrix getCustom(Subprogram<cl_kernel> subprogram, FloatMatrix source, FloatMatrix destination, int rowOffset, int columnOffset) {
    	CLFloatMatrix src = (CLFloatMatrix) source;
    	CLFloatMatrix dst = (CLFloatMatrix) destination;    	
    	CORE.execute(subprogram, dst.clRows, dst.clColumns, src, dst,
    			CLScalar.of(rowOffset+dst.rows), CLScalar.of(columnOffset+dst.columns),
    			CLScalar.of(rowOffset), CLScalar.of(columnOffset),
    			CLScalar.of(src.clRows));
        
        return destination;
    }

    /**
     * @see FloatMatrix#put(FloatMatrix, int, int, float)
     */
	@Override
	public FloatMatrix put(FloatMatrix dst, int rowIndex, int columnIndex, float value) {
    	CLFloatMatrix val = new CLFloatMatrix(1, 1, new float[] { value }, context);
		setSubMatrix(val, dst, rowIndex, columnIndex);
		val.release();
		return this;
	}

	/**
     * @see FloatMatrix#get(FloatMatrix, int, int)
     */
	@Override
	public float get(FloatMatrix src, int rowIndex, int columnIndex) {
    	CLFloatMatrix val = new CLFloatMatrix(1, 1, new float[] { 0 }, context);
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
