package org.nblas;

import org.nblas.cl.CLFloatMatrix;
import org.nblas.cuda.CudaFloatMatrix;
import org.nblas.java.JavaFloatMatrix;

/**
 * 
 * @author Nico
 *
 */
public interface FloatMatrixDefault extends FloatMatrix {
    
	/**
	 * Dirty allocation. Very fast but the content might not be 0. 
	 * 
	 * @param rows
	 * @param columns
	 * @param context
	 * @return
	 */
    public static FloatMatrix dirtyAllocation(int rows, int columns, Context context) {
    	
        if (context.isGPU()) {
            if (context.isCUDA()) {
            	return new CudaFloatMatrix(rows, columns);
            } else {
            	return new CLFloatMatrix(rows, columns);
            }
        } else {
            return new JavaFloatMatrix(rows, columns);
        }
    }
    
    // ------------------------------------- java getter methods --------------------------------------

    public default float[] toArray() {
        float[] values = new float[getRows() * getColumns()];
        readRowMajor(values);
        return values;
    }

	public default float[][] toArray2() {
        float[][] matrix = new float[getRows()][getColumns()];
        float[] array = toArray();
        for (int y = 0; y < getRows(); y++)
          	for (int x = 0; x < getColumns(); x++)
                matrix[y][x] = array[x * getRows() + y];
        return matrix;
    }

	
	
	// ------------------------------------- print methods --------------------------------------

    public default String toString2D() {
        StringBuilder builder = new StringBuilder();
        float[][] matrix = toArray2();
        int maxRows = Math.min(50, matrix.length);
        int maxColumns = Math.min(50, matrix[0].length);
        
        for (int y = 0; y < maxRows; y++) {
            builder.append("[");
            for (int x = 0; x < maxColumns; x++) {
                builder.append(String.format("%.6f", matrix[y][x]));
                if(x < maxColumns - 1) builder.append(", ");
            }
            builder.append("]");
            if(y < maxRows - 1) builder.append(", ");
            builder.append("\n");
        }

        return builder.toString();
    }
    
    // ------------------------------------- utility methods --------------------------------------
    public Context getContext();
    public void release();
    public boolean isReleased();
    
    public int getRows();
    public int getColumns();
    public FloatMatrix readRowMajor(float[] values);
    
    /**
     * copy the content of a to the result matrix
     * 
     * @param a
     * @param result
     * @return
     */
    public FloatMatrix dup(FloatMatrix a, FloatMatrix result);
    
    /**
     * create a copy of the this matrix
     * @return
     */
    public default FloatMatrix dup() {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	result.dup(this, result);
        return result;
    }
    
    /**
     * copy the content of this matrix to the result matrix
     * 
     * @param result
     * @return
     */
    public default FloatMatrix dup(FloatMatrix result) {
    	result.dup(this, result);
        return result;
    }
    
    public FloatMatrix transpose(FloatMatrix matrix, FloatMatrix transposed);
    
    public FloatMatrix repmat(FloatMatrix source, FloatMatrix destination, int rowMultiplicator, int columnMultiplicator);
    
    public default FloatMatrix repmat(int rowMultiplicator, int columnMultiplicator) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows() * rowMultiplicator, this.getColumns() * columnMultiplicator, this.getContext());
    	repmat(this, result, rowMultiplicator, columnMultiplicator);
    	return result;
    }
    
    
    
    // ---------------------------------- common inplace methods ----------------------------------
    
    public FloatMatrix setOne();
    public FloatMatrix setZero();
    public FloatMatrix randi();
    public FloatMatrix randni();
    
    
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- add methods ------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    
    /**
     * expert only
     * 
     * @param a
     * @param b
     * @param result
     * @return
     */
    public FloatMatrix add(FloatMatrix a, FloatMatrix b, FloatMatrix result);    
    
    public default FloatMatrix add(FloatMatrix b, FloatMatrix result) {
        add(this, b, result);
        return result;
    }

    public default FloatMatrix addi(FloatMatrix b) {
        add(this, b, this);
        return this;
    }

    public default FloatMatrix add(FloatMatrix b) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(b.getRows(), b.getColumns(), b.getContext());
        add(this, b, result);
        return result;
    }
    
    /**
     * expert only 
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix add(FloatMatrix a, float scalar, FloatMatrix result);
    
    public default FloatMatrix add(float scalar, FloatMatrix result) {
        add(this, scalar, result);
        return result;
    }

    public default FloatMatrix addi(float scalar) {
        add(this, scalar, this);
        return this;
    }

    public default FloatMatrix add(float scalar) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
        add(this, scalar, result);
        return result;
    }
    
    
    /**
     * expert only 
     * 
     * @param matrix
     * @param columnVector
     * @param result
     * @return
     */
    public FloatMatrix addColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result);
    
    public default FloatMatrix addColumnVector(FloatMatrix columnVector, FloatMatrix result) {
    	addColumnVector(this, columnVector, result);
        return result;
    }    

    public default FloatMatrix addiColumnVector(FloatMatrix columnVector) {
    	addColumnVector(this, columnVector, this);
        return this;
    }  
    
    public default FloatMatrix addColumnVector(FloatMatrix columnVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	addColumnVector(this, columnVector, result);
        return result;
    } 
    
    /**
     * expert only 
     * 
     * @param matrix
     * @param rowVector
     * @param result
     * @return
     */
    public FloatMatrix addRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result);
    
    public default FloatMatrix addRowVector(FloatMatrix rowVector, FloatMatrix result) {
    	addRowVector(this, rowVector, result);
        return result;
    }    

    public default FloatMatrix addiRowVector(FloatMatrix rowVector) {
    	addRowVector(this, rowVector, this);
        return this;
    }  
    
    public default FloatMatrix addRowVector(FloatMatrix rowVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	addRowVector(this, rowVector, result);
        return result;
    } 
    
    
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- sub methods ------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    
    /**
     * expert only
     * 
     * @param a
     * @param b
     * @param result
     * @return
     */
    public FloatMatrix sub(FloatMatrix a, FloatMatrix b, FloatMatrix result);    
    
    public default FloatMatrix sub(FloatMatrix b, FloatMatrix result) {
        sub(this, b, result);
        return result;
    }

    public default FloatMatrix subi(FloatMatrix b) {
        sub(this, b, this);
        return this;
    }

    public default FloatMatrix sub(FloatMatrix b) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(b.getRows(), b.getColumns(), b.getContext());
        sub(this, b, result);
        return result;
    }
    
    /**
     * expert only 
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix sub(FloatMatrix a, float scalar, FloatMatrix result);
    
    public default FloatMatrix sub(float scalar, FloatMatrix result) {
        sub(this, scalar, result);
        return result;
    }

    public default FloatMatrix subi(float scalar) {
        sub(this, scalar, this);
        return this;
    }

    public default FloatMatrix sub(float scalar) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
        sub(this, scalar, result);
        return result;
    }
    
    
    /**
     * expert only 
     * 
     * @param matrix
     * @param columnVector
     * @param result
     * @return
     */
    public FloatMatrix subColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result);
    
    public default FloatMatrix subColumnVector(FloatMatrix columnVector, FloatMatrix result) {
    	subColumnVector(this, columnVector, result);
        return result;
    }    

    public default FloatMatrix subiColumnVector(FloatMatrix columnVector) {
    	subColumnVector(this, columnVector, this);
        return this;
    }  
    
    public default FloatMatrix subColumnVector(FloatMatrix columnVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	subColumnVector(this, columnVector, result);
        return result;
    } 
    
    /**
     * expert only 
     * 
     * @param matrix
     * @param rowVector
     * @param result
     * @return
     */
    public FloatMatrix subRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result);
    
    public default FloatMatrix subRowVector(FloatMatrix rowVector, FloatMatrix result) {
    	subRowVector(this, rowVector, result);
        return result;
    }    

    public default FloatMatrix subiRowVector(FloatMatrix rowVector) {
    	subRowVector(this, rowVector, this);
        return this;
    }  
    
    public default FloatMatrix subRowVector(FloatMatrix rowVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	subRowVector(this, rowVector, result);
        return result;
    } 
    
    /**
     * expert only 
     * 
     * @param matrix
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix rsub(FloatMatrix matrix, float scalar, FloatMatrix result);
    
    public default FloatMatrix rsub(float scalar, FloatMatrix result) {
    	rsub(this, scalar, result);
        return result;
    }    

    public default FloatMatrix rsubi(float scalar) {
    	rsub(this, scalar, this);
        return this;
    }  
    
    public default FloatMatrix rsub(float scalar) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	rsub(this, scalar, result);
        return result;
    } 

    /**
     * expert only 
     * 
     * @param matrix
     * @param columnVector
     * @param result
     * @return
     */
    public FloatMatrix rsubColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result);

    public default FloatMatrix rsubColumnVector(FloatMatrix columnVector, FloatMatrix result) {
    	rsubColumnVector(this, columnVector, result);
        return result;
    }    

    public default FloatMatrix rsubiColumnVector(FloatMatrix columnVector) {
    	rsubColumnVector(this, columnVector, this);
        return this;
    }  
    
    public default FloatMatrix rsubColumnVector(FloatMatrix columnVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	rsubColumnVector(this, columnVector, result);
        return result;
    } 
    
    /**
     * expert only 
     * 
     * @param matrix
     * @param rowVector
     * @param result
     * @return
     */
    public FloatMatrix rsubRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result);
    
    public default FloatMatrix rsubRowVector(FloatMatrix rowVector, FloatMatrix result) {
    	rsubRowVector(this, rowVector, result);
        return result;
    }    

    public default FloatMatrix rsubiRowVector(FloatMatrix rowVector) {
    	rsubRowVector(this, rowVector, this);
        return this;
    }  
    
    public default FloatMatrix rsubRowVector(FloatMatrix rowVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	rsubRowVector(this, rowVector, result);
        return result;
    } 
    
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- mul methods ------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    
    /**
     * expert only
     * 
     * @param a
     * @param b
     * @param result
     * @return
     */
    public FloatMatrix mul(FloatMatrix a, FloatMatrix b, FloatMatrix result);    
    
    public default FloatMatrix mul(FloatMatrix b, FloatMatrix result) {
        mul(this, b, result);
        return result;
    }

    public default FloatMatrix muli(FloatMatrix b) {
        mul(this, b, this);
        return this;
    }

    public default FloatMatrix mul(FloatMatrix b) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(b.getRows(), b.getColumns(), b.getContext());
        mul(this, b, result);
        return result;
    }
    
    /**
     * expert only 
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix mul(FloatMatrix a, float scalar, FloatMatrix result);
    
    public default FloatMatrix mul(float scalar, FloatMatrix result) {
        mul(this, scalar, result);
        return result;
    }

    public default FloatMatrix muli(float scalar) {
        mul(this, scalar, this);
        return this;
    }

    public default FloatMatrix mul(float scalar) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
        mul(this, scalar, result);
        return result;
    }
    
    
    /**
     * expert only 
     * 
     * @param matrix
     * @param columnVector
     * @param result
     * @return
     */
    public FloatMatrix mulColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result);
    
    public default FloatMatrix mulColumnVector(FloatMatrix columnVector, FloatMatrix result) {
    	mulColumnVector(this, columnVector, result);
        return result;
    }    

    public default FloatMatrix muliColumnVector(FloatMatrix columnVector) {
    	mulColumnVector(this, columnVector, this);
        return this;
    }  
    
    public default FloatMatrix mulColumnVector(FloatMatrix columnVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	mulColumnVector(this, columnVector, result);
        return result;
    } 
    
    /**
     * expert only 
     * 
     * @param matrix
     * @param rowVector
     * @param result
     * @return
     */
    public FloatMatrix mulRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result);
    
    public default FloatMatrix mulRowVector(FloatMatrix rowVector, FloatMatrix result) {
    	mulRowVector(this, rowVector, result);
        return result;
    }    

    public default FloatMatrix muliRowVector(FloatMatrix rowVector) {
    	mulRowVector(this, rowVector, this);
        return this;
    }  
    
    public default FloatMatrix mulRowVector(FloatMatrix rowVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	mulRowVector(this, rowVector, result);
        return result;
    } 
    
    
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- div methods ------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    
    /**
     * expert only
     * 
     * @param a
     * @param b
     * @param result
     * @return
     */
    public FloatMatrix div(FloatMatrix a, FloatMatrix b, FloatMatrix result);    
    
    public default FloatMatrix div(FloatMatrix b, FloatMatrix result) {
        div(this, b, result);
        return result;
    }

    public default FloatMatrix divi(FloatMatrix b) {
        div(this, b, this);
        return this;
    }

    public default FloatMatrix div(FloatMatrix b) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(b.getRows(), b.getColumns(), b.getContext());
        div(this, b, result);
        return result;
    }
    
    /**
     * expert only 
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix div(FloatMatrix a, float scalar, FloatMatrix result);
    
    public default FloatMatrix div(float scalar, FloatMatrix result) {
        div(this, scalar, result);
        return result;
    }

    public default FloatMatrix divi(float scalar) {
        div(this, scalar, this);
        return this;
    }

    public default FloatMatrix div(float scalar) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
        div(this, scalar, result);
        return result;
    }
    
    
    /**
     * expert only 
     * 
     * @param matrix
     * @param columnVector
     * @param result
     * @return
     */
    public FloatMatrix divColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result);
    
    public default FloatMatrix divColumnVector(FloatMatrix columnVector, FloatMatrix result) {
    	divColumnVector(this, columnVector, result);
        return result;
    }    

    public default FloatMatrix diviColumnVector(FloatMatrix columnVector) {
    	divColumnVector(this, columnVector, this);
        return this;
    }  
    
    public default FloatMatrix divColumnVector(FloatMatrix columnVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	divColumnVector(this, columnVector, result);
        return result;
    } 
    
    /**
     * expert only 
     * 
     * @param matrix
     * @param rowVector
     * @param result
     * @return
     */
    public FloatMatrix divRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result);
    
    public default FloatMatrix divRowVector(FloatMatrix rowVector, FloatMatrix result) {
    	divRowVector(this, rowVector, result);
        return result;
    }    

    public default FloatMatrix diviRowVector(FloatMatrix rowVector) {
    	divRowVector(this, rowVector, this);
        return this;
    }  
    
    public default FloatMatrix divRowVector(FloatMatrix rowVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	divRowVector(this, rowVector, result);
        return result;
    } 
    
    /**
     * expert only 
     * 
     * @param matrix
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix rdiv(FloatMatrix matrix, float scalar, FloatMatrix result);
    
    public default FloatMatrix rdiv(float scalar, FloatMatrix result) {
    	rdiv(this, scalar, result);
        return result;
    }    

    public default FloatMatrix rdivi(float scalar) {
    	rdiv(this, scalar, this);
        return this;
    }  
    
    public default FloatMatrix rdiv(float scalar) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	rdiv(this, scalar, result);
        return result;
    } 

    /**
     * expert only 
     * 
     * @param matrix
     * @param columnVector
     * @param result
     * @return
     */
    public FloatMatrix rdivColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result);

    public default FloatMatrix rdivColumnVector(FloatMatrix columnVector, FloatMatrix result) {
    	rdivColumnVector(this, columnVector, result);
        return result;
    }    

    public default FloatMatrix rdiviColumnVector(FloatMatrix columnVector) {
    	rdivColumnVector(this, columnVector, this);
        return this;
    }  
    
    public default FloatMatrix rdivColumnVector(FloatMatrix columnVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	rdivColumnVector(this, columnVector, result);
        return result;
    } 
    
    /**
     * expert only 
     * 
     * @param matrix
     * @param rowVector
     * @param result
     * @return
     */
    public FloatMatrix rdivRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result);
    
    public default FloatMatrix rdivRowVector(FloatMatrix rowVector, FloatMatrix result) {
    	rdivRowVector(this, rowVector, result);
        return result;
    }    

    public default FloatMatrix rdiviRowVector(FloatMatrix rowVector) {
    	rdivRowVector(this, rowVector, this);
        return this;
    }  
    
    public default FloatMatrix rdivRowVector(FloatMatrix rowVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	rdivRowVector(this, rowVector, result);
        return result;
    } 
    
    // ----------------------------------------------------------------------------------------------------------
    // ------------------------------------------- exponential methods ------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    
    /**
     * expert only  
     * 
     * @param a
     * @param result
     * @return
     */
    public FloatMatrix exp(FloatMatrix a, FloatMatrix result);
	
	public default FloatMatrix exp(FloatMatrix result) {
        exp(this, result);
		return result;
	}
	
	public default FloatMatrix expi() {
        exp(this, this);
		return this;
	}

	public default FloatMatrix exp() {
		FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
        exp(this, result);
		return result;
	}
	
	
    // ----------------------------------------------------------------------------------------------------------
    // ------------------------------------------------ negate methods ------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
    /**
     * expert only   
     * 
     * @param a
     * @param result
     * @return
     */
    public FloatMatrix neg(FloatMatrix a, FloatMatrix result);

	public default FloatMatrix neg(FloatMatrix result) {
        neg(this, result);
		return result;
	}
	
	public default FloatMatrix negi() {
        neg(this, this);
		return this;
	}

	public default FloatMatrix neg() {
		FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
        neg(this, result);
		return result;
	}
	

    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- sigmoid methods ------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	    
    /**
     * expert only   
     * 
     * @param a
     * @param result
     * @return
     */
    public FloatMatrix sigmoid(FloatMatrix a, FloatMatrix result);

	public default FloatMatrix sigmoid(FloatMatrix result) {
        sigmoid(this, result);
		return result;
	}
	
	public default FloatMatrix sigmoidi() {
        sigmoid(this, this);
		return this;
	}
	
	public default FloatMatrix sigmoid() {
		FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
        sigmoid(this, result);
		return result;
	}
	
	

    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- greater than ---------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	    
	/**
	 * expert only  
	 * 
	 * @param a
	 * @param b
	 * @param result
	 * @return
	 */
    public FloatMatrix gt(FloatMatrix a, FloatMatrix b, FloatMatrix result);
    
    public default FloatMatrix gt(FloatMatrix b, FloatMatrix result) {
        gt(this, b, result);
        return result;
    }

    public default FloatMatrix gti(FloatMatrix b) {
        gt(this, b, this);
        return this;
    }

    public default FloatMatrix gt(FloatMatrix b) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(b.getRows(), b.getColumns(), b.getContext());
        gt(this, b, result);
        return result;
    }
    
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix gt(FloatMatrix a, float scalar, FloatMatrix result);

    public default FloatMatrix gt(float scalar, FloatMatrix result) {
        gt(this, scalar, result);
        return result;
    }

    public default FloatMatrix gti(float scalar) {
        gt(this, scalar, this);
        return this;
    }

    public default FloatMatrix gt(float scalar) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
        gt(this, scalar, result);
        return result;
    }
    
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix gtRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result);

    public default FloatMatrix gtRowVector(FloatMatrix rowVector, FloatMatrix result) {
    	gtRowVector(this, rowVector, result);
        return result;
    }

    public default FloatMatrix gtiRowVector(FloatMatrix rowVector) {
    	gtRowVector(this, rowVector, this);
        return this;
    }

    public default FloatMatrix gtRowVector(FloatMatrix rowVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	gtRowVector(this, rowVector, result);
        return result;
    }
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix gtColumnVector(FloatMatrix a, FloatMatrix ColumnVector, FloatMatrix result);

    public default FloatMatrix gtColumnVector(FloatMatrix ColumnVector, FloatMatrix result) {
    	gtColumnVector(this, ColumnVector, result);
        return result;
    }

    public default FloatMatrix gtiColumnVector(FloatMatrix ColumnVector) {
    	gtColumnVector(this, ColumnVector, this);
        return this;
    }

    public default FloatMatrix gtColumnVector(FloatMatrix ColumnVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getColumns(), this.getColumns(), this.getContext());
    	gtColumnVector(this, ColumnVector, result);
        return result;
    }
    
    
    
    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- greater or equal than ------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	    
	/**
	 * expert only  
	 * 
	 * @param a
	 * @param b
	 * @param result
	 * @return
	 */
    public FloatMatrix ge(FloatMatrix a, FloatMatrix b, FloatMatrix result);
    
    public default FloatMatrix ge(FloatMatrix b, FloatMatrix result) {
        ge(this, b, result);
        return result;
    }

    public default FloatMatrix gei(FloatMatrix b) {
        ge(this, b, this);
        return this;
    }

    public default FloatMatrix ge(FloatMatrix b) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(b.getRows(), b.getColumns(), b.getContext());
        ge(this, b, result);
        return result;
    }
    
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix ge(FloatMatrix a, float scalar, FloatMatrix result);

    public default FloatMatrix ge(float scalar, FloatMatrix result) {
        ge(this, scalar, result);
        return result;
    }

    public default FloatMatrix gei(float scalar) {
        ge(this, scalar, this);
        return this;
    }

    public default FloatMatrix ge(float scalar) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
        ge(this, scalar, result);
        return result;
    }
    
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix geRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result);

    public default FloatMatrix geRowVector(FloatMatrix rowVector, FloatMatrix result) {
    	geRowVector(this, rowVector, result);
        return result;
    }

    public default FloatMatrix geiRowVector(FloatMatrix rowVector) {
    	geRowVector(this, rowVector, this);
        return this;
    }

    public default FloatMatrix geRowVector(FloatMatrix rowVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	geRowVector(this, rowVector, result);
        return result;
    }
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix geColumnVector(FloatMatrix a, FloatMatrix ColumnVector, FloatMatrix result);

    public default FloatMatrix geColumnVector(FloatMatrix ColumnVector, FloatMatrix result) {
    	geColumnVector(this, ColumnVector, result);
        return result;
    }

    public default FloatMatrix geiColumnVector(FloatMatrix ColumnVector) {
    	geColumnVector(this, ColumnVector, this);
        return this;
    }

    public default FloatMatrix geColumnVector(FloatMatrix ColumnVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getColumns(), this.getColumns(), this.getContext());
    	geColumnVector(this, ColumnVector, result);
        return result;
    }
    
    
    
    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- less than ------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	    
	/**
	 * expert only  
	 * 
	 * @param a
	 * @param b
	 * @param result
	 * @return
	 */
    public FloatMatrix lt(FloatMatrix a, FloatMatrix b, FloatMatrix result);
    
    public default FloatMatrix lt(FloatMatrix b, FloatMatrix result) {
        lt(this, b, result);
        return result;
    }

    public default FloatMatrix lti(FloatMatrix b) {
        lt(this, b, this);
        return this;
    }

    public default FloatMatrix lt(FloatMatrix b) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(b.getRows(), b.getColumns(), b.getContext());
        lt(this, b, result);
        return result;
    }
    
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix lt(FloatMatrix a, float scalar, FloatMatrix result);

    public default FloatMatrix lt(float scalar, FloatMatrix result) {
        lt(this, scalar, result);
        return result;
    }

    public default FloatMatrix lti(float scalar) {
        lt(this, scalar, this);
        return this;
    }

    public default FloatMatrix lt(float scalar) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
        lt(this, scalar, result);
        return result;
    }
    
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix ltRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result);

    public default FloatMatrix ltRowVector(FloatMatrix rowVector, FloatMatrix result) {
    	ltRowVector(this, rowVector, result);
        return result;
    }

    public default FloatMatrix ltiRowVector(FloatMatrix rowVector) {
    	ltRowVector(this, rowVector, this);
        return this;
    }

    public default FloatMatrix ltRowVector(FloatMatrix rowVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	ltRowVector(this, rowVector, result);
        return result;
    }
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix ltColumnVector(FloatMatrix a, FloatMatrix ColumnVector, FloatMatrix result);

    public default FloatMatrix ltColumnVector(FloatMatrix ColumnVector, FloatMatrix result) {
    	ltColumnVector(this, ColumnVector, result);
        return result;
    }

    public default FloatMatrix ltiColumnVector(FloatMatrix ColumnVector) {
    	ltColumnVector(this, ColumnVector, this);
        return this;
    }

    public default FloatMatrix ltColumnVector(FloatMatrix ColumnVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getColumns(), this.getColumns(), this.getContext());
    	ltColumnVector(this, ColumnVector, result);
        return result;
    }
    
    
    
    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- less or equal than ---------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	    
	/**
	 * expert only  
	 * 
	 * @param a
	 * @param b
	 * @param result
	 * @return
	 */
    public FloatMatrix le(FloatMatrix a, FloatMatrix b, FloatMatrix result);
    
    public default FloatMatrix le(FloatMatrix b, FloatMatrix result) {
        le(this, b, result);
        return result;
    }

    public default FloatMatrix lei(FloatMatrix b) {
        le(this, b, this);
        return this;
    }

    public default FloatMatrix le(FloatMatrix b) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(b.getRows(), b.getColumns(), b.getContext());
        le(this, b, result);
        return result;
    }
    
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix le(FloatMatrix a, float scalar, FloatMatrix result);

    public default FloatMatrix le(float scalar, FloatMatrix result) {
        le(this, scalar, result);
        return result;
    }

    public default FloatMatrix lei(float scalar) {
        le(this, scalar, this);
        return this;
    }

    public default FloatMatrix le(float scalar) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
        le(this, scalar, result);
        return result;
    }
    
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix leRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result);

    public default FloatMatrix leRowVector(FloatMatrix rowVector, FloatMatrix result) {
    	leRowVector(this, rowVector, result);
        return result;
    }

    public default FloatMatrix leiRowVector(FloatMatrix rowVector) {
    	leRowVector(this, rowVector, this);
        return this;
    }

    public default FloatMatrix leRowVector(FloatMatrix rowVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	leRowVector(this, rowVector, result);
        return result;
    }
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix leColumnVector(FloatMatrix a, FloatMatrix ColumnVector, FloatMatrix result);

    public default FloatMatrix leColumnVector(FloatMatrix ColumnVector, FloatMatrix result) {
    	leColumnVector(this, ColumnVector, result);
        return result;
    }

    public default FloatMatrix leiColumnVector(FloatMatrix ColumnVector) {
    	leColumnVector(this, ColumnVector, this);
        return this;
    }

    public default FloatMatrix leColumnVector(FloatMatrix ColumnVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getColumns(), this.getColumns(), this.getContext());
    	leColumnVector(this, ColumnVector, result);
        return result;
    }
    
    
    // ----------------------------------------------------------------------------------------------------------
    // ------------------------------------------------- equal to -----------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	    
	/**
	 * expert only  
	 * 
	 * @param a
	 * @param b
	 * @param result
	 * @return
	 */
    public FloatMatrix eq(FloatMatrix a, FloatMatrix b, FloatMatrix result);
    
    public default FloatMatrix eq(FloatMatrix b, FloatMatrix result) {
        eq(this, b, result);
        return result;
    }

    public default FloatMatrix eqi(FloatMatrix b) {
        eq(this, b, this);
        return this;
    }

    public default FloatMatrix eq(FloatMatrix b) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(b.getRows(), b.getColumns(), b.getContext());
        eq(this, b, result);
        return result;
    }
    
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix eq(FloatMatrix a, float scalar, FloatMatrix result);

    public default FloatMatrix eq(float scalar, FloatMatrix result) {
        eq(this, scalar, result);
        return result;
    }

    public default FloatMatrix eqi(float scalar) {
        eq(this, scalar, this);
        return this;
    }

    public default FloatMatrix eq(float scalar) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
        eq(this, scalar, result);
        return result;
    }
    
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix eqRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result);

    public default FloatMatrix eqRowVector(FloatMatrix rowVector, FloatMatrix result) {
    	eqRowVector(this, rowVector, result);
        return result;
    }

    public default FloatMatrix eqiRowVector(FloatMatrix rowVector) {
    	eqRowVector(this, rowVector, this);
        return this;
    }

    public default FloatMatrix eqRowVector(FloatMatrix rowVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	eqRowVector(this, rowVector, result);
        return result;
    }
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix eqColumnVector(FloatMatrix a, FloatMatrix ColumnVector, FloatMatrix result);

    public default FloatMatrix eqColumnVector(FloatMatrix ColumnVector, FloatMatrix result) {
    	eqColumnVector(this, ColumnVector, result);
        return result;
    }

    public default FloatMatrix eqiColumnVector(FloatMatrix ColumnVector) {
    	eqColumnVector(this, ColumnVector, this);
        return this;
    }

    public default FloatMatrix eqColumnVector(FloatMatrix ColumnVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getColumns(), this.getColumns(), this.getContext());
    	eqColumnVector(this, ColumnVector, result);
        return result;
    }
    
    
    
    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- not equal to ---------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	    
	/**
	 * expert only  
	 * 
	 * @param a
	 * @param b
	 * @param result
	 * @return
	 */
    public FloatMatrix ne(FloatMatrix a, FloatMatrix b, FloatMatrix result);
    
    public default FloatMatrix ne(FloatMatrix b, FloatMatrix result) {
        ne(this, b, result);
        return result;
    }

    public default FloatMatrix nei(FloatMatrix b) {
        ne(this, b, this);
        return this;
    }

    public default FloatMatrix ne(FloatMatrix b) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(b.getRows(), b.getColumns(), b.getContext());
        ne(this, b, result);
        return result;
    }
    
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix ne(FloatMatrix a, float scalar, FloatMatrix result);

    public default FloatMatrix ne(float scalar, FloatMatrix result) {
        ne(this, scalar, result);
        return result;
    }

    public default FloatMatrix nei(float scalar) {
        ne(this, scalar, this);
        return this;
    }

    public default FloatMatrix ne(float scalar) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
        ne(this, scalar, result);
        return result;
    }
    
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix neRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result);

    public default FloatMatrix neRowVector(FloatMatrix rowVector, FloatMatrix result) {
    	neRowVector(this, rowVector, result);
        return result;
    }

    public default FloatMatrix neiRowVector(FloatMatrix rowVector) {
    	neRowVector(this, rowVector, this);
        return this;
    }

    public default FloatMatrix neRowVector(FloatMatrix rowVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), this.getColumns(), this.getContext());
    	neRowVector(this, rowVector, result);
        return result;
    }
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix neColumnVector(FloatMatrix a, FloatMatrix ColumnVector, FloatMatrix result);

    public default FloatMatrix neColumnVector(FloatMatrix ColumnVector, FloatMatrix result) {
    	neColumnVector(this, ColumnVector, result);
        return result;
    }

    public default FloatMatrix neiColumnVector(FloatMatrix ColumnVector) {
    	neColumnVector(this, ColumnVector, this);
        return this;
    }

    public default FloatMatrix neColumnVector(FloatMatrix ColumnVector) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getColumns(), this.getColumns(), this.getContext());
    	neColumnVector(this, ColumnVector, result);
        return result;
    }
    
    
    // ----------------------------------------------------------------------------------------------------------
    // ---------------------------------------- matrix multiplication -------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    
    /**
     * expert only  
     * 
     * @param a
     * @param b
     * @param result
     * @return
     */
    public FloatMatrix mmul(FloatMatrix a, FloatMatrix b, FloatMatrix result);
    
    public default FloatMatrix mmul(FloatMatrix b, FloatMatrix result) {
        mmul(this, b, result);
        return result;
    }

    public default FloatMatrix mmul(FloatMatrix b) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), b.getColumns(), this.getContext());
        mmul(this, b, result);
        return result;
    }
    
    /**
     * expert only  
     * 
     * @param a
     * @param b
     * @param result
     * @return
     */
    public FloatMatrix mmulTN(FloatMatrix a, FloatMatrix b, FloatMatrix result);
    
    public default FloatMatrix mmulTN(FloatMatrix b, FloatMatrix result) {
        mmulTN(this, b, result);
        return result;
    }

    public default FloatMatrix mmulTN(FloatMatrix b) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getColumns(), b.getColumns(), this.getContext());
        mmulTN(this, b, result);
        return result;
    }
    
    /**
     * expert only  
     * 
     * @param a
     * @param b
     * @param result
     * @return
     */
    public FloatMatrix mmulNT(FloatMatrix a, FloatMatrix b, FloatMatrix result);
    
    public default FloatMatrix mmulNT(FloatMatrix b, FloatMatrix result) {
        mmulNT(this, b, result);
        return result;
    }

    public default FloatMatrix mmulNT(FloatMatrix b) {
    	FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows(), b.getRows(), this.getContext());
        mmulNT(this, b, result);
        return result;
    }
    
    
    
    // ----------------------------------------------------------------------------------------------------------
    // ------------------------------------------- reduction methods --------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    
    /**
     * expert only  
     * 
     * @param a
     * @return
     */
	public float sum(FloatMatrix a);
	
	public default float sum() {
		return sum(this);
	}
	
    /**
     * expert only  
     * 
     * @param a
     * @return
     */
	public float mean(FloatMatrix a);
	
	public default float mean() {
		return mean(this);
	}
	
	/**
     * expert only  
     * 
     * @param a
     * @return
     */
	public float prod(FloatMatrix a);
	
	public default float prod() {
		return prod(this);
	}
	
	/**
     * expert only  
     * 
     * @param a
     * @return
     */
	public float max(FloatMatrix a);
	
	public default float max() {
		return max(this);
	}
	
	/**
     * expert only  
     * 
     * @param a
     * @return
     */
	public float min(FloatMatrix a);
	
	public default float min() {
		return min(this);
	}
	
	
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------- getter and setter methods ----------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	/**
	 * expert only  
	 * 
	 * @param a
	 * @param b
	 * @param rowOffset
	 * @param columnOffset
	 * @return
	 */
	public FloatMatrix setSubMatrix(FloatMatrix src, FloatMatrix dst, int rowOffset, int columnOffset);
	
	public default FloatMatrix setSubMatrix(FloatMatrix src, int rowOffset, int columnOffset) {
		setSubMatrix(src, this, rowOffset, columnOffset);
		return this;
	}
	
	/**
	 * expert only  
	 * 
	 * @param a
	 * @param b
	 * @param rowOffset
	 * @param columnOffset
	 * @return
	 */
	public FloatMatrix getSubMatrix(FloatMatrix src, FloatMatrix dst, int rowOffset, int columnOffset);
    
	public default FloatMatrix getSubMatrix(int rowOffset, int columnOffset, int rows, int columns) {
		FloatMatrix result = FloatMatrixDefault.dirtyAllocation(rows, columns, this.getContext());
		getSubMatrix(this, result, rowOffset, columnOffset);
		return result;
	}
	
	public default FloatMatrix getSubMatrix(FloatMatrix dst, int rowOffset, int columnOffset) {
		return getSubMatrix(this, dst, rowOffset, columnOffset);
	}
	
	public default FloatMatrix getSubMatrix(int rowOffset, int columnOffset) {
		FloatMatrix result = FloatMatrixDefault.dirtyAllocation(this.getRows()-rowOffset, this.getColumns()-columnOffset, this.getContext());
		getSubMatrix(this, result, rowOffset, columnOffset);
		return result;
	}
	
	/**
	 * expert only 
	 * 
	 * @param rowIndex
	 * @param columnIndex
	 * @param value
	 * @return
	 */
	public FloatMatrix put(FloatMatrix src, int rowIndex, int columnIndex, float value);
	
	public default FloatMatrix put(int rowIndex, int columnIndex, float value) {
		return put(this, rowIndex, columnIndex, value);
	}
	
	/**
	 * expert only 
	 * 
	 * @param rowIndex
	 * @param columnIndex
	 * @return
	 */
	public float get(FloatMatrix src, int rowIndex, int columnIndex);
	
	public default float get(int rowIndex, int columnIndex) {
		return get(this, rowIndex, columnIndex);
	}
	
	/**
	 * expert only 
	 * 
	 * @param src
	 * @param row
	 * @param rowIndex
	 * @return
	 */
	public FloatMatrix getRow(FloatMatrix src, FloatMatrix row, int rowIndex);
	
	public default FloatMatrix getRow(FloatMatrix row, int rowIndex) {
		return getRow(this, row, rowIndex);
	}
	
	public default FloatMatrix getRow(int rowIndex) {
		FloatMatrix row = FloatMatrixDefault.dirtyAllocation(1, this.getColumns(), this.getContext());
		return getRow(this, row, rowIndex);
	}
	
	/**
	 * expert only 
	 * 
	 * @param src
	 * @param column
	 * @param columnIndex
	 * @return
	 */
	public FloatMatrix getColumn(FloatMatrix src, FloatMatrix column, int columnIndex);
	
	public default FloatMatrix getColumn(FloatMatrix column, int columnIndex) {
		return getColumn(this, column, columnIndex);
	}
	
	public default FloatMatrix getColumn(int columnIndex) {
		FloatMatrix column = FloatMatrixDefault.dirtyAllocation(this.getColumns(), 1, this.getContext());
		return getColumn(this, column, columnIndex);
	}

	/**
	 * expert only 
	 * 
	 * @param dst
	 * @param row
	 * @param rowIndex
	 * @return
	 */
	public FloatMatrix putRow(FloatMatrix dst, FloatMatrix row, int rowIndex);
	
	public default FloatMatrix putRow(FloatMatrix row, int rowIndex) {
		return putRow(this, row, rowIndex);
	}
	
	/**
	 * expert only 
	 * 
	 * @param dst
	 * @param column
	 * @param columnIndex
	 * @return
	 */
	public FloatMatrix putColumn(FloatMatrix dst, FloatMatrix column, int columnIndex);
	
	public default FloatMatrix putColumn(FloatMatrix column, int columnIndex) {
		return putColumn(this, column, columnIndex);
	}
	
}
