package org.nblas;

import org.nblas.cl.CLFloatMatrix;
import org.nblas.cuda.CudaFloatMatrix;
import org.nblas.java.JavaFloatMatrix;

public interface FloatMatrix {
    
	/**
	 * dirty allocation 
	 * 
	 * TODO: sollte nach außen hin nicht sichtbar sein
	 * 
	 * @param rows
	 * @param columns
	 * @param context
	 * @return
	 */
    public static FloatMatrix create(int rows, int columns, Context context) {
    	
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
    
    public static FloatMatrix create(float[][] values, Context context) {
    	
    	// flat representation in column-major order
    	int rows = values.length;
    	int columns = values[0].length;
    	float[] flat = new float[rows * columns];
    	for (int c = 0; c < columns; c++)
    		for (int r = 0; r < rows; r++)
    			flat[r + c * rows] = values[r][c];
    	
        return create(rows, columns, flat, context);
    }
    
    public static FloatMatrix create(int rows, int columns, float[] values, Context context) {
    	
        if (context.isGPU()) {
            if (context.isCUDA()) {
            	return new CudaFloatMatrix(rows, columns, values);
            } else {
            	return new CLFloatMatrix(rows, columns, values);
            }
        } else {
            return new JavaFloatMatrix(rows, columns, values);
        }
    }

    public static FloatMatrix zeros(int rows, int columns, Context context) {
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
    
    public static FloatMatrix ones(int rows, int columns, Context context) {
    	FloatMatrix matrix = zeros(rows, columns, context);
    	matrix.setOne();
    	return matrix;
    }
    
    
    public static FloatMatrix rand(int rows, int columns, Context context) {
      	FloatMatrix matrix = zeros(rows, columns, context);
    	matrix.randi();
    	return matrix;
    }
    
    public static FloatMatrix randn(int rows, int columns, Context context) {
      	FloatMatrix matrix = zeros(rows, columns, context);
    	matrix.randni();
    	return matrix;
    }
    
    
    
    // ------------------------------------- java getter methods --------------------------------------

    public default float[] toArray() {
        float[] values = new float[getRows() * getColumns()];
        getColumnWiseOn(values);
        return values;
    }

	public default float[][] toArray2() {
        float[][] matrix = new float[getRows()][getColumns()];
        float[] array = toArray();
        for (int i = 0; i < getRows(); i++) {
            for (int j = 0; j < getColumns(); j++) {
                matrix[i][j] = array[i + j * getRows()];
            }
        }
        return matrix;
    }

	
	
	// ------------------------------------- print methods --------------------------------------
	
    public default String toString1D() {
        StringBuilder builder = new StringBuilder();
        float[][] matrix = toArray2();
        builder.append("[");
        for (int i = 0; i < getRows() - 1; i++) {
            for (int j = 0; j < getColumns() - 1; j++) {
                builder.append(String.format("%.6f", matrix[i][j]));
                builder.append(", ");
            }

            builder.append(String.format("%.6f", matrix[i][getColumns() - 1]));
            builder.append("; ");
        }
        for (int j = 0; j < getColumns() - 1; j++) {
            builder.append(String.format("%.6f", matrix[getRows() - 1][j]));
            builder.append(", ");
        }
        builder.append(String.format("%.6f", matrix[getRows() - 1][getColumns() - 1]));
        builder.append("]");

        return builder.toString();
    }

    public default String toString2D() {
        StringBuilder builder = new StringBuilder();
        float[][] matrix = toArray2();
        for (int i = 0; i < getRows() - 1; i++) {
            builder.append("[");
            for (int j = 0; j < getColumns() - 1; j++) {
                builder.append(String.format("%.6f", matrix[i][j]));
                builder.append(", ");
            }

            builder.append(String.format("%.6f", matrix[i][getColumns() - 1]));
            builder.append("]\n");
        }

        builder.append("[");
        for (int j = 0; j < getColumns() - 1; j++) {
            builder.append(String.format("%.6f", matrix[getRows() - 1][j]));
            builder.append(", ");
        }
        builder.append(String.format("%.6f", matrix[getRows() - 1][getColumns() - 1]));
        builder.append("]\n");

        return builder.toString();
    }
    
    // ------------------------------------- utility methods --------------------------------------
    public Context getContext();
    public void free();
    public boolean isReleased();
    
    public int getRows();
    public int getColumns();
    public FloatMatrix getColumnWiseOn(float[] values);
    
    public FloatMatrix dup(FloatMatrix a, FloatMatrix result);
    
    public default FloatMatrix dup() {
    	FloatMatrix result = FloatMatrix.zeros(this.getRows(), this.getColumns(), this.getContext());
    	result.dup(this, result);
        return result;
    }
    
    public default FloatMatrix dup(FloatMatrix result) {
    	result.dup(this, result);
        return result;
    }
    
    public FloatMatrix transpose(FloatMatrix matrix, FloatMatrix transposed);
    
    // ---------------------------------- common inplace methods ----------------------------------
    
    public FloatMatrix setOne();
    public FloatMatrix setZero();
    public FloatMatrix randi();
    public FloatMatrix randni();
    
    
	// --------------------------------------- add methods ----------------------------------------
    /**
     * expert only
     * 
     * @param a
     * @param b
     * @param result
     * @return
     */
    public FloatMatrix add(FloatMatrix a, FloatMatrix b, FloatMatrix result);    
    
    /**
     * expert only 
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix add(FloatMatrix a, float scalar, FloatMatrix result);

    public default FloatMatrix add(FloatMatrix b, FloatMatrix result) {
        add(this, b, result);
        return result;
    }

    public default FloatMatrix addi(FloatMatrix b) {
        add(this, b, this);
        return this;
    }

    public default FloatMatrix add(FloatMatrix b) {
    	FloatMatrix result = FloatMatrix.zeros(b.getRows(), b.getColumns(), b.getContext());
        add(this, b, result);
        return result;
    }
    
    public default FloatMatrix add(float scalar, FloatMatrix result) {
        add(this, scalar, result);
        return result;
    }

    public default FloatMatrix addi(float scalar) {
        add(this, scalar, this);
        return this;
    }

    public default FloatMatrix add(float scalar) {
    	FloatMatrix result = FloatMatrix.zeros(this.getRows(), this.getColumns(), this.getContext());
        add(this, scalar, result);
        return result;
    }
        
    
    
    // --------------------------------------- sub methods ----------------------------------------
    /**
     * expert only
     * 
     * @param a
     * @param b
     * @param result
     * @return
     */
    public FloatMatrix sub(FloatMatrix a, FloatMatrix b, FloatMatrix result);    
    
    /**
     * expert only 
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix sub(FloatMatrix a, float scalar, FloatMatrix result);

    public default FloatMatrix sub(FloatMatrix b, FloatMatrix result) {
        sub(this, b, result);
        return result;
    }

    public default FloatMatrix subi(FloatMatrix b) {
        sub(this, b, this);
        return this;
    }

    public default FloatMatrix sub(FloatMatrix b) {
    	FloatMatrix result = FloatMatrix.zeros(b.getRows(), b.getColumns(), b.getContext());
        sub(this, b, result);
        return result;
    }
    
    public default FloatMatrix sub(float scalar, FloatMatrix result) {
        sub(this, scalar, result);
        return result;
    }

    public default FloatMatrix subi(float scalar) {
        sub(this, scalar, this);
        return this;
    }

    public default FloatMatrix sub(float scalar) {
    	FloatMatrix result = FloatMatrix.zeros(this.getRows(), this.getColumns(), this.getContext());
        sub(this, scalar, result);
        return result;
    }
    
    
    
	// --------------------------------------- mul methods ----------------------------------------
    /**
     * expert only
     * 
     * @param a
     * @param b
     * @param result
     * @return
     */
    public FloatMatrix mul(FloatMatrix a, FloatMatrix b, FloatMatrix result);    
    
    /**
     * expert only 
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix mul(FloatMatrix a, float scalar, FloatMatrix result);

    public default FloatMatrix mul(FloatMatrix b, FloatMatrix result) {
        mul(this, b, result);
        return result;
    }

    public default FloatMatrix muli(FloatMatrix b) {
        mul(this, b, this);
        return this;
    }

    public default FloatMatrix mul(FloatMatrix b) {
    	FloatMatrix result = FloatMatrix.zeros(b.getRows(), b.getColumns(), b.getContext());
        mul(this, b, result);
        return result;
    }
    
    public default FloatMatrix mul(float scalar, FloatMatrix result) {
        mul(this, scalar, result);
        return result;
    }

    public default FloatMatrix muli(float scalar) {
        mul(this, scalar, this);
        return this;
    }

    public default FloatMatrix mul(float scalar) {
    	FloatMatrix result = FloatMatrix.zeros(this.getRows(), this.getColumns(), this.getContext());
        mul(this, scalar, result);
        return result;
    }
    
    
    
    // --------------------------------------- div methods ----------------------------------------
    /**
     * expert only
     * 
     * @param a
     * @param b
     * @param result
     * @return
     */
    public FloatMatrix div(FloatMatrix a, FloatMatrix b, FloatMatrix result);    
    
    /**
     * expert only 
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix div(FloatMatrix a, float scalar, FloatMatrix result);

    public default FloatMatrix div(FloatMatrix b, FloatMatrix result) {
        div(this, b, result);
        return result;
    }

    public default FloatMatrix divi(FloatMatrix b) {
        div(this, b, this);
        return this;
    }

    public default FloatMatrix div(FloatMatrix b) {
    	FloatMatrix result = FloatMatrix.zeros(b.getRows(), b.getColumns(), b.getContext());
        div(this, b, result);
        return result;
    }
    
    public default FloatMatrix div(float scalar, FloatMatrix result) {
        div(this, scalar, result);
        return result;
    }

    public default FloatMatrix divi(float scalar) {
        div(this, scalar, this);
        return this;
    }

    public default FloatMatrix div(float scalar) {
    	FloatMatrix result = FloatMatrix.zeros(this.getRows(), this.getColumns(), this.getContext());
        div(this, scalar, result);
        return result;
    }
    
    
    // --------------------------------------- exp methods ----------------------------------------
    
    /**
     * expert only  
     * 
     * @param a
     * @param result
     * @return
     */
    public FloatMatrix exp(FloatMatrix a, FloatMatrix result);

	public default FloatMatrix exp() {
		FloatMatrix result = FloatMatrix.zeros(this.getRows(), this.getColumns(), this.getContext());
        exp(this, result);
		return result;
	}
	
	public default FloatMatrix expi() {
        exp(this, this);
		return this;
	}
	
	public default FloatMatrix exp(FloatMatrix result) {
        exp(this, result);
		return result;
	}
	
	
    // --------------------------------------- neg methods ----------------------------------------
    
    /**
     * expert only   
     * 
     * @param a
     * @param result
     * @return
     */
    public FloatMatrix neg(FloatMatrix a, FloatMatrix result);

	public default FloatMatrix neg() {
		FloatMatrix result = FloatMatrix.zeros(this.getRows(), this.getColumns(), this.getContext());
        neg(this, result);
		return result;
	}
	
	public default FloatMatrix negi() {
        neg(this, this);
		return this;
	}
	
	public default FloatMatrix neg(FloatMatrix result) {
        neg(this, result);
		return result;
	}
	

    // --------------------------------------- sigmoid methods ----------------------------------------
    
    /**
     * expert only   
     * 
     * @param a
     * @param result
     * @return
     */
    public FloatMatrix sigmoid(FloatMatrix a, FloatMatrix result);

	public default FloatMatrix sigmoid() {
		FloatMatrix result = FloatMatrix.zeros(this.getRows(), this.getColumns(), this.getContext());
        sigmoid(this, result);
		return result;
	}
	
	public default FloatMatrix sigmoidi() {
        sigmoid(this, this);
		return this;
	}
	
	public default FloatMatrix sigmoid(FloatMatrix result) {
        sigmoid(this, result);
		return result;
	}
	
	
	
	
	
	// --------------------------------------- greater than ----------------------------------------
    
	/**
	 * expert only  
	 * 
	 * @param a
	 * @param b
	 * @param result
	 * @return
	 */
    public FloatMatrix gt(FloatMatrix a, FloatMatrix b, FloatMatrix result);
    
    /**
     * expert only  
     * 
     * @param a
     * @param scalar
     * @param result
     * @return
     */
    public FloatMatrix gt(FloatMatrix a, float scalar, FloatMatrix result);

    public default FloatMatrix gt(FloatMatrix b, FloatMatrix result) {
        gt(this, b, result);
        return result;
    }

    public default FloatMatrix gti(FloatMatrix b) {
        gt(this, b, this);
        return this;
    }

    public default FloatMatrix gt(FloatMatrix b) {
    	FloatMatrix result = FloatMatrix.zeros(b.getRows(), b.getColumns(), b.getContext());
        gt(this, b, result);
        return result;
    }
    
    public default FloatMatrix gt(float scalar, FloatMatrix result) {
        gt(this, scalar, result);
        return result;
    }

    public default FloatMatrix gti(float scalar) {
        gt(this, scalar, this);
        return this;
    }

    public default FloatMatrix gt(float scalar) {
    	FloatMatrix result = FloatMatrix.zeros(this.getRows(), this.getColumns(), this.getContext());
        gt(this, scalar, result);
        return result;
    }
    
    

	// --------------------------------------- mmul ----------------------------------------
    
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
    	FloatMatrix result = FloatMatrix.zeros(this.getRows(), b.getColumns(), this.getContext());
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
    	FloatMatrix result = FloatMatrix.zeros(this.getColumns(), b.getColumns(), this.getContext());
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
    	FloatMatrix result = FloatMatrix.zeros(this.getRows(), b.getRows(), this.getContext());
        mmulNT(this, b, result);
        return result;
    }
    
    
    
	// --------------------------------------- reduction methods ----------------------------------------
    
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
	
	
	
	
	// --------------------------------------- getter and setter methods ----------------------------------------
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
	 * @param a
	 * @param b
	 * @param rowOffset
	 * @param columnOffset
	 * @return
	 */
	public FloatMatrix getSubMatrix(FloatMatrix src, FloatMatrix dst, int rowOffset, int columnOffset);
    
	public default FloatMatrix getSubMatrix(int rowOffset, int columnOffset, int rows, int columns) {
		FloatMatrix result = FloatMatrix.zeros(rows, columns, this.getContext());
		return getSubMatrix(this, result, rowOffset, columnOffset);
	}
	
	public default FloatMatrix getSubMatrix(int rowOffset, int columnOffset) {
		FloatMatrix result = FloatMatrix.zeros(this.getRows()-rowOffset, this.getColumns()-columnOffset, this.getContext());
		return getSubMatrix(this, result, rowOffset, columnOffset);
	}
	
	public default FloatMatrix getSubMatrix(FloatMatrix dst, int rowOffset, int columnOffset) {
		return getSubMatrix(this, dst, rowOffset, columnOffset);
	}

}
