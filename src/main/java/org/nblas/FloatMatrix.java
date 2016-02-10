package org.nblas;

import org.nblas.cl.CLFloatMatrix;
import org.nblas.cuda.CudaFloatMatrix;
import org.nblas.java.JavaFloatMatrix;

/**
 * TODO Es gibt FloatMatrixwelches FloatMatrix extended
 * FloatMatrix hat die ganzen Experten Funktionen nicht.
 * 
 * @author Nico
 *
 */
public interface FloatMatrix  {
  
    
    /**
     * column-major order
     * 
     * @param values
     * @param context
     * @return
     */
    public static FloatMatrix create(float[][] values, Context context) {
    	
    	// flat representation in column-major order
    	int rows = values.length;
    	int columns = values[0].length;
    	float[] flat = new float[rows * columns];
   		for (int y = 0; y < rows; y++)
   			for (int x = 0; x < columns; x++)
    			flat[x * rows + y] = values[y][x];
    	
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
            	FloatMatrix mat = new CLFloatMatrix(rows, columns);
            	mat.setZero();
            	return mat;
            }
        } else {
            return new JavaFloatMatrix(rows, columns);
        }
    }
    
    public static FloatMatrix ones(int rows, int columns, Context context) {
    	FloatMatrix matrix = null;
    	
    	// dirty allocation 
        if (context.isGPU()) {
            if (context.isCUDA()) {
            	matrix = new CudaFloatMatrix(rows, columns);
            } else {
            	matrix = new CLFloatMatrix(rows, columns);
            }
        } else {
        	matrix = new JavaFloatMatrix(rows, columns);
        }
        
    	matrix.setOne();
    	return matrix;
    }    
    
    public static FloatMatrix rand(int rows, int columns, Context context) {
      	FloatMatrix matrix = null;
      	
      	// dirty allocation 
        if (context.isGPU()) {
            if (context.isCUDA()) {
            	matrix = new CudaFloatMatrix(rows, columns);
            } else {
            	matrix = new CLFloatMatrix(rows, columns);
            }
        } else {
        	matrix = new JavaFloatMatrix(rows, columns);
        }
        
    	matrix.randi();
    	return matrix;
    }
    
    public static FloatMatrix randn(int rows, int columns, Context context) {
      	FloatMatrix matrix = null;
      	
    	// dirty allocation 
        if (context.isGPU()) {
            if (context.isCUDA()) {
            	matrix = new CudaFloatMatrix(rows, columns);
            } else {
            	matrix = new CLFloatMatrix(rows, columns);
            }
        } else {
        	matrix = new JavaFloatMatrix(rows, columns);
        }
        
    	matrix.randni();
    	return matrix;
    }
    
    
    
    // ------------------------------------- java getter methods --------------------------------------

    public float[] toArray();
	public float[][] toArray2();

	// ------------------------------------- print methods --------------------------------------
	
    public String toString1D();
    public String toString2D();
    
    // ------------------------------------- utility methods --------------------------------------
    public Context getContext();
    public void free();
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
    public FloatMatrix dup();
    
    /**
     * copy the content of this matrix to the result matrix
     * 
     * @param result
     * @return
     */
    public FloatMatrix dup(FloatMatrix result);
    
    public FloatMatrix transpose(FloatMatrix matrix, FloatMatrix transposed);
    
    public FloatMatrix repmat(FloatMatrix source, FloatMatrix destination, int rowMultiplicator, int columnMultiplicator);
    
    public FloatMatrix repmat(int rowMultiplicator, int columnMultiplicator);
    
    
    
    // ---------------------------------- common inplace methods ----------------------------------
    
    public FloatMatrix setOne();
    public FloatMatrix setZero();
    public FloatMatrix randi();
    public FloatMatrix randni();
    
    
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- add methods ------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------  
    
    public FloatMatrix add(FloatMatrix b, FloatMatrix result);

    public FloatMatrix addi(FloatMatrix b);

    public FloatMatrix add(FloatMatrix b);
    
    
    public FloatMatrix add(float scalar, FloatMatrix result);

    public FloatMatrix addi(float scalar);

    public FloatMatrix add(float scalar);
    
    public FloatMatrix addColumnVector(FloatMatrix columnVector, FloatMatrix result);

    public FloatMatrix addiColumnVector(FloatMatrix columnVector);
    
    public FloatMatrix addColumnVector(FloatMatrix columnVector);
    
    public FloatMatrix addRowVector(FloatMatrix rowVector, FloatMatrix result);

    public FloatMatrix addiRowVector(FloatMatrix rowVector);
    
    public FloatMatrix addRowVector(FloatMatrix rowVector);
    
    
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- sub methods ------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    
    public FloatMatrix sub(FloatMatrix b, FloatMatrix result);

    public FloatMatrix subi(FloatMatrix b);

    public FloatMatrix sub(FloatMatrix b);
    
    public FloatMatrix sub(float scalar, FloatMatrix result);

    public FloatMatrix subi(float scalar);

    public FloatMatrix sub(float scalar);
    
    public FloatMatrix subColumnVector(FloatMatrix columnVector, FloatMatrix result);

    public FloatMatrix subiColumnVector(FloatMatrix columnVector);
    
    public FloatMatrix subColumnVector(FloatMatrix columnVector);
    
    public FloatMatrix subRowVector(FloatMatrix rowVector, FloatMatrix result);

    public FloatMatrix subiRowVector(FloatMatrix rowVector);
    
    public FloatMatrix subRowVector(FloatMatrix rowVector);
    
    public FloatMatrix rsub(float scalar, FloatMatrix result);

    public FloatMatrix rsubi(float scalar);
    
    public FloatMatrix rsub(float scalar);

    public FloatMatrix rsubColumnVector(FloatMatrix columnVector, FloatMatrix result);

    public FloatMatrix rsubiColumnVector(FloatMatrix columnVector);
    
    public FloatMatrix rsubColumnVector(FloatMatrix columnVector);
    
    public FloatMatrix rsubRowVector(FloatMatrix rowVector, FloatMatrix result);

    public FloatMatrix rsubiRowVector(FloatMatrix rowVector);
    
    public FloatMatrix rsubRowVector(FloatMatrix rowVector);
    
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- mul methods ------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    
    public FloatMatrix mul(FloatMatrix b, FloatMatrix result);

    public FloatMatrix muli(FloatMatrix b);

    public FloatMatrix mul(FloatMatrix b);
    
    public FloatMatrix mul(float scalar, FloatMatrix result);
    
    public FloatMatrix muli(float scalar);

    public FloatMatrix mul(float scalar);
    
    public FloatMatrix mulColumnVector(FloatMatrix columnVector, FloatMatrix result);

    public FloatMatrix muliColumnVector(FloatMatrix columnVector);
    
    public FloatMatrix mulColumnVector(FloatMatrix columnVector);
    
    public FloatMatrix mulRowVector(FloatMatrix rowVector, FloatMatrix result);

    public FloatMatrix muliRowVector(FloatMatrix rowVector);
    
    public FloatMatrix mulRowVector(FloatMatrix rowVector);
    
    
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- div methods ------------------------------------------------
	// ---------------------------------------------------------------------------------------------------------- 
    
    public FloatMatrix div(FloatMatrix b, FloatMatrix result);

    public FloatMatrix divi(FloatMatrix b);

    public FloatMatrix div(FloatMatrix b);
    
    public FloatMatrix div(float scalar, FloatMatrix result);

    public FloatMatrix divi(float scalar);

    public FloatMatrix div(float scalar);
    
    public FloatMatrix divColumnVector(FloatMatrix columnVector, FloatMatrix result);

    public FloatMatrix diviColumnVector(FloatMatrix columnVector);
    
    public FloatMatrix divColumnVector(FloatMatrix columnVector);
    
    public FloatMatrix divRowVector(FloatMatrix rowVector, FloatMatrix result);

    public FloatMatrix diviRowVector(FloatMatrix rowVector);
    
    public FloatMatrix divRowVector(FloatMatrix rowVector); 
    
    public FloatMatrix rdiv(float scalar, FloatMatrix result);

    public FloatMatrix rdivi(float scalar);
    
    public FloatMatrix rdiv(float scalar); 

    public FloatMatrix rdivColumnVector(FloatMatrix columnVector, FloatMatrix result);

    public FloatMatrix rdiviColumnVector(FloatMatrix columnVector);
    
    public FloatMatrix rdivColumnVector(FloatMatrix columnVector); 
    
    public FloatMatrix rdivRowVector(FloatMatrix rowVector, FloatMatrix result);

    public FloatMatrix rdiviRowVector(FloatMatrix rowVector);
    
    public FloatMatrix rdivRowVector(FloatMatrix rowVector);
    
    // ----------------------------------------------------------------------------------------------------------
    // ------------------------------------------- exponential methods ------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	public FloatMatrix exp(FloatMatrix result);
	
	public FloatMatrix expi();

	public FloatMatrix exp();
	
	
    // ----------------------------------------------------------------------------------------------------------
    // ------------------------------------------------ negate methods ------------------------------------------
	// ----------------------------------------------------------------------------------------------------------

	public FloatMatrix neg(FloatMatrix result);
	
	public FloatMatrix negi();

	public FloatMatrix neg();
	

    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- sigmoid methods ------------------------------------------
	// ----------------------------------------------------------------------------------------------------------

	public FloatMatrix sigmoid(FloatMatrix result);
	
	public FloatMatrix sigmoidi();
	
	public FloatMatrix sigmoid();
	
	

    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- greater than ---------------------------------------------
	// ----------------------------------------------------------------------------------------------------------

    public FloatMatrix gt(FloatMatrix b, FloatMatrix result);

    public FloatMatrix gti(FloatMatrix b);
    
    public FloatMatrix gt(FloatMatrix b);

    public FloatMatrix gt(float scalar, FloatMatrix result);

    public FloatMatrix gti(float scalar);

    public FloatMatrix gt(float scalar);

    public FloatMatrix gtRowVector(FloatMatrix rowVector, FloatMatrix result);

    public FloatMatrix gtiRowVector(FloatMatrix rowVector);

    public FloatMatrix gtRowVector(FloatMatrix rowVector);

    public FloatMatrix gtColumnVector(FloatMatrix ColumnVector, FloatMatrix result);

    public FloatMatrix gtiColumnVector(FloatMatrix ColumnVector);

    public FloatMatrix gtColumnVector(FloatMatrix ColumnVector);
    
    
    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- greater or equal than ------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    
    public FloatMatrix ge(FloatMatrix b, FloatMatrix result);

    public FloatMatrix gei(FloatMatrix b);

    public FloatMatrix ge(FloatMatrix b);

    public FloatMatrix ge(float scalar, FloatMatrix result);

    public FloatMatrix gei(float scalar);

    public FloatMatrix ge(float scalar);

    public FloatMatrix geRowVector(FloatMatrix rowVector, FloatMatrix result);

    public FloatMatrix geiRowVector(FloatMatrix rowVector);

    public FloatMatrix geRowVector(FloatMatrix rowVector);

    public FloatMatrix geColumnVector(FloatMatrix ColumnVector, FloatMatrix result);

    public FloatMatrix geiColumnVector(FloatMatrix ColumnVector);

    public FloatMatrix geColumnVector(FloatMatrix ColumnVector);
    
    
    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- less than ------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    
    public FloatMatrix lt(FloatMatrix b, FloatMatrix result);

    public FloatMatrix lti(FloatMatrix b);

    public FloatMatrix lt(FloatMatrix b);

    public FloatMatrix lt(float scalar, FloatMatrix result);

    public FloatMatrix lti(float scalar);

    public FloatMatrix lt(float scalar);

    public FloatMatrix ltRowVector(FloatMatrix rowVector, FloatMatrix result);

    public FloatMatrix ltiRowVector(FloatMatrix rowVector);

    public FloatMatrix ltRowVector(FloatMatrix rowVector);
    
    public FloatMatrix ltColumnVector(FloatMatrix ColumnVector, FloatMatrix result);

    public FloatMatrix ltiColumnVector(FloatMatrix ColumnVector);

    public FloatMatrix ltColumnVector(FloatMatrix ColumnVector);
    
    
    
    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- less or equal than ---------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    
    public FloatMatrix le(FloatMatrix b, FloatMatrix result);

    public FloatMatrix lei(FloatMatrix b);

    public FloatMatrix le(FloatMatrix b);

    public FloatMatrix le(float scalar, FloatMatrix result);

    public FloatMatrix lei(float scalar);

    public FloatMatrix le(float scalar);

    public FloatMatrix leRowVector(FloatMatrix rowVector, FloatMatrix result);

    public FloatMatrix leiRowVector(FloatMatrix rowVector);

    public FloatMatrix leRowVector(FloatMatrix rowVector);

    public FloatMatrix leColumnVector(FloatMatrix ColumnVector, FloatMatrix result);

    public FloatMatrix leiColumnVector(FloatMatrix ColumnVector);

    public FloatMatrix leColumnVector(FloatMatrix ColumnVector);
    
    
    // ----------------------------------------------------------------------------------------------------------
    // ------------------------------------------------- equal to -----------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    
    public FloatMatrix eq(FloatMatrix b, FloatMatrix result);

    public FloatMatrix eqi(FloatMatrix b);

    public FloatMatrix eq(FloatMatrix b);
    
    public FloatMatrix eq(float scalar, FloatMatrix result);

    public FloatMatrix eqi(float scalar);

    public FloatMatrix eq(float scalar);

    public FloatMatrix eqRowVector(FloatMatrix rowVector, FloatMatrix result);

    public FloatMatrix eqiRowVector(FloatMatrix rowVector);

    public FloatMatrix eqRowVector(FloatMatrix rowVector);

    public FloatMatrix eqColumnVector(FloatMatrix ColumnVector, FloatMatrix result);

    public FloatMatrix eqiColumnVector(FloatMatrix ColumnVector);

    public FloatMatrix eqColumnVector(FloatMatrix ColumnVector); 
    
    
    // ----------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- not equal to ---------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    
    public FloatMatrix ne(FloatMatrix b, FloatMatrix result);

    public FloatMatrix nei(FloatMatrix b);

    public FloatMatrix ne(FloatMatrix b);

    public FloatMatrix ne(float scalar, FloatMatrix result);

    public FloatMatrix nei(float scalar);

    public FloatMatrix ne(float scalar);

    public FloatMatrix neRowVector(FloatMatrix rowVector, FloatMatrix result);

    public FloatMatrix neiRowVector(FloatMatrix rowVector);

    public FloatMatrix neRowVector(FloatMatrix rowVector);

    public FloatMatrix neColumnVector(FloatMatrix ColumnVector, FloatMatrix result);

    public FloatMatrix neiColumnVector(FloatMatrix ColumnVector);

    public FloatMatrix neColumnVector(FloatMatrix ColumnVector);    
    
    // ----------------------------------------------------------------------------------------------------------
    // ---------------------------------------- matrix multiplication -------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    
    public FloatMatrix mmul(FloatMatrix b, FloatMatrix result);

    public FloatMatrix mmul(FloatMatrix b);
    
    public FloatMatrix mmulTN(FloatMatrix b, FloatMatrix result);

    public FloatMatrix mmulTN(FloatMatrix b);
    
    public FloatMatrix mmulNT(FloatMatrix b, FloatMatrix result);

    public FloatMatrix mmulNT(FloatMatrix b);   
    
    
    // ----------------------------------------------------------------------------------------------------------
    // ------------------------------------------- reduction methods --------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
    	
	public float sum();

	public float mean();
	
	public float prod();

	public float max();
	
	public float min();
	
	
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------- getter and setter methods ----------------------------------------
	// ----------------------------------------------------------------------------------------------------------

	public FloatMatrix setSubMatrix(FloatMatrix src, int rowOffset, int columnOffset);
	
	public FloatMatrix getSubMatrix(int rowOffset, int columnOffset, int rows, int columns);
	
	public FloatMatrix getSubMatrix(FloatMatrix dst, int rowOffset, int columnOffset);
	
	public FloatMatrix getSubMatrix(int rowOffset, int columnOffset);
	
	public FloatMatrix put(int rowIndex, int columnIndex, float value);
	
	public float get(int rowIndex, int columnIndex);
	
	public FloatMatrix getRow(FloatMatrix row, int rowIndex);
	
	public FloatMatrix getRow(int rowIndex);
	
	public FloatMatrix getColumn(FloatMatrix column, int columnIndex);
	
	public FloatMatrix getColumn(int columnIndex);
	
	public FloatMatrix putRow(FloatMatrix row, int rowIndex);
	
	public FloatMatrix putColumn(FloatMatrix column, int columnIndex);
}
