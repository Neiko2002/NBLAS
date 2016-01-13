package org.nblas.java;

import org.jblas.util.Random;
import org.nblas.Context;
import org.nblas.FloatMatrix;
import org.nblas.generic.AMatrix;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * 
 * @author Nico
 *
 */
public class JavaFloatMatrix extends AMatrix implements FloatMatrix {

	protected org.jblas.FloatMatrix matrix;
	
	public JavaFloatMatrix(int rows, int columns) {
		super(rows, columns);
		this.matrix = new org.jblas.FloatMatrix(rows, columns);
	}

	public JavaFloatMatrix(int rows, int columns, float[] values) {
		super(rows, columns);
		this.matrix = new org.jblas.FloatMatrix(rows, columns, values);
	}
	
	public JavaFloatMatrix(org.jblas.FloatMatrix matrix) {
		super(matrix.getRows(), matrix.getColumns());
		this.matrix = matrix;
	}
	
    // ---------------------------------- utility methods -------------------------------------

	@Override
	public Context getContext() {
		return Context.createJBLASSinglePrecisionContext();
	}

    @Override
    public void free() {
        matrix = null;
        released = true;
    }    
    
	@Override
	public int getRows() {
		return matrix.getRows();
	}

	@Override
	public int getColumns() {
		return matrix.getColumns();
	}
	
	@Override
	public FloatMatrix readRowMajor(float[] values) {
		 float[] data = ((org.jblas.FloatMatrix) matrix).data;
         System.arraycopy(data, 0, values, 0, data.length);
         return this;
	}

	@Override
	public String toString() {
		return toString1D();
	}
	
	@Override
	public FloatMatrix dup(FloatMatrix a, FloatMatrix result) {
		org.jblas.FloatMatrix mat = ((JavaFloatMatrix) a).matrix;
		org.jblas.FloatMatrix r = ((JavaFloatMatrix) result).matrix;
    	for (int i = 0; i < mat.data.length; i++)
			r.data[i] = mat.data[i];
		return result;
	}
	
	@Override
	public FloatMatrix transpose(FloatMatrix matrix, FloatMatrix transposed) {
		org.jblas.FloatMatrix mat = ((JavaFloatMatrix) matrix).matrix;
		org.jblas.FloatMatrix r = ((JavaFloatMatrix) transposed).matrix;
		
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                r.put(j, i, mat.get(i, j));
            }
        }

		return transposed;
	}
	
	// ---------------------------------- inplace methods -------------------------------------

	@Override
	public FloatMatrix setOne() {
		matrix.fill(1);
		return this;
	}

	@Override
	public FloatMatrix setZero() {
		matrix.fill(0);
		return this;
	}
	
	@Override
	public FloatMatrix randi() {
        for (int i = 0; i < matrix.data.length; i++)
        	matrix.data[i] = (float) Random.nextFloat();
		return this;
	}
	
	@Override
	public FloatMatrix randni() {
        for (int i = 0; i < matrix.data.length; i++)
        	matrix.data[i] = (float) Random.nextGaussian();
		return this;
	}
	
    // --------------------------------------- add methods ----------------------------------------
    
    @Override
    public FloatMatrix add(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	((JavaFloatMatrix) matrixA).matrix.addi(((JavaFloatMatrix) matrixB).matrix, ((JavaFloatMatrix) result).matrix);
    	return result;
    }

    @Override
    public FloatMatrix add(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	((JavaFloatMatrix) matrix).matrix.addi(scalar, ((JavaFloatMatrix) result).matrix);    	
    	return result;
    }   
    
	@Override
	public FloatMatrix addColumnVector(FloatMatrix a, FloatMatrix columnVector, FloatMatrix result) {
		throw new NotImplementedException();
	}

	@Override
	public FloatMatrix addRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result) {
		throw new NotImplementedException();
	}

    // --------------------------------------- sub methods ----------------------------------------
    
    @Override
    public FloatMatrix sub(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	((JavaFloatMatrix) matrixA).matrix.subi(((JavaFloatMatrix) matrixB).matrix, ((JavaFloatMatrix) result).matrix);
    	return result;
    }

    @Override
    public FloatMatrix sub(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	((JavaFloatMatrix) matrix).matrix.subi(scalar, ((JavaFloatMatrix) result).matrix);    	
    	return result;
    }
	
	@Override
	public FloatMatrix subColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
		throw new NotImplementedException();
	}

	@Override
	public FloatMatrix subRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
		throw new NotImplementedException();
	}

	@Override
	public FloatMatrix rsub(FloatMatrix matrix, float scalar, FloatMatrix result) {
		throw new NotImplementedException();
	}

	@Override
	public FloatMatrix rsubColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
		throw new NotImplementedException();
	}

	@Override
	public FloatMatrix rsubRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
		throw new NotImplementedException();
	}

    // --------------------------------------- mul methods ----------------------------------------
    
    @Override
    public FloatMatrix mul(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	((JavaFloatMatrix) matrixA).matrix.muli(((JavaFloatMatrix) matrixB).matrix, ((JavaFloatMatrix) result).matrix);
    	return result;
    }

    @Override
    public FloatMatrix mul(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	((JavaFloatMatrix) matrix).matrix.muli(scalar, ((JavaFloatMatrix) result).matrix);    	
    	return result;
    }

	@Override
	public FloatMatrix mulColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
		throw new NotImplementedException();
	}

	@Override
	public FloatMatrix mulRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
		throw new NotImplementedException();
	}
    
    // --------------------------------------- div methods ----------------------------------------
    
    @Override
    public FloatMatrix div(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	((JavaFloatMatrix) matrixA).matrix.divi(((JavaFloatMatrix) matrixB).matrix, ((JavaFloatMatrix) result).matrix);
    	return result;
    }

    @Override
    public FloatMatrix div(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	((JavaFloatMatrix) matrix).matrix.divi(scalar, ((JavaFloatMatrix) result).matrix);    	
    	return result;
    }

	@Override
	public FloatMatrix divColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
		throw new NotImplementedException();
	}

	@Override
	public FloatMatrix divRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
		throw new NotImplementedException();
	}
	
	@Override
	public FloatMatrix rdiv(FloatMatrix matrix, float scalar, FloatMatrix result) {
		throw new NotImplementedException();
	}

	@Override
	public FloatMatrix rdivColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
		throw new NotImplementedException();
	}

	@Override
	public FloatMatrix rdivRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
		throw new NotImplementedException();
	}
    
    // --------------------------------------- mathematical functions ----------------------------------------
    
	@Override
	public FloatMatrix exp(FloatMatrix a, FloatMatrix result) {
		org.jblas.FloatMatrix x = ((JavaFloatMatrix) a).matrix;
		org.jblas.FloatMatrix r = ((JavaFloatMatrix) result).matrix;
		for (int i = 0; i < x.data.length; i++)
			r.data[i] = (float) Math.exp(x.data[i]);
		return result;
	}

	@Override
	public FloatMatrix neg(FloatMatrix a, FloatMatrix result) {
		org.jblas.FloatMatrix x = ((JavaFloatMatrix) a).matrix;
		org.jblas.FloatMatrix r = ((JavaFloatMatrix) result).matrix;
		for (int i = 0; i < x.data.length; i++)
			r.data[i] = -x.data[i];
		return result;
	}

	@Override
	public FloatMatrix sigmoid(FloatMatrix a, FloatMatrix result) {
		org.jblas.FloatMatrix x = ((JavaFloatMatrix) a).matrix;
		org.jblas.FloatMatrix r = ((JavaFloatMatrix) result).matrix;
		for (int i = 0; i < x.data.length; i++)
			r.data[i] = (float) (1. / ( 1. + Math.exp(-x.data[i]) ));
		return result;
	}
	
	
	
	// --------------------------------------- greater than ----------------------------------------

	@Override
	public FloatMatrix gt(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
		((JavaFloatMatrix) a).matrix.divi(((JavaFloatMatrix) b).matrix, ((JavaFloatMatrix) result).matrix);
		return result;
	}

	@Override
	public FloatMatrix gt(FloatMatrix a, float scalar, FloatMatrix result) {
		((JavaFloatMatrix) a).matrix.divi(scalar, ((JavaFloatMatrix) result).matrix);  
		return result;
	}

	
	// --------------------------------------- mmul----------------------------------------
	
	@Override
	public FloatMatrix mmul(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
		org.jblas.FloatMatrix matrixA = ((JavaFloatMatrix) a).matrix;
		org.jblas.FloatMatrix matrixB = ((JavaFloatMatrix) b).matrix;
		org.jblas.FloatMatrix matrixR = ((JavaFloatMatrix) result).matrix;
		matrixA.mmuli(matrixB, matrixR);
		return result;
	}

	@Override
	public FloatMatrix mmulTN(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
		org.jblas.FloatMatrix matrixA = ((JavaFloatMatrix) a).matrix;
		org.jblas.FloatMatrix matrixB = ((JavaFloatMatrix) b).matrix;
		org.jblas.FloatMatrix matrixR = ((JavaFloatMatrix) result).matrix;
		matrixA.transpose().mmuli(matrixB, matrixR);
		return result;
	}

	@Override
	public FloatMatrix mmulNT(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
		org.jblas.FloatMatrix matrixA = ((JavaFloatMatrix) a).matrix;
		org.jblas.FloatMatrix matrixB = ((JavaFloatMatrix) b).matrix;
		org.jblas.FloatMatrix matrixR = ((JavaFloatMatrix) result).matrix;
		matrixA.mmuli(matrixB.transpose(), matrixR);
		return result;
	}

	
	
	// --------------------------------------- reduction methods ----------------------------------------
	
	@Override
	public float sum(FloatMatrix a) {
		return ((JavaFloatMatrix) a).matrix.sum();
	}
	
	

	// --------------------------------------- getter and setter methods ----------------------------------------

	
	@Override
	public FloatMatrix setSubMatrix(FloatMatrix src, FloatMatrix dst, int rowOffset,	int columnOffset) {
		throw new NotImplementedException();
	}

	@Override
	public FloatMatrix getSubMatrix(FloatMatrix src, FloatMatrix dst, int rowOffset,	int columnOffset) {
		throw new NotImplementedException();
	}

}
