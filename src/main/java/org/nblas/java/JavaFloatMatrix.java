package org.nblas.java;

import org.jblas.util.Random;
import org.nblas.Context;
import org.nblas.FloatMatrix;
import org.nblas.impl.FloatMatrixDefault;
import org.nblas.generic.AMatrix;

/**
 * TODO sometimes very slow implementations
 * TODO ebenfalls eine BLAS Level Klasse wo Lamda's gespeichert werden und dann mit consumer Funktionen arbeiten ähnlich wie Iterable.forEach(Consumer<? super T> action)
 * 
 * @author Nico
 *
 */
public class JavaFloatMatrix extends AMatrix implements FloatMatrixDefault {

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
		return Context.JBLASSinglePrecisionContext;
	}

    @Override
    public void release() {
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
		return toString2D();
	}
	
	@Override
	public FloatMatrix dup(FloatMatrix a, FloatMatrix result) {
		org.jblas.FloatMatrix mat = ((JavaFloatMatrix) a).matrix;
		org.jblas.FloatMatrix r = ((JavaFloatMatrix) result).matrix;
		System.arraycopy(mat.data, 0, r.data, 0, mat.data.length);
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
	
	@Override
	public FloatMatrix repmat(FloatMatrix source, FloatMatrix destination, int rowMultiplicator, int columnMultiplicator) {
		// TODO too many copies
		((JavaFloatMatrix) destination).matrix.copy(((JavaFloatMatrix) source).matrix.repmat(rowMultiplicator, columnMultiplicator));
		return destination;
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
    
	/**
	 * @see FloatMatrix#add(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
    @Override
    public FloatMatrix add(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	((JavaFloatMatrix) matrixA).matrix.addi(((JavaFloatMatrix) matrixB).matrix, ((JavaFloatMatrix) result).matrix);
    	return result;
    }
    
	/**
	 * @see FloatMatrix#add(FloatMatrix, float, FloatMatrix)
	 */
    @Override
    public FloatMatrix add(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	((JavaFloatMatrix) matrix).matrix.addi(scalar, ((JavaFloatMatrix) result).matrix);    	
    	return result;
    }   
    
	/**
	 * @see FloatMatrix#addColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
	@Override
	public FloatMatrix addColumnVector(FloatMatrix a, FloatMatrix columnVector, FloatMatrix result) {
		((JavaFloatMatrix) dup(a, result)).matrix.addiColumnVector(((JavaFloatMatrix) columnVector).matrix);
		return result;
	}
    
	/**
	 * @see FloatMatrix#addRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
	@Override
	public FloatMatrix addRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result) {
		((JavaFloatMatrix) dup(a, result)).matrix.addiRowVector(((JavaFloatMatrix) rowVector).matrix);
		return result;
	}

    // --------------------------------------- sub methods ----------------------------------------

	/**
	 * @see FloatMatrix#sub(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
    @Override
    public FloatMatrix sub(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	((JavaFloatMatrix) matrixA).matrix.subi(((JavaFloatMatrix) matrixB).matrix, ((JavaFloatMatrix) result).matrix);
    	return result;
    }
    
	/**
	 * @see FloatMatrix#sub(FloatMatrix, float, FloatMatrix)
	 */
    @Override
    public FloatMatrix sub(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	((JavaFloatMatrix) matrix).matrix.subi(scalar, ((JavaFloatMatrix) result).matrix);    	
    	return result;
    }   
    
	/**
	 * @see FloatMatrix#subColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
	@Override
	public FloatMatrix subColumnVector(FloatMatrix a, FloatMatrix columnVector, FloatMatrix result) {
		((JavaFloatMatrix) dup(a, result)).matrix.subiColumnVector(((JavaFloatMatrix) columnVector).matrix);
		return result;
	}
    
	/**
	 * @see FloatMatrix#subRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
	@Override
	public FloatMatrix subRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result) {
		((JavaFloatMatrix) dup(a, result)).matrix.subiRowVector(((JavaFloatMatrix) rowVector).matrix);
		return result;
	}

	/**
	 * @see FloatMatrix#rsub(FloatMatrix, float, FloatMatrix)
	 */
	@Override
	public FloatMatrix rsub(FloatMatrix matrix, float scalar, FloatMatrix result) {
		((JavaFloatMatrix) matrix).matrix.rsubi(scalar, ((JavaFloatMatrix) result).matrix);    	
    	return result;
	}

	/**
	 * @see FloatMatrix#rsubColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
	@Override
	public FloatMatrix rsubColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
		
		org.jblas.FloatMatrix vec = ((JavaFloatMatrix) columnVector).matrix;
		org.jblas.FloatMatrix a = ((JavaFloatMatrix) matrix).matrix;
		org.jblas.FloatMatrix dst = ((JavaFloatMatrix) result).matrix;
		
		for (int c = 0; c < matrix.getColumns(); c++)
			for (int r = 0; r < matrix.getRows(); r++)
				dst.put(r, c, vec.get(r, 0) - a.get(r, c));		
		
		return result;
	}

	/**
	 * @see FloatMatrix#rsubRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
	@Override
	public FloatMatrix rsubRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {

		org.jblas.FloatMatrix vec = ((JavaFloatMatrix) rowVector).matrix;
		org.jblas.FloatMatrix a = ((JavaFloatMatrix) matrix).matrix;
		org.jblas.FloatMatrix dst = ((JavaFloatMatrix) result).matrix;
		
		for (int c = 0; c < matrix.getColumns(); c++)
			for (int r = 0; r < matrix.getRows(); r++)
				dst.put(r, c, vec.get(0, c) - a.get(r, c));		
		
		return result;
	}

    // --------------------------------------- mul methods ----------------------------------------
    
	/**
	 * @see FloatMatrix#mul(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
    @Override
    public FloatMatrix mul(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	((JavaFloatMatrix) matrixA).matrix.muli(((JavaFloatMatrix) matrixB).matrix, ((JavaFloatMatrix) result).matrix);
    	return result;
    }
    
	/**
	 * @see FloatMatrix#mul(FloatMatrix, float, FloatMatrix)
	 */
    @Override
    public FloatMatrix mul(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	((JavaFloatMatrix) matrix).matrix.muli(scalar, ((JavaFloatMatrix) result).matrix);    	
    	return result;
    }   
    
	/**
	 * @see FloatMatrix#mulColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
	@Override
	public FloatMatrix mulColumnVector(FloatMatrix a, FloatMatrix columnVector, FloatMatrix result) {
		((JavaFloatMatrix) dup(a, result)).matrix.muliColumnVector(((JavaFloatMatrix) columnVector).matrix);
		return result;
	}
    
	/**
	 * @see FloatMatrix#mulRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
	@Override
	public FloatMatrix mulRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result) {
		((JavaFloatMatrix) dup(a, result)).matrix.muliRowVector(((JavaFloatMatrix) rowVector).matrix);
		return result;
	}
    
    // --------------------------------------- div methods ----------------------------------------
    
	/**
	 * @see FloatMatrix#div(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
    @Override
    public FloatMatrix div(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	((JavaFloatMatrix) matrixA).matrix.divi(((JavaFloatMatrix) matrixB).matrix, ((JavaFloatMatrix) result).matrix);
    	return result;
    }
    
	/**
	 * @see FloatMatrix#div(FloatMatrix, float, FloatMatrix)
	 */
    @Override
    public FloatMatrix div(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	((JavaFloatMatrix) matrix).matrix.divi(scalar, ((JavaFloatMatrix) result).matrix);    	
    	return result;
    }   
    
	/**
	 * @see FloatMatrix#divColumnVector(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
	@Override
	public FloatMatrix divColumnVector(FloatMatrix a, FloatMatrix columnVector, FloatMatrix result) {
		((JavaFloatMatrix) dup(a, result)).matrix.diviColumnVector(((JavaFloatMatrix) columnVector).matrix);
		return result;
	}
    
	/**
	 * @see FloatMatrix#divRowVector(FloatMatrix, FloatMatrix, FloatMatrix)
	 */
	@Override
	public FloatMatrix divRowVector(FloatMatrix a, FloatMatrix rowVector, FloatMatrix result) {
		((JavaFloatMatrix) dup(a, result)).matrix.diviRowVector(((JavaFloatMatrix) rowVector).matrix);
		return result;
	}
	
	@Override
	public FloatMatrix rdiv(FloatMatrix matrix, float scalar, FloatMatrix result) {
		((JavaFloatMatrix) matrix).matrix.rdivi(scalar, ((JavaFloatMatrix) result).matrix);    	
    	return result;
	}

	@Override
	public FloatMatrix rdivColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {

		org.jblas.FloatMatrix vec = ((JavaFloatMatrix) columnVector).matrix;
		org.jblas.FloatMatrix a = ((JavaFloatMatrix) matrix).matrix;
		org.jblas.FloatMatrix dst = ((JavaFloatMatrix) result).matrix;
		
		for (int c = 0; c < matrix.getColumns(); c++)
			for (int r = 0; r < matrix.getRows(); r++)
				dst.put(r, c, vec.get(r, 0) / a.get(r, c));		
		
		return result;
	}

	@Override
	public FloatMatrix rdivRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
		
		org.jblas.FloatMatrix vec = ((JavaFloatMatrix) rowVector).matrix;
		org.jblas.FloatMatrix a = ((JavaFloatMatrix) matrix).matrix;
		org.jblas.FloatMatrix dst = ((JavaFloatMatrix) result).matrix;
		
		for (int c = 0; c < matrix.getColumns(); c++)
			for (int r = 0; r < matrix.getRows(); r++)
				dst.put(r, c, vec.get(0, c) / a.get(r, c));		
		
		return result;
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
    
    /**
  	 * @see FloatMatrix#gt(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix gt(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	((JavaFloatMatrix) matrixA).matrix.gti(((JavaFloatMatrix) matrixB).matrix, ((JavaFloatMatrix) result).matrix);
		return result;
    }
    
    /**
  	 * @see FloatMatrix#gt(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix gt(FloatMatrix matrix, float scalar, FloatMatrix result) {
		((JavaFloatMatrix) matrix).matrix.gti(scalar, ((JavaFloatMatrix) result).matrix);  
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#gtColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix gtColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	return gt(matrix, columnVector.repmat(1, matrix.getColumns()), result);
    }

    /**
  	 * @see FloatMatrix#gtRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix gtRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	return gt(matrix, rowVector.repmat(matrix.getRows(), 1), result);
    }   
        
    
    // --------------------------------------- greater or equal than ----------------------------------------
    
    /**
  	 * @see FloatMatrix#ge(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ge(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	((JavaFloatMatrix) matrixA).matrix.gei(((JavaFloatMatrix) matrixB).matrix, ((JavaFloatMatrix) result).matrix);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#ge(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ge(FloatMatrix matrix, float scalar, FloatMatrix result) {
		((JavaFloatMatrix) matrix).matrix.gei(scalar, ((JavaFloatMatrix) result).matrix);  
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#geColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix geColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	return ge(matrix, columnVector.repmat(1, matrix.getColumns()), result);
    }

    /**
  	 * @see FloatMatrix#geRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix geRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	return ge(matrix, rowVector.repmat(matrix.getRows(), 1), result);
    }  
    
    // --------------------------------------- less than ----------------------------------------
    
    /**
  	 * @see FloatMatrix#lt(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix lt(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	((JavaFloatMatrix) matrixA).matrix.lti(((JavaFloatMatrix) matrixB).matrix, ((JavaFloatMatrix) result).matrix);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#lt(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix lt(FloatMatrix matrix, float scalar, FloatMatrix result) {
		((JavaFloatMatrix) matrix).matrix.lti(scalar, ((JavaFloatMatrix) result).matrix);  
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#ltColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ltColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	return lt(matrix, columnVector.repmat(1, matrix.getColumns()), result);
    }

    /**
  	 * @see FloatMatrix#ltRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ltRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	return lt(matrix, rowVector.repmat(matrix.getRows(), 1), result);
    }   
    
    
    // --------------------------------------- less or equal than ----------------------------------------
    
    /**
  	 * @see FloatMatrix#le(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix le(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	((JavaFloatMatrix) matrixA).matrix.lei(((JavaFloatMatrix) matrixB).matrix, ((JavaFloatMatrix) result).matrix);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#le(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix le(FloatMatrix matrix, float scalar, FloatMatrix result) {
		((JavaFloatMatrix) matrix).matrix.lei(scalar, ((JavaFloatMatrix) result).matrix);  
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#leColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix leColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	return le(matrix, columnVector.repmat(1, matrix.getColumns()), result);
    }

    /**
  	 * @see FloatMatrix#leRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix leRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	return le(matrix, rowVector.repmat(matrix.getRows(), 1), result);
    }   
    
    
	// --------------------------------------- equal to ----------------------------------------
    
    /**
  	 * @see FloatMatrix#eq(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix eq(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	((JavaFloatMatrix) matrixA).matrix.eqi(((JavaFloatMatrix) matrixB).matrix, ((JavaFloatMatrix) result).matrix);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#eq(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix eq(FloatMatrix matrix, float scalar, FloatMatrix result) {
		((JavaFloatMatrix) matrix).matrix.eqi(scalar, ((JavaFloatMatrix) result).matrix);  
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#eqColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix eqColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	return eq(matrix, columnVector.repmat(1, matrix.getColumns()), result);
    }

    /**
  	 * @see FloatMatrix#eqRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix eqRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	return eq(matrix, rowVector.repmat(matrix.getRows(), 1), result);
    }   
    
    
    // --------------------------------------- not equal to ----------------------------------------
    
    /**
  	 * @see FloatMatrix#ne(FloatMatrix, FloatMatrix, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ne(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	((JavaFloatMatrix) matrixA).matrix.nei(((JavaFloatMatrix) matrixB).matrix, ((JavaFloatMatrix) result).matrix);
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#ne(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix ne(FloatMatrix matrix, float scalar, FloatMatrix result) {
		((JavaFloatMatrix) matrix).matrix.nei(scalar, ((JavaFloatMatrix) result).matrix);  
    	return result;
    }
    
    /**
  	 * @see FloatMatrix#neColumnVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix neColumnVector(FloatMatrix matrix, FloatMatrix columnVector, FloatMatrix result) {
    	return ne(matrix, columnVector.repmat(1, matrix.getColumns()), result);
    }

    /**
  	 * @see FloatMatrix#neRowVector(FloatMatrix, float, FloatMatrix)
  	 */
    @Override
    public FloatMatrix neRowVector(FloatMatrix matrix, FloatMatrix rowVector, FloatMatrix result) {
    	return ne(matrix, rowVector.repmat(matrix.getRows(), 1), result);
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
	
	@Override
	public float mean(FloatMatrix a) {
		return ((JavaFloatMatrix) a).matrix.mean();
	}

	@Override
	public float prod(FloatMatrix a) {
		return ((JavaFloatMatrix) a).matrix.prod();
	}

	@Override
	public float max(FloatMatrix a) {
		return ((JavaFloatMatrix) a).matrix.max();
	}

	@Override
	public float min(FloatMatrix a) {
		return ((JavaFloatMatrix) a).matrix.min();
	}

	// --------------------------------------- getter and setter methods ----------------------------------------

	@Override
	public FloatMatrix setSubMatrix(FloatMatrix src, FloatMatrix dst, int rowOffset, int columnOffset) {
		
		org.jblas.FloatMatrix matrix = ((JavaFloatMatrix) src).matrix;
		org.jblas.FloatMatrix result = ((JavaFloatMatrix) dst).matrix;
		
		for (int c = 0; c < matrix.getColumns(); c++)
			for (int r = 0; r < matrix.getRows(); r++)
				result.put(r + rowOffset, c + columnOffset, matrix.get(r, c));
		 
		return dst;
	}

	@Override
	public FloatMatrix getSubMatrix(FloatMatrix source, FloatMatrix destination, int rowOffset, int columnOffset) {
		org.jblas.FloatMatrix dst = ((JavaFloatMatrix) destination).matrix;
		org.jblas.FloatMatrix src = ((JavaFloatMatrix) source).matrix;
		dst.copy(src.getRange(rowOffset, rowOffset + destination.getRows(), columnOffset, columnOffset + destination.getColumns()));
		return destination;
	}

	@Override
	public FloatMatrix put(FloatMatrix src, int rowIndex, int columnIndex, float value) {
		((JavaFloatMatrix) src).matrix.put(rowIndex, columnIndex, value);
		return src;
	}

	@Override
	public float get(FloatMatrix src, int rowIndex, int columnIndex) {
		return ((JavaFloatMatrix) src).matrix.get(rowIndex, columnIndex);
	}

	@Override
	public FloatMatrix getRow(FloatMatrix src, FloatMatrix row, int rowIndex) {
		
		org.jblas.FloatMatrix srcMatrix = ((JavaFloatMatrix) src).matrix;
		org.jblas.FloatMatrix rowMatrix = ((JavaFloatMatrix) row).matrix;
		srcMatrix.getRow(rowIndex, rowMatrix);
		
		return row;
	}

	@Override
	public FloatMatrix getColumn(FloatMatrix src, FloatMatrix column, int columnIndex) {
		
		org.jblas.FloatMatrix srcMatrix = ((JavaFloatMatrix) src).matrix;
		org.jblas.FloatMatrix columnMatrix = ((JavaFloatMatrix) column).matrix;
		srcMatrix.getColumn(columnIndex, columnMatrix);
		
		return column;
	}

	
	@Override
	public FloatMatrix putRow(FloatMatrix dst, FloatMatrix row, int rowIndex) {
		
		org.jblas.FloatMatrix dstMatrix = ((JavaFloatMatrix) dst).matrix;
		org.jblas.FloatMatrix rowMatrix = ((JavaFloatMatrix) row).matrix;
		dstMatrix.putRow(rowIndex, rowMatrix);
		
		return dst;
	}

	@Override
	public FloatMatrix putColumn(FloatMatrix dst, FloatMatrix column, int columnIndex) {
		
		org.jblas.FloatMatrix dstMatrix = ((JavaFloatMatrix) dst).matrix;
		org.jblas.FloatMatrix columnMatrix = ((JavaFloatMatrix) column).matrix;
		dstMatrix.putColumn(columnIndex, columnMatrix);
		
		return dst;
	}

}
