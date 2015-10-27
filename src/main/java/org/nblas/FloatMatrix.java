package org.nblas;

import org.nblas.cl.CLFloatMatrix;
import org.nblas.cuda.CudaFloatMatrix;
import org.nblas.exception.AccessViolationException;
import org.nblas.function.Context;
import org.nblas.generic.AMatrix;
import org.nblas.generic.ANativeMatrix;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class FloatMatrix {
//    private static Context CONTEXT = Context.createCudaSinglePrecisionContext();
    private static Context CONTEXT = Context.createOpenCLSinglePrecisionContext();
    private final Object matrix;


    public FloatMatrix(int rows, int columns, float... values) {
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                this.matrix = new CudaFloatMatrix(rows, columns, values);
            } else {
                this.matrix = new CLFloatMatrix(rows, columns, values);
            }
        } else {
            this.matrix = new org.jblas.FloatMatrix(rows, columns, values);
        }
    }

    public FloatMatrix(float[][] values) {
        if (CONTEXT.isGPU()) {
        	int rows = values.length;
        	int columns = values[0].length;
        	float[] flat = new float[rows * columns];
        	for (int c = 0; c < columns; c++)
        		for (int r = 0; r < rows; r++)
        			flat[r + c * rows] = values[r][c];
			
            if (CONTEXT.isCUDA()) {
                this.matrix = new CudaFloatMatrix(rows, columns, flat);
            } else {
                this.matrix = new CLFloatMatrix(rows, columns, flat);
            }
        } else {
            this.matrix = new org.jblas.FloatMatrix(values);
        }
    }
    
    private FloatMatrix(Object matrix) {
        this.matrix = matrix;
    }

    public static FloatMatrix zeros(int rows, int columns) {
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                return new FloatMatrix(CudaFloatMatrix.zeros(rows, columns));
            } else {
                return new FloatMatrix(CLFloatMatrix.zeros(rows, columns));
            }
        } else {
            return new FloatMatrix(org.jblas.FloatMatrix.zeros(rows, columns));
        }
    }

    public static FloatMatrix ones(int rows, int columns) {
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                return new FloatMatrix(CudaFloatMatrix.ones(rows, columns));
            } else {
                return new FloatMatrix(CLFloatMatrix.ones(rows, columns));
            }
        } else {
            return new FloatMatrix(org.jblas.FloatMatrix.ones(rows, columns));
        }
    }

    public static void add(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
        a.isReleased();
        b.isReleased();
        result.isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                CudaFloatMatrix.add((CudaFloatMatrix) a.matrix, (CudaFloatMatrix) b.matrix, (CudaFloatMatrix) result.matrix);
            } else {
                CLFloatMatrix.add((CLFloatMatrix) a.matrix, (CLFloatMatrix) b.matrix, (CLFloatMatrix) result.matrix);
            }
        } else {
            ((org.jblas.FloatMatrix) a.matrix).addi((org.jblas.FloatMatrix) b.matrix, (org.jblas.FloatMatrix) result.matrix);
        }
    }
    
    public static void add(FloatMatrix a, float scalar, FloatMatrix result) {
        a.isReleased();
        result.isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                CudaFloatMatrix.add((CudaFloatMatrix) a.matrix, scalar, (CudaFloatMatrix) result.matrix);
            } else {
                CLFloatMatrix.add((CLFloatMatrix) a.matrix, scalar, (CLFloatMatrix) result.matrix);
            }
        } else {
            ((org.jblas.FloatMatrix) a.matrix).addi(scalar, (org.jblas.FloatMatrix) result.matrix);
        }
    }

    public FloatMatrix add(FloatMatrix b, FloatMatrix result) {
        FloatMatrix.add(this, b, result);
        return result;
    }

    public FloatMatrix addi(FloatMatrix b) {
        FloatMatrix.add(this, b, this);
        return this;
    }

    public FloatMatrix add(FloatMatrix b) {
        FloatMatrix result = FloatMatrix.zeros(b.getRows(), b.getColumns());
        FloatMatrix.add(this, b, result);
        return result;
    }
    
    public FloatMatrix add(float scalar, FloatMatrix result) {
        FloatMatrix.add(this, scalar, result);
        return result;
    }

    public FloatMatrix addi(float scalar) {
        FloatMatrix.add(this, scalar, this);
        return this;
    }

    public FloatMatrix add(float scalar) {
        FloatMatrix result = FloatMatrix.zeros(this.getRows(), this.getColumns());
        FloatMatrix.add(this, scalar, result);
        return result;
    }

    public static void sub(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
        a.isReleased();
        b.isReleased();
        result.isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                CudaFloatMatrix.sub((CudaFloatMatrix) a.matrix, (CudaFloatMatrix) b.matrix, (CudaFloatMatrix) result.matrix);
            } else {
                CLFloatMatrix.sub((CLFloatMatrix) a.matrix, (CLFloatMatrix) b.matrix, (CLFloatMatrix) result.matrix);
            }
        } else {
            ((org.jblas.FloatMatrix) a.matrix).subi((org.jblas.FloatMatrix) b.matrix, (org.jblas.FloatMatrix) result.matrix);
        }
    }
    
    public static void sub(FloatMatrix a, float scalar, FloatMatrix result) {
        a.isReleased();
        result.isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                CudaFloatMatrix.sub((CudaFloatMatrix) a.matrix, scalar, (CudaFloatMatrix) result.matrix);
            } else {
                CLFloatMatrix.sub((CLFloatMatrix) a.matrix, scalar, (CLFloatMatrix) result.matrix);
            }
        } else {
            ((org.jblas.FloatMatrix) a.matrix).subi(scalar, (org.jblas.FloatMatrix) result.matrix);
        }
    }

    public FloatMatrix sub(FloatMatrix b, FloatMatrix result) {
        FloatMatrix.sub(this, b, result);
        return result;
    }

    public FloatMatrix subi(FloatMatrix b) {
        FloatMatrix.sub(this, b, this);
        return this;
    }

    public FloatMatrix sub(FloatMatrix b) {
        FloatMatrix result = FloatMatrix.zeros(b.getRows(), b.getColumns());
        FloatMatrix.sub(this, b, result);
        return result;
    }
    
    public FloatMatrix sub(float scalar, FloatMatrix result) {
        FloatMatrix.sub(this, scalar, result);
        return result;
    }

    public FloatMatrix subi(float scalar) {
        FloatMatrix.sub(this, scalar, this);
        return this;
    }

    public FloatMatrix sub(float scalar) {
        FloatMatrix result = FloatMatrix.zeros(this.getRows(), this.getColumns());
        FloatMatrix.sub(this, scalar, result);
        return result;
    }
    
    public static void mul(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
        a.isReleased();
        b.isReleased();
        result.isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                CudaFloatMatrix.mul((CudaFloatMatrix) a.matrix, (CudaFloatMatrix) b.matrix, (CudaFloatMatrix) result.matrix);
            } else {
                CLFloatMatrix.mul((CLFloatMatrix) a.matrix, (CLFloatMatrix) b.matrix, (CLFloatMatrix) result.matrix);
            }
        } else {
            ((org.jblas.FloatMatrix) a.matrix).muli((org.jblas.FloatMatrix) b.matrix, (org.jblas.FloatMatrix) result.matrix);
        }
    }
    
    public static void mul(FloatMatrix a, float scalar, FloatMatrix result) {
        a.isReleased();
        result.isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                CudaFloatMatrix.mul((CudaFloatMatrix) a.matrix, scalar, (CudaFloatMatrix) result.matrix);
            } else {
                CLFloatMatrix.mul((CLFloatMatrix) a.matrix, scalar, (CLFloatMatrix) result.matrix);
            }
        } else {
            ((org.jblas.FloatMatrix) a.matrix).muli(scalar, (org.jblas.FloatMatrix) result.matrix);
        }
    }

    public FloatMatrix mul(FloatMatrix b, FloatMatrix result) {
        FloatMatrix.mul(this, b, result);
        return result;
    }

    public FloatMatrix muli(FloatMatrix b) {
        FloatMatrix.mul(this, b, this);
        return this;
    }

    public FloatMatrix mul(FloatMatrix b) {
        FloatMatrix result = FloatMatrix.zeros(b.getRows(), b.getColumns());
        FloatMatrix.mul(this, b, result);
        return result;
    }
    
    public FloatMatrix mul(float scalar, FloatMatrix result) {
        FloatMatrix.mul(this, scalar, result);
        return result;
    }

    public FloatMatrix muli(float scalar) {
        FloatMatrix.mul(this, scalar, this);
        return this;
    }

    public FloatMatrix mul(float scalar) {
        FloatMatrix result = FloatMatrix.zeros(this.getRows(), this.getColumns());
        FloatMatrix.mul(this, scalar, result);
        return result;
    }
    
    public static void div(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
        a.isReleased();
        b.isReleased();
        result.isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                CudaFloatMatrix.div((CudaFloatMatrix) a.matrix, (CudaFloatMatrix) b.matrix, (CudaFloatMatrix) result.matrix);
            } else {
                CLFloatMatrix.div((CLFloatMatrix) a.matrix, (CLFloatMatrix) b.matrix, (CLFloatMatrix) result.matrix);
            }
        } else {
            ((org.jblas.FloatMatrix) a.matrix).divi((org.jblas.FloatMatrix) b.matrix, (org.jblas.FloatMatrix) result.matrix);
        }
    }
    
    public static void div(FloatMatrix a, float scalar, FloatMatrix result) {
        a.isReleased();
        result.isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                CudaFloatMatrix.div((CudaFloatMatrix) a.matrix, scalar, (CudaFloatMatrix) result.matrix);
            } else {
                CLFloatMatrix.div((CLFloatMatrix) a.matrix, scalar, (CLFloatMatrix) result.matrix);
            }
        } else {
            ((org.jblas.FloatMatrix) a.matrix).divi(scalar, (org.jblas.FloatMatrix) result.matrix);
        }
    }

    public FloatMatrix div(FloatMatrix b, FloatMatrix result) {
        FloatMatrix.div(this, b, result);
        return result;
    }

    public FloatMatrix divi(FloatMatrix b) {
        FloatMatrix.div(this, b, this);
        return this;
    }

    public FloatMatrix div(FloatMatrix b) {
        FloatMatrix result = FloatMatrix.zeros(b.getRows(), b.getColumns());
        FloatMatrix.div(this, b, result);
        return result;
    }
    
    public FloatMatrix div(float scalar, FloatMatrix result) {
        FloatMatrix.div(this, scalar, result);
        return result;
    }

    public FloatMatrix divi(float scalar) {
        FloatMatrix.div(this, scalar, this);
        return this;
    }

    public FloatMatrix div(float scalar) {
        FloatMatrix result = FloatMatrix.zeros(this.getRows(), this.getColumns());
        FloatMatrix.div(this, scalar, result);
        return result;
    }
    
    public static void mmul(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
        a.isReleased();
        b.isReleased();
        result.isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                CudaFloatMatrix.mmul((CudaFloatMatrix) a.matrix, (CudaFloatMatrix) b.matrix, (CudaFloatMatrix) result.matrix);
            } else {
                CLFloatMatrix.mmul((CLFloatMatrix) a.matrix, (CLFloatMatrix) b.matrix, (CLFloatMatrix) result.matrix);
            }
        } else {
            ((org.jblas.FloatMatrix) a.matrix).mmuli((org.jblas.FloatMatrix) b.matrix, (org.jblas.FloatMatrix) result.matrix);
        }
    }
    
    public FloatMatrix mmul(FloatMatrix b, FloatMatrix result) {
        FloatMatrix.mmul(this, b, result);
        return result;
    }

    public FloatMatrix mmul(FloatMatrix b) {
        FloatMatrix result = FloatMatrix.zeros(this.getRows(), b.getColumns());
        FloatMatrix.mmul(this, b, result);
        return result;
    }
    
    public static void mmulTN(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
        a.isReleased();
        b.isReleased();
        result.isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                CudaFloatMatrix.mmulTransposeA((CudaFloatMatrix) a.matrix, (CudaFloatMatrix) b.matrix, (CudaFloatMatrix) result.matrix);
            } else {
                CLFloatMatrix.mmulTransposeA((CLFloatMatrix) a.matrix, (CLFloatMatrix) b.matrix, (CLFloatMatrix) result.matrix);
            }
        } else {
            ((org.jblas.FloatMatrix) a.matrix).transpose().mmuli((org.jblas.FloatMatrix) b.matrix, (org.jblas.FloatMatrix) result.matrix);
        }
    }
    
    public FloatMatrix mmulTN(FloatMatrix b, FloatMatrix result) {
        FloatMatrix.mmulTN(this, b, result);
        return result;
    }

    public FloatMatrix mmulTN(FloatMatrix b) {
        FloatMatrix result = FloatMatrix.zeros(this.getColumns(), b.getColumns());
        FloatMatrix.mmulTN(this, b, result);
        return result;
    }
    
    public static void mmulNT(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
        a.isReleased();
        b.isReleased();
        result.isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                CudaFloatMatrix.mmulTransposeB((CudaFloatMatrix) a.matrix, (CudaFloatMatrix) b.matrix, (CudaFloatMatrix) result.matrix);
            } else {
                CLFloatMatrix.mmulTransposeB((CLFloatMatrix) a.matrix, (CLFloatMatrix) b.matrix, (CLFloatMatrix) result.matrix);
            }
        } else {
            ((org.jblas.FloatMatrix) a.matrix).mmuli(((org.jblas.FloatMatrix) b.matrix).transpose(), (org.jblas.FloatMatrix) result.matrix);
        }
    }
    
    public FloatMatrix mmulNT(FloatMatrix b, FloatMatrix result) {
        FloatMatrix.mmulNT(this, b, result);
        return result;
    }

    public FloatMatrix mmulNT(FloatMatrix b) {
        FloatMatrix result = FloatMatrix.zeros(this.getRows(), b.getRows());
        FloatMatrix.mmulNT(this, b, result);
        return result;
    }
    
    public static void gt(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
        a.isReleased();
        b.isReleased();
        result.isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
            	throw new NotImplementedException();
            } else {
                CLFloatMatrix.gt((CLFloatMatrix) a.matrix, (CLFloatMatrix) b.matrix, (CLFloatMatrix) result.matrix);
            }
        } else {
            ((org.jblas.FloatMatrix) a.matrix).gti((org.jblas.FloatMatrix) b.matrix, (org.jblas.FloatMatrix) result.matrix);
        }
    }
    
    public static void gt(FloatMatrix a, float scalar, FloatMatrix result) {
        a.isReleased();
        result.isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
            	throw new NotImplementedException();
            } else {
                CLFloatMatrix.gt((CLFloatMatrix) a.matrix, scalar, (CLFloatMatrix) result.matrix);
            }
        } else {
            ((org.jblas.FloatMatrix) a.matrix).gti(scalar, (org.jblas.FloatMatrix) result.matrix);
        }
    }

    public FloatMatrix gt(FloatMatrix b, FloatMatrix result) {
        FloatMatrix.mul(this, b, result);
        return result;
    }

    public FloatMatrix gti(FloatMatrix b) {
        FloatMatrix.mul(this, b, this);
        return this;
    }

    public FloatMatrix gt(FloatMatrix b) {
        FloatMatrix result = FloatMatrix.zeros(b.getRows(), b.getColumns());
        FloatMatrix.mul(this, b, result);
        return result;
    }
    
    public FloatMatrix gt(float scalar, FloatMatrix result) {
        FloatMatrix.mul(this, scalar, result);
        return result;
    }

    public FloatMatrix gti(float scalar) {
        FloatMatrix.mul(this, scalar, this);
        return this;
    }

    public FloatMatrix gt(float scalar) {
        FloatMatrix result = FloatMatrix.zeros(this.getRows(), this.getColumns());
        FloatMatrix.mul(this, scalar, result);
        return result;
    }
    
    public void getColumnWiseOn(float[] values) {
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                ((CudaFloatMatrix) matrix).getColumnWiseOn(values);
            } else {
                ((CLFloatMatrix) matrix).getColumnWiseOn(values);
            }
        } else {
            float[] data = ((org.jblas.FloatMatrix) matrix).data;
            System.arraycopy(data, 0, values, 0, data.length);
        }
    }

    public void free() {
        if (CONTEXT.isGPU()) {
        	((ANativeMatrix) matrix).isReleased();
            if (CONTEXT.isCUDA()) {
                ((CudaFloatMatrix) matrix).free();
            } else {
                ((CLFloatMatrix) matrix).free();
            }
        }
    }

    private void isReleased() {
    	 if (CONTEXT.isGPU() && ((ANativeMatrix) matrix).isReleased())
	            throw new AccessViolationException(
	                    "Access violation on: " + this.getClass().getName() + "@" + Integer.toHexString(hashCode()));
    }

    public int getRows() {
        if (CONTEXT.isGPU()) {
            return ((AMatrix) matrix).getRows();
        } else return ((org.jblas.FloatMatrix) matrix).getRows();
    }

    public int getColumns() {
        if (CONTEXT.isGPU()) {
            return ((AMatrix) matrix).getColumns();
        } else return ((org.jblas.FloatMatrix) matrix).getColumns();
    }

	public static FloatMatrix setSubMatrix(FloatMatrix a, FloatMatrix b, int rowOffset, int columnOffset) {
		
		a.isReleased();
        b.isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
            	((CudaFloatMatrix) a.matrix).setSubMatrix((CudaFloatMatrix) b.matrix, rowOffset, columnOffset);
            } else {
                ((CLFloatMatrix) a.matrix).setSubMatrix((CLFloatMatrix) b.matrix, rowOffset, columnOffset);
            }
        } else {
            throw new NotImplementedException();
        }
        
		return a;
	}
	
	public FloatMatrix setSubMatrix(FloatMatrix b, int rowOffset, int columnOffset) {
		FloatMatrix.setSubMatrix(this, b, rowOffset, columnOffset);
		return this;
	}
	
	public static FloatMatrix getSubMatrix(FloatMatrix a, FloatMatrix b, int rowOffset, int columnOffset) {
		
		a.isReleased();
        b.isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
            	((CudaFloatMatrix) a.matrix).getSubMatrix((CudaFloatMatrix) b.matrix, rowOffset, columnOffset);
            } else {
                ((CLFloatMatrix) a.matrix).getSubMatrix((CLFloatMatrix) b.matrix, rowOffset, columnOffset);
            }
        } else {
            throw new NotImplementedException();
        }
        
		return b;
	}
    
	public FloatMatrix getSubMatrix(int rowOffset, int columnOffset, int rows, int columns) {
		FloatMatrix result = FloatMatrix.zeros(rows, columns);
		return FloatMatrix.getSubMatrix(this, result, rowOffset, columnOffset);
	}
	
	public FloatMatrix getSubMatrix(int rowOffset, int columnOffset) {
		FloatMatrix result = FloatMatrix.zeros(this.getRows()-rowOffset, this.getColumns()-columnOffset);
		return FloatMatrix.getSubMatrix(this, result, rowOffset, columnOffset);
	}
	
	public FloatMatrix getSubMatrix(FloatMatrix b, int rowOffset, int columnOffset) {
		return FloatMatrix.getSubMatrix(this, b, rowOffset, columnOffset);
	}

	public static float sum(FloatMatrix a) {
		
		a.isReleased();
		float result = 0;
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
            	result = CudaFloatMatrix.sum((CudaFloatMatrix) a.matrix);
            } else {
            	result = CLFloatMatrix.sum((CLFloatMatrix) a.matrix);
            }
        } else {
        	result = a.sum();
        }
        
		return result;
	}
	
	public float sum() {
		return sum(this);
	}
}
