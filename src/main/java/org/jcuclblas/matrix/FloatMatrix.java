package org.jcuclblas.matrix;

import org.jcuclblas.function.Context;


public class FloatMatrix extends AMatrix {
    private static Context CONTEXT = Context.createCudaSinglePrecisionContext();
    private final Object matrix;


    public FloatMatrix(int rows, int columns, float values) {
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

    public FloatMatrix addi(FloatMatrix b, FloatMatrix result) {
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

    @Override
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
        isReleased();
        if (CONTEXT.isGPU()) {
            if (CONTEXT.isCUDA()) {
                ((CudaFloatMatrix) matrix).free();
            } else {
                ((CLFloatMatrix) matrix).free();
            }
        }
        notReleased = false;

    }

    private void isReleased() {
        if (!notReleased)
            throw new AccessViolationException(
                    "Access violation on: " + this.getClass().getName() + "@" + Integer.toHexString(hashCode()));
    }

    @Override
    public int getRows() {
        if (CONTEXT.isGPU()) {
            return ((AMatrix) matrix).getRows();
        } else return ((org.jblas.FloatMatrix) matrix).getRows();
    }

    @Override
    public int getColumns() {
        if (CONTEXT.isGPU()) {
            return ((AMatrix) matrix).getColumns();
        } else return ((org.jblas.FloatMatrix) matrix).getColumns();
    }
}
