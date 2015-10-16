package org.math.joclblas;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;

import java.util.Arrays;

public class CLFloatMatrix extends ACLMatrix {


    public CLFloatMatrix(float[][] array) {
        float[] columnMajorArray = new float[array.length * array[0].length];
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[0].length; j++) {
                columnMajorArray[i + array.length * j] = array[i][j];
            }
        }
        init(array.length, array[0].length, columnMajorArray);
    }

    private CLFloatMatrix(CLFloatMatrix matrix) {

        this.rows = matrix.rows;
        this.columns = matrix.columns;
        this.length = matrix.length;
        this.clRows = matrix.clRows;
        this.clColumns = matrix.clColumns;
        this.clLength = matrix.clLength;
        this.buffer = CL.clCreateBuffer(Core.getInstance().getContext(),
                CL.CL_MEM_READ_WRITE,
                Sizeof.cl_float * clLength, null, null);
        Core.getInstance().copy(matrix.buffer, buffer, clRows, clColumns);
    }

    private CLFloatMatrix(int rows, int columns) {

        this.rows = rows;
        this.columns = columns;
        this.length = rows * columns;
        setCLDimensions(rows, columns);
        this.buffer = CL.clCreateBuffer(Core.getInstance().getContext(),
                CL.CL_MEM_READ_WRITE,
                Sizeof.cl_float * clLength, null, null);
    }

    public CLFloatMatrix(int rows, int columns, float[] columnMajorArray) {
        init(rows, columns, columnMajorArray);
    }

    private void init(int rows, int columns, float[] columnMajorArray) {
        this.length = rows * columns;
        if (this.length != columnMajorArray.length)
            throw new IllegalArgumentException("rows times columns is not equal to the array's length!");
        this.rows = rows;
        this.columns = columns;
        setCLDimensions(rows, columns);

        float[] clArray;
        if (clColumns != columns || clRows != rows) {
            clArray = createCLArray(rows, columns, columnMajorArray);
        } else {
            clArray = columnMajorArray;
        }
        Pointer pointer = Pointer.to(clArray);

        buffer = CL.clCreateBuffer(Core.getInstance().getContext(),
                CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * clLength, pointer, null);
    }

    private void setCLDimensions(int rows, int columns) {
        int[] clDimensions = Core.getInstance().getDimensions(rows, columns);
        this.clRows = clDimensions[0];
        this.clColumns = clDimensions[1];
        this.clLength = clColumns * clRows;
    }

    public static CLFloatMatrix zeros(int rows, int columns) {
        CLFloatMatrix clFloatMatrix = new CLFloatMatrix(rows, columns);
        Core.getInstance().setZero(clFloatMatrix.buffer, clFloatMatrix.clRows, clFloatMatrix.clColumns);
        return clFloatMatrix;
    }

    public static CLFloatMatrix ones(int rows, int columns) {
        CLFloatMatrix clFloatMatrix = new CLFloatMatrix(rows, columns);
        Core.getInstance().setOne(clFloatMatrix.buffer, clFloatMatrix.clRows, clFloatMatrix.clColumns, clFloatMatrix.rows, clFloatMatrix.columns);
        return clFloatMatrix;
    }

    public static CLFloatMatrix rand(int rows, int columns) {
        float[] columnMajorArray = new float[rows * columns];
        for (int i = 0; i < columnMajorArray.length; i++) {
            columnMajorArray[i] = RANDOM.nextFloat();
        }
        return new CLFloatMatrix(rows, columns, columnMajorArray);
    }

    public static CLFloatMatrix randn(int rows, int columns) {
        float[] columnMajorArray = new float[rows * columns];
        for (int i = 0; i < columnMajorArray.length; i++) {
            columnMajorArray[i] = (float) RANDOM.nextGaussian();
        }
        return new CLFloatMatrix(rows, columns, columnMajorArray);
    }

    public static void defineUnaryElementwiseFunctionX(String name, String operation) {
        Core.getInstance().defineFloatXKernel(name, operation);
        Core.getInstance().compileCustomKernels();
    }

    public static void defineBinaryElementwiseFunctionXY(String name, String operation) {
        Core.getInstance().defineFloatXYKernel(name, operation);
        Core.getInstance().compileCustomKernels();
    }

    public static void applyUnaryFunction(String name, CLFloatMatrix a, CLFloatMatrix result) {
        checkSameSize(a, result);
        Core.getInstance().applyCustomKernel(name, a.buffer, result.buffer, a.clColumns, a.clRows, a.columns, a.rows);
    }

    public static void applyBinaryFunction(String name, CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkSameSize(a, b, result);
        Core.getInstance().applyCustomKernel(name, a.buffer, b.buffer, result.buffer, a.clColumns, a.clRows, a.columns, a.rows);
    }

    private float[] createCLArray(int rows, int columns, float[] columnMajorArray) {
        float[] result = new float[clColumns * clRows];
        for (int i = 0; i < columns; i++) {
            int sourceStride = i * rows;
            int destinationStride = i * clRows;
            for (int j = 0; j < rows; j++) {
                result[destinationStride + j] = columnMajorArray[sourceStride + j];
            }
        }
        return result;
    }

    public static void mmul(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        isMmulable(a, b, result);
        Core.getInstance().sgemm(a.buffer, b.buffer, result.buffer, a.clRows, b.clColumns, a.clColumns);
    }

    public static void transpose(CLFloatMatrix a, CLFloatMatrix b) {
        isTransposeable(a, b);
        if (a.isColumnVector() || a.isRowVector()) {
            Core.getInstance().setZero(b.buffer, b.clRows, b.clColumns);
            if (a.length < b.length) {
                Core.getInstance().copy(a.buffer, b.buffer, a.clRows, a.clColumns);
            } else {
                Core.getInstance().copy(a.buffer, b.buffer, b.clRows, b.clColumns);
            }
        } else {
            Core.getInstance().transpose(a.buffer, b.buffer, a.clRows, a.clColumns);
        }
    }

    public static void add(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkSameSize(a, b, result);
        Core.getInstance().add(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns);
    }

    public static void add(CLFloatMatrix a, float b, CLFloatMatrix result) {
        checkSameSize(a, result);
        Core.getInstance().addScalar(a.buffer, b, result.buffer, a.clRows, a.clColumns, a.rows, a.columns);
    }

    public static void addColumnVector(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        Core.getInstance().addColumnVector(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns);
    }

    public static void addRowVector(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        Core.getInstance().addRowVector(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns);
    }

    public static void mul(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkSameSize(a, b, result);
        Core.getInstance().mul(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns);
    }

    public static void mul(CLFloatMatrix a, float b, CLFloatMatrix result) {
        checkSameSize(a, result);
        Core.getInstance().mulScalar(a.buffer, b, result.buffer, a.clRows, a.clColumns);
    }

    public static void mulColumnVector(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        Core.getInstance().mulColumnVector(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns);
    }

    public static void mulRowVector(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        Core.getInstance().mulRowVector(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns);
    }

    public static void sub(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkSameSize(a, b, result);
        Core.getInstance().sub(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns);
    }

    public static void sub(CLFloatMatrix a, float b, CLFloatMatrix result) {
        checkSameSize(a, result);
        Core.getInstance().subScalar(a.buffer, b, result.buffer, a.clRows, a.clColumns, a.rows, a.columns);
    }

    public static void subColumnVector(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        Core.getInstance().subColumnVector(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns);
    }

    public static void subRowVector(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        Core.getInstance().subRowVector(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns);
    }

    public static void rsub(CLFloatMatrix a, float b, CLFloatMatrix result) {
        checkSameSize(a, result);
        Core.getInstance().rsubScalar(a.buffer, b, result.buffer, a.clRows, a.clColumns, a.rows, a.columns);
    }

    public static void rsubColumnVector(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        Core.getInstance().rsubColumnVector(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns);
    }

    public static void rsubRowVector(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        Core.getInstance().rsubRowVector(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns);
    }

    public static void div(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkSameSize(a, b, result);
        Core.getInstance().div(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns, a.rows, a.columns);
    }

    public static void div(CLFloatMatrix a, float b, CLFloatMatrix result) {
        checkSameSize(a, result);
        Core.getInstance().divScalar(a.buffer, b, result.buffer, a.clRows, a.clColumns, a.rows, a.columns);
    }

    public static void divColumnVector(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        Core.getInstance().divColumnVector(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns, a.rows, a.columns);
    }

    public static void divRowVector(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        Core.getInstance().divRowVector(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns, a.clRows, a.clColumns);
    }

    public static void rdiv(CLFloatMatrix a, float b, CLFloatMatrix result) {
        checkSameSize(a, result);
        Core.getInstance().rdivScalar(a.buffer, b, result.buffer, a.clRows, a.clColumns, a.rows, a.columns);
    }

    public static void rdivColumnVector(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        Core.getInstance().rdivColumnVector(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns, a.rows, a.columns);
    }

    public static void rdivRowVector(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        Core.getInstance().rdivRowVector(a.buffer, b.buffer, result.buffer, a.clRows, a.clColumns, a.rows, a.columns);
    }

    public float sum() {
        return Core.getInstance().sum(buffer, clRows, clColumns);
    }

    public float mean() {
        return sum() / length;
    }

    public CLFloatMatrix repmat(int rowTimes, int columnTimes) {
        CLFloatMatrix result = new CLFloatMatrix(rows * rowTimes, columns * columnTimes);
        Core.getInstance().repmat(buffer, result.buffer,
                result.clRows, result.clColumns,
                result.rows, result.columns, rows, columns, clRows);
        return result;
    }

    public void setSubMatrix(CLFloatMatrix matrix, int rowOffset, int columnOffset) {
        Core.getInstance().setSubMatrix(matrix.buffer, buffer,
                matrix.clRows, matrix.clColumns,
                matrix.rows, matrix.columns, rowOffset, columnOffset, clRows);
    }

    public CLFloatMatrix getSubMatrix(int rowOffset, int columnOffset, int rows, int columns) {
        CLFloatMatrix result = new CLFloatMatrix(rows, columns);
        Core.getInstance().getSubMatrix(buffer, result.buffer,
                result.clRows, result.clColumns,
                result.rows, result.columns, rowOffset, columnOffset, clRows);
        return result;
    }

    // getter
    public float[] toArray() {
        float[] clArray = Core.getInstance().getArray(buffer, clLength);
        float[] result = new float[length];
        boolean notZero = false;
        for (int i = 0; i < clColumns; i++) {
            int sourceStride = i * clRows;
            for (int j = 0; j < clRows; j++) {
                if (i >= columns || j >= rows) {
                    if (clArray[sourceStride + j] != 0.0f) {
                        notZero = true;
                        System.out.println("not 0.0");
                        break;
                    }
                }
                if (notZero) break;
            }
        }
        for (int i = 0; i < columns; i++) {
            int sourceStride = i * clRows;
            int destinationStride = i * rows;
            for (int j = 0; j < rows; j++) {
                result[destinationStride + j] = clArray[sourceStride + j];
            }
        }
        return result;
    }


    @Override
    public String toString() {
        return Arrays.toString(toArray());
    }

}
