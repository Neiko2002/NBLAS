package org.nblas.matrix;


import org.jblas.util.Random;
import org.jocl.cl_mem;
import org.nblas.function.ArgumentType;
import org.nblas.function.functionobjects.common.Arg;
import org.nblas.function.functionobjects.common.Value;
import org.nblas.function.functionobjects.generic.AFunctionObject;
import org.nblas.function.functionobjects.predefined.binary.Add;
import org.nblas.function.functionobjects.predefined.binary.Comparator;
import org.nblas.function.functionobjects.predefined.binary.Div;
import org.nblas.function.functionobjects.predefined.binary.Mul;
import org.nblas.function.functionobjects.predefined.binary.Sub;

import java.util.Optional;

class CLFloatMatrix extends AMatrix {


    private static final CLCore CORE = CLCore.getCore();

    private static final String ADD_MATRIX;
    private static final String ADD_SCALAR;
    private static final String ADD_C_VECTOR;
    private static final String ADD_R_VECTOR;

    private static final String MUL_MATRIX;
    private static final String MUL_SCALAR;
    private static final String MUL_C_VECTOR;
    private static final String MUL_R_VECTOR;

    private static final String SUB_MATRIX;
    private static final String SUB_SCALAR;
    private static final String SUB_C_VECTOR;
    private static final String SUB_R_VECTOR;

    private static final String RSUB_SCALAR;
    private static final String RSUB_C_VECTOR;
    private static final String RSUB_R_VECTOR;

    private static final String DIV_MATRIX;
    private static final String DIV_SCALAR;
    private static final String DIV_C_VECTOR;
    private static final String DIV_R_VECTOR;

    private static final String RDIV_SCALAR;
    private static final String RDIV_C_VECTOR;
    private static final String RDIV_R_VECTOR;

    private static final String SET_ONE;
    private static final String COPY_MATRIX;
    
    // greater then scalar
    private static final String GT_MATRIX;
    private static final String GT_SCALAR;
    private static final String GT_C_VECTOR;
    private static final String GT_R_VECTOR;

    static {
        CLFloatFunctionBuilder builder = new CLFloatFunctionBuilder();
        // add Functions
        AFunctionObject add = new Add(new Arg(0), new Arg(1));

        ADD_MATRIX = buildPredefinedFunction(builder, add, ArgumentType.MATRIX);
        ADD_SCALAR = buildPredefinedFunction(builder, add, ArgumentType.SCALAR);
        ADD_R_VECTOR = buildPredefinedFunction(builder, add, ArgumentType.ROW_VECTOR);
        ADD_C_VECTOR = buildPredefinedFunction(builder, add, ArgumentType.COLUMN_VECTOR);


        AFunctionObject mul = new Mul(new Arg(0), new Arg(1));

        MUL_MATRIX = buildPredefinedFunction(builder, mul, ArgumentType.MATRIX);
        MUL_SCALAR = buildPredefinedFunction(builder, mul, ArgumentType.SCALAR);
        MUL_R_VECTOR = buildPredefinedFunction(builder, mul, ArgumentType.ROW_VECTOR);
        MUL_C_VECTOR = buildPredefinedFunction(builder, mul, ArgumentType.COLUMN_VECTOR);


        AFunctionObject sub = new Sub(new Arg(0), new Arg(1));

        SUB_MATRIX = buildPredefinedFunction(builder, sub, ArgumentType.MATRIX);
        SUB_SCALAR = buildPredefinedFunction(builder, sub, ArgumentType.SCALAR);
        SUB_R_VECTOR = buildPredefinedFunction(builder, sub, ArgumentType.ROW_VECTOR);
        SUB_C_VECTOR = buildPredefinedFunction(builder, sub, ArgumentType.COLUMN_VECTOR);


        AFunctionObject rsub = new Sub(new Arg(1), new Arg(0));

        RSUB_SCALAR = buildPredefinedFunction(builder, rsub, ArgumentType.SCALAR);
        RSUB_R_VECTOR = buildPredefinedFunction(builder, rsub, ArgumentType.ROW_VECTOR);
        RSUB_C_VECTOR = buildPredefinedFunction(builder, rsub, ArgumentType.COLUMN_VECTOR);


        AFunctionObject div = new Div(new Arg(0), new Arg(1));

        DIV_MATRIX = buildPredefinedFunction(builder, div, ArgumentType.MATRIX);
        DIV_SCALAR = buildPredefinedFunction(builder, div, ArgumentType.SCALAR);
        DIV_R_VECTOR = buildPredefinedFunction(builder, div, ArgumentType.ROW_VECTOR);
        DIV_C_VECTOR = buildPredefinedFunction(builder, div, ArgumentType.COLUMN_VECTOR);


        AFunctionObject rdiv = new Div(new Arg(1), new Arg(0));

        RDIV_SCALAR = buildPredefinedFunction(builder, rdiv, ArgumentType.SCALAR);
        RDIV_R_VECTOR = buildPredefinedFunction(builder, rdiv, ArgumentType.ROW_VECTOR);
        RDIV_C_VECTOR = buildPredefinedFunction(builder, rdiv, ArgumentType.COLUMN_VECTOR);

        
        AFunctionObject greaterThen = new Comparator("<", new Arg(1), new Arg(0));
        
        GT_MATRIX = buildPredefinedFunction(builder, greaterThen, ArgumentType.MATRIX);
        GT_SCALAR = buildPredefinedFunction(builder, greaterThen, ArgumentType.SCALAR);
        GT_R_VECTOR = buildPredefinedFunction(builder, greaterThen, ArgumentType.ROW_VECTOR);
        GT_C_VECTOR = buildPredefinedFunction(builder, greaterThen, ArgumentType.COLUMN_VECTOR);
                
        
        AFunctionObject one = new Value(1.0);

        String function = builder.buildFunction(one);
        SET_ONE = builder.getFunctionName();
        CORE.loadFromGeneratedFunction(SET_ONE, function);


        AFunctionObject copy = new Arg(0);
        COPY_MATRIX = buildPredefinedFunctionSingle(builder, copy, ArgumentType.MATRIX);

        CORE.compileMatrixFunctions();

    }

    private static String buildPredefinedFunctionSingle(AFunctionBuilder builder, AFunctionObject functionObject, ArgumentType argumentType) {
        String function = builder.buildFunction(functionObject, argumentType);
        String functionName = builder.getFunctionName();
        CORE.loadFromGeneratedFunction(functionName, function);
        return functionName;
    }

    private static String buildPredefinedFunction(AFunctionBuilder builder, AFunctionObject functionObject, ArgumentType argumentType) {
        String function = builder.buildFunction(functionObject, ArgumentType.MATRIX, argumentType);
        String functionName = builder.getFunctionName();
        CORE.loadFromGeneratedFunction(functionName, function);
        return functionName;
    }

    private cl_mem dataPointer;
    private Optional<cl_mem> randomDataPointer;
    private int clRows, clColumns, clLength;

    public CLFloatMatrix(int rows, int columns, float... values) {
        this.columns = columns;
        this.rows = rows;
        this.length = columns * rows;

        this.clColumns = (int) Math.ceil(columns / (double) CORE.getThreadCount_Y()) * CORE.getThreadCount_Y();
        this.clRows = (int) Math.ceil(rows / (double) CORE.getThreadCount_X()) * CORE.getThreadCount_X();
        this.clLength = clColumns * clRows;

        if (values.length == 0) {
            this.dataPointer = CORE.malloc(this.clLength);
            setZero();
        } else {
            if (rows * columns != values.length) throw new IllegalArgumentException(
                    "rows times columns " + (rows * columns) + " != " +
                            "data length = " + values.length);

            float[] clValues = getCLMatrix(rows, columns, values);
            this.dataPointer = CORE.malloc(clValues);
        }
        randomDataPointer = Optional.empty();
    }

    private float[] getCLMatrix(int rows, int columns, float[] values) {
        float[] clValues = new float[clLength];
        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                clValues[i * clRows + j] = values[i * rows + j];
            }
        }
        return clValues;
    }

    public CLFloatMatrix(float value) {
        this(1, 1, value);
    }


    public static CLFloatMatrix zeros(int rows, int columns) {
        CLFloatMatrix matrix = new CLFloatMatrix(rows, columns);
        matrix.setZero();
        return matrix;
    }

    public static CLFloatMatrix ones(int rows, int columns) {
        CLFloatMatrix matrix = new CLFloatMatrix(rows, columns);
        matrix.setOne();
        return matrix;
    }

    public static CLFloatMatrix rand(int rows, int columns) {
        CLFloatMatrix matrix = new CLFloatMatrix(rows, columns);
        matrix.nextRand();
        return matrix;
    }

    public static CLFloatMatrix randn(int rows, int columns) {
        CLFloatMatrix matrix = new CLFloatMatrix(rows, columns);
        matrix.nextRandn();
        return matrix;
    }

    private void initRandom() {
        if (!randomDataPointer.isPresent()) {
            int[] initRandom = new int[CORE.getThreadCount_Y() * CORE.getThreadCount_X() * 4];
            for (int i = 0; i < initRandom.length; i++) {
                initRandom[i] = Random.nextInt(Integer.MAX_VALUE - 1234) + 1234;
            }
            randomDataPointer = Optional.of(CORE.mallocRandom(initRandom));
        }
    }


    public void setOne() {
        CORE.execute(SET_ONE, this.clRows, this.clColumns, this.rows, this.columns, dataPointer);
    }

    public void setZero() {
        CORE.setZero(this.clRows, this.clColumns, dataPointer);
    }
    
    
    // ADD

    public static void add(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	runMatrixMatrixElementWiseOperation(ADD_MATRIX, matrixA, matrixB, result);
    }

    public static void add(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	runMatrixScalarElementWiseOperation(ADD_SCALAR, matrix, scalar, result);
    }

    public static void addColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(ADD_C_VECTOR, matrix, columnVector, result);
    }

    public static void addRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	runMatrixRowVectorElementWiseOperation(ADD_R_VECTOR, matrix, rowVector, result);
    }


    
    // MUL

    public static void mul(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	runMatrixMatrixElementWiseOperation(MUL_MATRIX, matrixA, matrixB, result);
    }

    public static void mul(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	runMatrixScalarElementWiseOperation(MUL_SCALAR, matrix, scalar, result);
    }

    public static void mulColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(MUL_C_VECTOR, matrix, columnVector, result);
    }

    public static void mulRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	runMatrixRowVectorElementWiseOperation(MUL_R_VECTOR, matrix, rowVector, result);
    }


    
    // SUB

    public static void sub(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	runMatrixMatrixElementWiseOperation(SUB_MATRIX, matrixA, matrixB, result);
    }

    public static void sub(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	runMatrixScalarElementWiseOperation(SUB_SCALAR, matrix, scalar, result);
    }

    public static void subColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(SUB_C_VECTOR, matrix, columnVector, result);
    }

    public static void subRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	runMatrixRowVectorElementWiseOperation(SUB_R_VECTOR, matrix, rowVector, result);
    }

    public static void rsub(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	runMatrixScalarElementWiseOperation(RSUB_SCALAR, matrix, scalar, result);
    }

    public static void rsubColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(RSUB_C_VECTOR, matrix, columnVector, result);
    }

    public static void rsubRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	runMatrixRowVectorElementWiseOperation(RSUB_R_VECTOR, matrix, rowVector, result);
    }


    // DIV

    public static void div(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	runMatrixMatrixElementWiseOperation(DIV_MATRIX, matrixA, matrixB, result);
    }

    public static void div(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	runMatrixScalarElementWiseOperation(DIV_SCALAR, matrix, scalar, result);
    }

    public static void divColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(DIV_C_VECTOR, matrix, columnVector, result);
    }

    public static void divRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	runMatrixRowVectorElementWiseOperation(DIV_R_VECTOR, matrix, rowVector, result);
    }

    public static void rdiv(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	runMatrixScalarElementWiseOperation(RDIV_SCALAR, matrix, scalar, result);
    }

    public static void rdivColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(RDIV_C_VECTOR, matrix, columnVector, result);
    }

    public static void rdivRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	runMatrixRowVectorElementWiseOperation(RDIV_R_VECTOR, matrix, rowVector, result);
    }
    
    
    //// MATRIX_MULTIPLICATION

    public static void mmul(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        CORE.sgemm_nn(a.dataPointer, b.dataPointer, result.dataPointer, a.clRows, b.clColumns, a.clColumns);
    }

    public static void mmulTransposeA(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        CORE.sgemm_tn(a.dataPointer, b.dataPointer, result.dataPointer, a.clColumns, b.clColumns, a.clRows);
    }

    public static void mmulTransposeB(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        CORE.sgemm_nt(a.dataPointer, b.dataPointer, result.dataPointer, a.clRows, b.clRows, a.clColumns);
    }

    
    
    // GREATER THEN
    
    public static void gt(CLFloatMatrix a, float scalar, CLFloatMatrix result) {
    	runMatrixScalarElementWiseOperation(GT_SCALAR, a, scalar, result);
    }

    public static void gt(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
    	runMatrixMatrixElementWiseOperation(GT_MATRIX, a, b, result);
    }

    public static void gtColumnVector(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(GT_C_VECTOR, a, b, result);
    }

    public static void gtRowVector(CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
    	runMatrixRowVectorElementWiseOperation(GT_R_VECTOR, a, b, result);
    }   
    
    
    // TRANSPOSE

    public static void transpose(CLFloatMatrix matrix, CLFloatMatrix transposed) {
        CORE.transpose(matrix.dataPointer, transposed.dataPointer, matrix.clRows, matrix.clColumns, matrix.rows, matrix.columns);
    }

    
    
    // REDUCE

    public static float sum(CLFloatMatrix matrix) {
        return CORE.reduce2D("sumFloats", matrix.dataPointer, matrix.rows, matrix.columns, 0);
    }

    public static CLFloatMatrix testsum(CLFloatMatrix matrix) {
        int tempSizeX = (int) Math.ceil((double) matrix.rows / 32);
        int tempSizeY = (int) Math.ceil((double) matrix.columns / 32);
        CLFloatMatrix result = new CLFloatMatrix(tempSizeX, tempSizeY);
        CORE.reduce2D("sumFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, result.rows, result.columns, 0);
        return result;
    }

    public static float mean(CLFloatMatrix matrix) {
        return sum(matrix) / matrix.length;
    }

    public static float prod(CLFloatMatrix matrix) {
        return CORE.reduce2D("productFloats", matrix.dataPointer, matrix.rows, matrix.columns, 1);
    }

    public static float max(CLFloatMatrix matrix) {
        return CORE.reduce2D("maxFloats", matrix.dataPointer, matrix.rows, matrix.columns, Float.NEGATIVE_INFINITY);
    }

    public static float min(CLFloatMatrix matrix) {
        return CORE.reduce2D("minFloats", matrix.dataPointer, matrix.rows, matrix.columns, Float.POSITIVE_INFINITY);
    }

    public static void columnSums(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceColumns("columnSumsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, 0);
    }

    public static void columnMeans(CLFloatMatrix matrix, CLFloatMatrix result) {
        columnSums(matrix, result);
        div(result, matrix.rows, result);
    }

    public static void columnProds(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceColumns("columnProductsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, 1);
    }

    public static void columnMaxs(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceColumns("columnMaxsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, Float.NEGATIVE_INFINITY);
    }

    public static void columnMins(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceColumns("columnMinsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, Float.POSITIVE_INFINITY);
    }


    // ROW REDUCTION

    public static void rowSums(CLFloatMatrix matrix, CLFloatMatrix result) {
        CORE.reduceRows("rowSumsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, 0);
    }

    public static void rowMeans(CLFloatMatrix matrix, CLFloatMatrix result) {
        rowSums(matrix, result);
        div(result, matrix.columns, result);
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

    
    
    /**
     * Führe ein OpenCL Programm auf zwei gleich große Matrizen durch,
     * das Resultat wird in eine dritte Matrix gespeichert
     *  
     * @param programId
     * @param a
     * @param b
     * @param result
     */
	protected static void runMatrixMatrixElementWiseOperation(String programId, CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
		checkSameSize(a, b, result);
        CORE.execute(programId, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
	}
	
	/**
     * Führe ein OpenCL Programm auf einer Matrix und einem Scalar durch,
     * das Resultat wird in eine zweite Matrix gespeichert
     * 
	 * @param programId
	 * @param matrix
	 * @param scalar
	 * @param result
	 */
	protected static void runMatrixScalarElementWiseOperation(String programId, CLFloatMatrix a, float scalar, CLFloatMatrix result) {
	    CLFloatMatrix b = new CLFloatMatrix(1, 1, scalar);
        CORE.execute(programId, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
        b.free();
	}
	
	/**
     * Führe ein OpenCL Programm auf einer Matrix und einem Row Vector durch,
     * das Resultat wird in eine zweite Matrix gespeichert
     * 
	 * @param programId
	 * @param matrix 
	 * @param row vector
	 * @param result
	 */
	protected static void runMatrixRowVectorElementWiseOperation(String programId, CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(programId, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
	}	
	
	
	/**
     * Führe ein OpenCL Programm auf einer Matrix und einem Column Vector durch,
     * das Resultat wird in eine zweite Matrix gespeichert
     * 
	 * @param programId
	 * @param matrix
	 * @param column vector
	 * @param result
	 */
	protected static void runMatrixColumnVectorElementWiseOperation(String programId, CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
		checkColumnVectorSize(a, b, result);
		CORE.execute(programId, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
	}	

   


    // RANDOM

    public void nextRand() {
        initRandom();
        CORE.uniform(dataPointer, randomDataPointer.get(), this.clRows, this.clColumns, this.rows, this.columns);
    }

    public void nextRandn() {
        initRandom();
        CORE.boxMuller(dataPointer, randomDataPointer.get(), this.clRows, this.clColumns, this.rows, this.columns);
    }

    public void free() {
        CORE.free(dataPointer);
        if (randomDataPointer.isPresent()) {
            CORE.free(randomDataPointer.get());
        }
    }

    @Override
    public void getColumnWiseOn(float[] values) {
        float[] clValues = new float[clLength];
        CORE.getData(dataPointer, clValues);
        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                values[i * rows + j] = clValues[i * clRows + j];
            }
        }
    }

}
