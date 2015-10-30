package org.nblas.cl;


import org.jblas.util.Random;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.nblas.generic.ANativeFloatMatrix;
import org.nblas.generic.Subprogram;
import org.nblas.function.AFunctionBuilder;
import org.nblas.function.ArgumentType;
import org.nblas.function.common.Arg;
import org.nblas.function.common.Value;
import org.nblas.function.generic.AFunctionObject;
import org.nblas.function.predefined.MatrixFunctions;
import org.nblas.function.predefined.binary.Add;
import org.nblas.function.predefined.binary.Comparator;
import org.nblas.function.predefined.binary.Div;
import org.nblas.function.predefined.binary.Mul;
import org.nblas.function.predefined.binary.Sub;
import org.nblas.function.predefined.unary.Exp;
import org.nblas.function.predefined.unary.Negate;

import java.util.Optional;

public class CLFloatMatrix extends ANativeFloatMatrix {


    private static final CLCore CORE = CLCore.getCore();

    private static final Subprogram<cl_kernel> ADD_MATRIX;
    private static final Subprogram<cl_kernel> ADD_SCALAR;
    private static final Subprogram<cl_kernel> ADD_C_VECTOR;
    private static final Subprogram<cl_kernel> ADD_R_VECTOR;

    private static final Subprogram<cl_kernel> MUL_MATRIX;
    private static final Subprogram<cl_kernel> MUL_SCALAR;
    private static final Subprogram<cl_kernel> MUL_C_VECTOR;
    private static final Subprogram<cl_kernel> MUL_R_VECTOR;

    private static final Subprogram<cl_kernel> SUB_MATRIX;
    private static final Subprogram<cl_kernel> SUB_SCALAR;
    private static final Subprogram<cl_kernel> SUB_C_VECTOR;
    private static final Subprogram<cl_kernel> SUB_R_VECTOR;

    private static final Subprogram<cl_kernel> RSUB_SCALAR;
    private static final Subprogram<cl_kernel> RSUB_C_VECTOR;
    private static final Subprogram<cl_kernel> RSUB_R_VECTOR;

    private static final Subprogram<cl_kernel> DIV_MATRIX;
    private static final Subprogram<cl_kernel> DIV_SCALAR;
    private static final Subprogram<cl_kernel> DIV_C_VECTOR;
    private static final Subprogram<cl_kernel> DIV_R_VECTOR;

    private static final Subprogram<cl_kernel> RDIV_SCALAR;
    private static final Subprogram<cl_kernel> RDIV_C_VECTOR;
    private static final Subprogram<cl_kernel> RDIV_R_VECTOR;

    private static final Subprogram<cl_kernel> SET_ONE;
    private static final Subprogram<cl_kernel> COPY_MATRIX;
//    private static final Subprogram<cl_kernel> DUP;
    
    // greater than
    private static final Subprogram<cl_kernel> GT_MATRIX;
    private static final Subprogram<cl_kernel> GT_SCALAR;
    private static final Subprogram<cl_kernel> GT_C_VECTOR;
    private static final Subprogram<cl_kernel> GT_R_VECTOR;
    
    // greater than or equal
    private static final Subprogram<cl_kernel> GE_MATRIX;
    private static final Subprogram<cl_kernel> GE_SCALAR;
    private static final Subprogram<cl_kernel> GE_C_VECTOR;
    private static final Subprogram<cl_kernel> GE_R_VECTOR;
    
    // lower than
    private static final Subprogram<cl_kernel> LT_MATRIX;
    private static final Subprogram<cl_kernel> LT_SCALAR;
    private static final Subprogram<cl_kernel> LT_C_VECTOR;
    private static final Subprogram<cl_kernel> LT_R_VECTOR;
    
    // lower than or equal
    private static final Subprogram<cl_kernel> LE_MATRIX;
    private static final Subprogram<cl_kernel> LE_SCALAR;
    private static final Subprogram<cl_kernel> LE_C_VECTOR;
    private static final Subprogram<cl_kernel> LE_R_VECTOR;
    
    // equal
    private static final Subprogram<cl_kernel> EQ_MATRIX;
    private static final Subprogram<cl_kernel> EQ_SCALAR;
    private static final Subprogram<cl_kernel> EQ_C_VECTOR;
    private static final Subprogram<cl_kernel> EQ_R_VECTOR;
    
    // not equal
    private static final Subprogram<cl_kernel> NE_MATRIX;
    private static final Subprogram<cl_kernel> NE_SCALAR;
    private static final Subprogram<cl_kernel> NE_C_VECTOR;
    private static final Subprogram<cl_kernel> NE_R_VECTOR;
    
    // special functions  
    private static final Subprogram<cl_kernel> EXP;
    private static final Subprogram<cl_kernel> NEG;
    private static final Subprogram<cl_kernel> SIGMOID;
    

    static {
        CLFloatFunctionBuilder builder = new CLFloatFunctionBuilder();
        // add Functions
        AFunctionObject add = new Add(new Arg(0), new Arg(1));

        ADD_MATRIX = buildPredefinedFunction(builder, add, ArgumentType.MATRIX, ArgumentType.MATRIX);
        ADD_SCALAR = buildPredefinedFunction(builder, add, ArgumentType.MATRIX, ArgumentType.SCALAR);
        ADD_R_VECTOR = buildPredefinedFunction(builder, add, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        ADD_C_VECTOR = buildPredefinedFunction(builder, add, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);


        AFunctionObject mul = new Mul(new Arg(0), new Arg(1));

        MUL_MATRIX = buildPredefinedFunction(builder, mul, ArgumentType.MATRIX, ArgumentType.MATRIX);
        MUL_SCALAR = buildPredefinedFunction(builder, mul, ArgumentType.MATRIX, ArgumentType.SCALAR);
        MUL_R_VECTOR = buildPredefinedFunction(builder, mul, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        MUL_C_VECTOR = buildPredefinedFunction(builder, mul, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);


        AFunctionObject sub = new Sub(new Arg(0), new Arg(1));

        SUB_MATRIX = buildPredefinedFunction(builder, sub, ArgumentType.MATRIX, ArgumentType.MATRIX);
        SUB_SCALAR = buildPredefinedFunction(builder, sub, ArgumentType.MATRIX, ArgumentType.SCALAR);
        SUB_R_VECTOR = buildPredefinedFunction(builder, sub, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        SUB_C_VECTOR = buildPredefinedFunction(builder, sub, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);


        AFunctionObject rsub = new Sub(new Arg(1), new Arg(0));

        RSUB_SCALAR = buildPredefinedFunction(builder, rsub, ArgumentType.MATRIX, ArgumentType.SCALAR);
        RSUB_R_VECTOR = buildPredefinedFunction(builder, rsub, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        RSUB_C_VECTOR = buildPredefinedFunction(builder, rsub, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);


        AFunctionObject div = new Div(new Arg(0), new Arg(1));

        DIV_MATRIX = buildPredefinedFunction(builder, div, ArgumentType.MATRIX, ArgumentType.MATRIX);
        DIV_SCALAR = buildPredefinedFunction(builder, div, ArgumentType.MATRIX, ArgumentType.SCALAR);
        DIV_R_VECTOR = buildPredefinedFunction(builder, div, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        DIV_C_VECTOR = buildPredefinedFunction(builder, div, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);


        AFunctionObject rdiv = new Div(new Arg(1), new Arg(0));

        RDIV_SCALAR = buildPredefinedFunction(builder, rdiv, ArgumentType.MATRIX, ArgumentType.SCALAR);
        RDIV_R_VECTOR = buildPredefinedFunction(builder, rdiv, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        RDIV_C_VECTOR = buildPredefinedFunction(builder, rdiv, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);

        
        AFunctionObject greaterThan = new Comparator(">", new Arg(0), new Arg(1));
        
        GT_MATRIX = buildPredefinedFunction(builder, greaterThan, ArgumentType.MATRIX, ArgumentType.MATRIX);
        GT_SCALAR = buildPredefinedFunction(builder, greaterThan, ArgumentType.MATRIX, ArgumentType.SCALAR);
        GT_R_VECTOR = buildPredefinedFunction(builder, greaterThan, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        GT_C_VECTOR = buildPredefinedFunction(builder, greaterThan, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);                
        
        AFunctionObject greaterEqual = new Comparator(">=", new Arg(0), new Arg(1));
        
        GE_MATRIX = buildPredefinedFunction(builder, greaterEqual, ArgumentType.MATRIX, ArgumentType.MATRIX);
        GE_SCALAR = buildPredefinedFunction(builder, greaterEqual, ArgumentType.MATRIX, ArgumentType.SCALAR);
        GE_R_VECTOR = buildPredefinedFunction(builder, greaterEqual, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        GE_C_VECTOR = buildPredefinedFunction(builder, greaterEqual, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);
        
        AFunctionObject lowerThan = new Comparator("<", new Arg(0), new Arg(1));
        
        LT_MATRIX = buildPredefinedFunction(builder, lowerThan, ArgumentType.MATRIX, ArgumentType.MATRIX);
        LT_SCALAR = buildPredefinedFunction(builder, lowerThan, ArgumentType.MATRIX, ArgumentType.SCALAR);
        LT_R_VECTOR = buildPredefinedFunction(builder, lowerThan, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        LT_C_VECTOR = buildPredefinedFunction(builder, lowerThan, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);                
        
        AFunctionObject lowerEqual = new Comparator("<=", new Arg(0), new Arg(1));
        
        LE_MATRIX = buildPredefinedFunction(builder, lowerEqual, ArgumentType.MATRIX, ArgumentType.MATRIX);
        LE_SCALAR = buildPredefinedFunction(builder, lowerEqual, ArgumentType.MATRIX, ArgumentType.SCALAR);
        LE_R_VECTOR = buildPredefinedFunction(builder, lowerEqual, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        LE_C_VECTOR = buildPredefinedFunction(builder, lowerEqual, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);
        
        AFunctionObject equal = new Comparator("==", new Arg(0), new Arg(1));
        
        EQ_MATRIX = buildPredefinedFunction(builder, equal, ArgumentType.MATRIX, ArgumentType.MATRIX);
        EQ_SCALAR = buildPredefinedFunction(builder, equal, ArgumentType.MATRIX, ArgumentType.SCALAR);
        EQ_R_VECTOR = buildPredefinedFunction(builder, equal, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        EQ_C_VECTOR = buildPredefinedFunction(builder, equal, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);                
        
        AFunctionObject notEqual = new Comparator("!=", new Arg(0), new Arg(1));
        
        NE_MATRIX = buildPredefinedFunction(builder, notEqual, ArgumentType.MATRIX, ArgumentType.MATRIX);
        NE_SCALAR = buildPredefinedFunction(builder, notEqual, ArgumentType.MATRIX, ArgumentType.SCALAR);
        NE_R_VECTOR = buildPredefinedFunction(builder, notEqual, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        NE_C_VECTOR = buildPredefinedFunction(builder, notEqual, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);
        
        AFunctionObject one = new Value(1.0);
        SET_ONE = buildPredefinedFunction(builder, one);
        
        AFunctionObject exp = new Exp(new Arg(0));
        EXP = buildPredefinedFunction(builder, exp, ArgumentType.MATRIX);
        
//        AFunctionObject duplicate = new Duplicate(new Arg(0));
//        DUP = buildPredefinedFunction(builder, duplicate, ArgumentType.MATRIX);        
        
        AFunctionObject negate = new Negate(new Arg(0));
        NEG = buildPredefinedFunction(builder, negate, ArgumentType.MATRIX);
        
        AFunctionObject sigmoid = MatrixFunctions.sigmoid(new Arg(0));
        SIGMOID = buildPredefinedFunction(builder, sigmoid, ArgumentType.MATRIX);
        
        AFunctionObject copy = new Arg(0);
        COPY_MATRIX = buildPredefinedFunction(builder, copy, ArgumentType.MATRIX);

        
        CORE.compileMatrixFunctions();
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

    private static Subprogram<cl_kernel> buildPredefinedFunction(AFunctionBuilder<cl_kernel> builder, AFunctionObject functionObject, ArgumentType... argumentTypes) {
    	Subprogram<cl_kernel> subprogram = builder.buildFunction(functionObject, argumentTypes);
    	subprogram.setCustom(false);
        CORE.loadFromGeneratedSubprogram(subprogram);
        return subprogram;
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
            randomDataPointer = Optional.of(CORE.malloc(initRandom));
        }
    }


    public void setOne() {
        CORE.execute(SET_ONE, this.clRows, this.clColumns, this.rows, this.columns, dataPointer);
    }

    public void setZero() {
        CORE.execute(CLPredefined.getSubprogram("setZero"), this.clRows, this.clColumns, dataPointer);
    }
    
    public CLFloatMatrix repmat(int rowTimes, int columnTimes) {
        CLFloatMatrix result = new CLFloatMatrix(rows * rowTimes, columns * columnTimes);
        CORE.repmat(dataPointer, result.dataPointer,
                result.clRows, result.clColumns,
                result.rows, result.columns, rows, columns, clRows);
        return result;
    }

    public void setSubMatrix(CLFloatMatrix matrix, int rowOffset, int columnOffset) {
    	CORE.setSubMatrix(matrix.dataPointer, dataPointer,
                matrix.clRows, matrix.clColumns,
                matrix.rows, matrix.columns, rowOffset, columnOffset, clRows);
    }

    public CLFloatMatrix getSubMatrix(CLFloatMatrix result, int rowOffset, int columnOffset) {
        CORE.getSubMatrix(dataPointer, result.dataPointer,
                result.clRows, result.clColumns,
                result.rows, result.columns, rowOffset, columnOffset, clRows);
        return result;
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

    
    
    // GREATER THAN
    
    public static void gt(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	runMatrixScalarElementWiseOperation(GT_SCALAR, matrix, scalar, result);
    }

    public static void gt(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	runMatrixMatrixElementWiseOperation(GT_MATRIX, matrixA, matrixB, result);
    }

    public static void gtColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(GT_C_VECTOR, matrix, columnVector, result);
    }

    public static void gtRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	runMatrixRowVectorElementWiseOperation(GT_R_VECTOR, matrix, rowVector, result);
    }   
    
    
    
    // GREATER THAN OR EQUAL
    
    public static void ge(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	runMatrixScalarElementWiseOperation(GE_SCALAR, matrix, scalar, result);
    }

    public static void ge(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	runMatrixMatrixElementWiseOperation(GE_MATRIX, matrixA, matrixB, result);
    }

    public static void geColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(GE_C_VECTOR, matrix, columnVector, result);
    }

    public static void geRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	runMatrixRowVectorElementWiseOperation(GE_R_VECTOR, matrix, rowVector, result);
    }   
    
    
    
    // LOWER THAN
    
    public static void lt(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	runMatrixScalarElementWiseOperation(LT_SCALAR, matrix, scalar, result);
    }

    public static void lt(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	runMatrixMatrixElementWiseOperation(LT_MATRIX, matrixA, matrixB, result);
    }

    public static void ltColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(LT_C_VECTOR, matrix, columnVector, result);
    }

    public static void ltRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	runMatrixRowVectorElementWiseOperation(LT_R_VECTOR, matrix, rowVector, result);
    }   
    
    
    
    // LOWER THAN OR EQUAL
    
    public static void le(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	runMatrixScalarElementWiseOperation(LE_SCALAR, matrix, scalar, result);
    }

    public static void le(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	runMatrixMatrixElementWiseOperation(LE_MATRIX, matrixA, matrixB, result);
    }

    public static void leColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(LE_C_VECTOR, matrix, columnVector, result);
    }

    public static void leRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	runMatrixRowVectorElementWiseOperation(LE_R_VECTOR, matrix, rowVector, result);
    } 
    
    
    
    // EQUAL
    
    public static void eq(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	runMatrixScalarElementWiseOperation(EQ_SCALAR, matrix, scalar, result);
    }

    public static void eq(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	runMatrixMatrixElementWiseOperation(EQ_MATRIX, matrixA, matrixB, result);
    }

    public static void eqColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(EQ_C_VECTOR, matrix, columnVector, result);
    }

    public static void eqRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	runMatrixRowVectorElementWiseOperation(EQ_R_VECTOR, matrix, rowVector, result);
    }   
    
    
    
    // NOT EQUAL
    
    public static void ne(CLFloatMatrix matrix, float scalar, CLFloatMatrix result) {
    	runMatrixScalarElementWiseOperation(NE_SCALAR, matrix, scalar, result);
    }

    public static void ne(CLFloatMatrix matrixA, CLFloatMatrix matrixB, CLFloatMatrix result) {
    	runMatrixMatrixElementWiseOperation(NE_MATRIX, matrixA, matrixB, result);
    }

    public static void neColumnVector(CLFloatMatrix matrix, CLFloatMatrix columnVector, CLFloatMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(NE_C_VECTOR, matrix, columnVector, result);
    }

    public static void neRowVector(CLFloatMatrix matrix, CLFloatMatrix rowVector, CLFloatMatrix result) {
    	runMatrixRowVectorElementWiseOperation(NE_R_VECTOR, matrix, rowVector, result);
    } 
 
    public static void dup(CLFloatMatrix matrix, CLFloatMatrix result) {
		runMatrixElementWiseOperation(COPY_MATRIX, matrix, result);
	}
	
    public static void exp(CLFloatMatrix matrix, CLFloatMatrix result) {
		runMatrixElementWiseOperation(EXP, matrix, result);
	}

    public static void neg(CLFloatMatrix matrix, CLFloatMatrix result) {
		runMatrixElementWiseOperation(NEG, matrix, result);
	}
	
	public static void sigmoid(CLFloatMatrix matrix, CLFloatMatrix result) {
		runMatrixElementWiseOperation(SIGMOID, matrix, result);
	}
    
    // TRANSPOSE

    public static void transpose(CLFloatMatrix matrix, CLFloatMatrix transposed) {
        CORE.transpose(matrix.dataPointer, transposed.dataPointer, matrix.clRows, matrix.clColumns, matrix.rows, matrix.columns);
    }

    
    
    // REDUCE

    public static float sum(CLFloatMatrix matrix) {
//        return CORE.reduce2D("sumFloats", matrix.dataPointer, matrix.rows, matrix.columns, 0);
        return CORE.reduce1D("sumFloats1D", matrix.dataPointer, matrix.clRows*matrix.clColumns);
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
	protected static void runMatrixMatrixElementWiseOperation(Subprogram<cl_kernel> subprogram, CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
		checkSameSize(a, b, result);
        CORE.execute(subprogram, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
	}
	
	
	
	
	/**
     * Führe ein OpenCL Programm auf einer Matrix durch,
     * das Resultat wird in eine zweite Matrix gespeichert
     * 
	 * @param programId
	 * @param matrix
	 * @param scalar
	 * @param result
	 */
	protected static void runMatrixElementWiseOperation(Subprogram<cl_kernel> subprogram, CLFloatMatrix a, CLFloatMatrix result) {
		checkSameSize(a, result);
        CORE.execute(subprogram, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer);
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
	protected static void runMatrixScalarElementWiseOperation(Subprogram<cl_kernel> subprogram, CLFloatMatrix a, float scalar, CLFloatMatrix result) {
	    CLFloatMatrix b = new CLFloatMatrix(1, 1, scalar);
        CORE.execute(subprogram, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
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
	protected static void runMatrixRowVectorElementWiseOperation(Subprogram<cl_kernel> subprogram, CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(subprogram, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
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
	protected static void runMatrixColumnVectorElementWiseOperation(Subprogram<cl_kernel> subprogram, CLFloatMatrix a, CLFloatMatrix b, CLFloatMatrix result) {
		checkColumnVectorSize(a, b, result);
		CORE.execute(subprogram, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
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

    public static void waitOnComplete() {
        CORE.waitOnComplete();
    }
    
    
    @Override
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
