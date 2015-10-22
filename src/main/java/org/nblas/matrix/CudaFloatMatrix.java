package org.nblas.matrix;


import org.nblas.function.ArgumentType;
import org.nblas.function.functionobjects.common.Arg;
import org.nblas.function.functionobjects.common.Value;
import org.nblas.function.functionobjects.generic.AFunctionObject;
import org.nblas.function.functionobjects.predefined.binary.Add;
import org.nblas.function.functionobjects.predefined.binary.Div;
import org.nblas.function.functionobjects.predefined.binary.Mul;
import org.nblas.function.functionobjects.predefined.binary.Sub;

import jcuda.Pointer;

class CudaFloatMatrix extends AMatrix {

    private static final CudaCore CORE = CudaCore.getCore();
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

    private static final String SET_ZERO;
    private static final String SET_ONE;
    private static final String COPY_MATRIX;

    static {
        CudaFloatFunctionBuilder builder = new CudaFloatFunctionBuilder();
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


        AFunctionObject zero = new Value(0.0);

        String function = builder.buildFunction(zero);
        SET_ZERO = builder.getFunctionName();
        CORE.loadFromGeneratedFunction(SET_ZERO, function);

        AFunctionObject one = new Value(1.0);

        function = builder.buildFunction(one);
        SET_ONE = builder.getFunctionName();
        CORE.loadFromGeneratedFunction(SET_ONE, function);


        AFunctionObject copy = new Arg(0);
        COPY_MATRIX = buildPredefinedFunctionSingle(builder, copy, ArgumentType.MATRIX);

        for (String functionName : CudaPredefined.kernels.keySet()) {
            CORE.loadFromGeneratedFunction(functionName, CudaPredefined.kernels.get(functionName));
        }

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

    private final Pointer dataPointer;

    public void free() {
        CORE.free(dataPointer);
    }

    public CudaFloatMatrix(int rows, int columns, float... values) {
        this.columns = columns;
        this.rows = rows;
        this.length = columns * rows;
        if (values.length == 0) {
            this.dataPointer = CORE.malloc(this.length);
            setZero();
        } else {
            if (rows * columns != values.length) throw new IllegalArgumentException(
                    "rows times columns " + (rows * columns) + " != " +
                            "data length = " + values.length);


            this.dataPointer = CORE.malloc(values);
        }
    }

    public CudaFloatMatrix(float value) {
        this(1, 1, value);
    }

    private CudaFloatMatrix(int rows, int columns) {
        this.columns = columns;
        this.rows = rows;
        this.length = columns * rows;
        this.dataPointer = CORE.malloc(this.length);
    }

    public static CudaFloatMatrix fromDataColumnWise(int rows, int columns, float... values) {
        return new CudaFloatMatrix(rows, columns, values);
    }

    public static CudaFloatMatrix zeros(int rows, int columns) {
        CudaFloatMatrix matrix = new CudaFloatMatrix(rows, columns);
        matrix.setZero();
        return matrix;
    }

    public static CudaFloatMatrix ones(int rows, int columns) {
        CudaFloatMatrix matrix = new CudaFloatMatrix(rows, columns);
        matrix.setOne();
        return matrix;
    }

    public static CudaFloatMatrix rand(int rows, int columns) {
        CudaFloatMatrix matrix = new CudaFloatMatrix(rows, columns);
        matrix.nextRand();
        return matrix;
    }

    public static CudaFloatMatrix randn(int rows, int columns) {
        CudaFloatMatrix matrix = new CudaFloatMatrix(rows, columns);
        matrix.nextRandn();
        return matrix;
    }

    public static CudaFloatMatrix eye(int length) {
        float[] values = new float[length * length];
        for (int i = 0; i < length; i++) {
            values[i * length + i] = 1.0f;
        }
        return new CudaFloatMatrix(length, length, values);
    }

    public void setOne() {
        CORE.execute(SET_ONE, this.rows, this.columns, dataPointer);
    }

    public void setZero() {
        CORE.execute(SET_ZERO, this.rows, this.columns, dataPointer);
    }

    public static void copy(CudaFloatMatrix source, CudaFloatMatrix copy) {
        checkSameSize(source, copy);
        CORE.execute(COPY_MATRIX, source.rows, source.columns, copy.dataPointer, source.dataPointer);
    }


    ///// ARITHMETICS


    // ADD

    public static void add(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkSameSize(a, b, result);
        CORE.execute(ADD_MATRIX, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }

    public static void add(CudaFloatMatrix a, float x, CudaFloatMatrix result) {
        CudaFloatMatrix b = new CudaFloatMatrix(1, 1, x);

        CORE.execute(ADD_SCALAR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
        b.free();
    }

    public static void addColumnVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        CORE.execute(ADD_C_VECTOR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }

    public static void addRowVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(ADD_R_VECTOR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }


    // MUL

    public static void mul(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkSameSize(a, b, result);
        CORE.execute(MUL_MATRIX, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }

    public static void mul(CudaFloatMatrix a, float x, CudaFloatMatrix result) {
        CudaFloatMatrix b = new CudaFloatMatrix(1, 1, x);

        CORE.execute(MUL_SCALAR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
        b.free();
    }

    public static void mulColumnVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        CORE.execute(MUL_C_VECTOR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }

    public static void mulRowVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(MUL_R_VECTOR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }


    // SUB

    public static void sub(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkSameSize(a, b, result);
        CORE.execute(SUB_MATRIX, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }

    public static void sub(CudaFloatMatrix a, float x, CudaFloatMatrix result) {
        CudaFloatMatrix b = new CudaFloatMatrix(1, 1, x);

        CORE.execute(SUB_SCALAR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
        b.free();
    }

    public static void subColumnVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        CORE.execute(SUB_C_VECTOR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }

    public static void subRowVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(SUB_R_VECTOR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }


    public static void rsub(CudaFloatMatrix a, float x, CudaFloatMatrix result) {
        CudaFloatMatrix b = new CudaFloatMatrix(1, 1, x);

        CORE.execute(RSUB_SCALAR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
        b.free();
    }

    public static void rsubColumnVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        CORE.execute(RSUB_C_VECTOR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }

    public static void rsubRowVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(RSUB_R_VECTOR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }


    // DIV

    public static void div(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkSameSize(a, b, result);
        CORE.execute(DIV_MATRIX, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }

    public static void div(CudaFloatMatrix a, float x, CudaFloatMatrix result) {
        CudaFloatMatrix b = new CudaFloatMatrix(1, 1, x);

        CORE.execute(DIV_SCALAR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
        b.free();
    }

    public static void divColumnVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        CORE.execute(DIV_C_VECTOR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }

    public static void divRowVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(DIV_R_VECTOR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }


    public static void rdiv(CudaFloatMatrix a, float x, CudaFloatMatrix result) {
        CudaFloatMatrix b = new CudaFloatMatrix(1, 1, x);

        CORE.execute(RDIV_SCALAR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
        b.free();
    }

    public static void rdivColumnVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        CORE.execute(RDIV_C_VECTOR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }

    public static void rdivRowVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(RDIV_R_VECTOR, result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }


    //// MATRIX_MULTIPLICATION

    public static void mmul(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        CORE.mmul(a.dataPointer, a.rows, a.columns,
                b.dataPointer, b.columns, result.dataPointer);
    }

    public static void mmulTransposeA(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        CORE.mmulTransposeA(a.dataPointer, a.rows, a.columns,
                b.dataPointer, b.columns, result.dataPointer);
    }

    public static void mmulTransposeB(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        CORE.mmulTransposeB(a.dataPointer, a.rows, a.columns,
                b.dataPointer, b.rows, result.dataPointer);
    }

    // TRANSPOSE

    public static void transpose(CudaFloatMatrix matrix, CudaFloatMatrix transposed) {
        CORE.transpose(matrix.dataPointer, transposed.dataPointer, matrix.rows, matrix.columns);
    }


    ///// REDUCTION


    // FULL REDUCTION

    public static float sum(CudaFloatMatrix matrix) {
        return CORE.reduce("sumFloats", matrix.dataPointer, matrix.length, 0);
    }

    public static float mean(CudaFloatMatrix matrix) {
        return sum(matrix) / matrix.length;
    }

    public static float prod(CudaFloatMatrix matrix) {
        return CORE.reduce("productFloats", matrix.dataPointer, matrix.length, 1);
    }

    public static float max(CudaFloatMatrix matrix) {
        return CORE.reduce("maxFloats", matrix.dataPointer, matrix.length, Float.NEGATIVE_INFINITY);
    }

    public static float min(CudaFloatMatrix matrix) {
        return CORE.reduce("minFloats", matrix.dataPointer, matrix.length, Float.POSITIVE_INFINITY);
    }


    // ROW REDUCTION

    public static void rowSums(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceRows("rowSumsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, 0);
    }

    public static void rowMeans(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        rowSums(matrix, result);
        div(result, matrix.columns, result);
    }

    public static void rowProds(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceRows("rowProductsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, 1);
    }

    public static void rowMaxs(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceRows("rowMaxsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, Float.NEGATIVE_INFINITY);
    }

    public static void rowMins(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceRows("rowMinsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, Float.POSITIVE_INFINITY);

    }


    // COLUMN REDUCTION

    public static void columnSums(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceColumns("columnSumsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, 0);
    }

    public static void columnMeans(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        columnSums(matrix, result);
        div(result, matrix.rows, result);
    }

    public static void columnProds(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceColumns("columnProductsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, 1);
    }

    public static void columnMaxs(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceColumns("columnMaxsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, Float.NEGATIVE_INFINITY);
    }

    public static void columnMins(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceColumns("columnMinsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, Float.POSITIVE_INFINITY);
    }


    //// RANDOM

    public void nextRand() {
        CORE.rand(dataPointer, length);
    }

    public void nextRandn() {
        CORE.randn(dataPointer, length);
    }

    //// GETTER & SETTER


    // MATRIX GETTER

    public CudaFloatMatrix getSubMatrix(int aRow, int bRow, int aColumn, int bColumn) {
        CudaFloatMatrix result = new CudaFloatMatrix(bRow - aRow, bColumn - aColumn);
        CORE.getSubMatrix(dataPointer, result.dataPointer, result.rows, result.columns, rows, aRow, aColumn);
        return result;
    }


    public void getColumnWiseOn(float[] values) {
        if (getRows() * getColumns() != values.length)
            throw new IllegalArgumentException("Array's length is not the size of rows times columns.");
        CORE.getData(dataPointer, values);
    }


    // MATRIX SETTER

    public void setSubMatrix(CudaFloatMatrix matrix, int offsetRow, int offsetColumn) {
        CORE.setSubMatrix(dataPointer, matrix.dataPointer, matrix.rows, matrix.rows, rows, offsetRow, offsetColumn);
    }


}
