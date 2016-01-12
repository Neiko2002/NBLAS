package org.nblas.cuda;


import org.nblas.Context;
import org.nblas.FloatMatrix;
import org.nblas.function.AFunctionBuilder;
import org.nblas.function.ArgumentType;
import org.nblas.function.common.Arg;
import org.nblas.function.common.Value;
import org.nblas.function.generic.AFunctionObject;
import org.nblas.function.predefined.binary.Add;
import org.nblas.function.predefined.binary.Div;
import org.nblas.function.predefined.binary.Mul;
import org.nblas.function.predefined.binary.Sub;
import org.nblas.generic.Subprogram;

import jcuda.driver.CUfunction;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class CudaFloatMatrix extends CudaMatrix implements FloatMatrix  {

    private static final Subprogram<CUfunction> ADD_MATRIX;
    private static final Subprogram<CUfunction> ADD_SCALAR;
    private static final Subprogram<CUfunction> ADD_C_VECTOR;
    private static final Subprogram<CUfunction> ADD_R_VECTOR;

    private static final Subprogram<CUfunction> MUL_MATRIX;
    private static final Subprogram<CUfunction> MUL_SCALAR;
    private static final Subprogram<CUfunction> MUL_C_VECTOR;
    private static final Subprogram<CUfunction> MUL_R_VECTOR;

    private static final Subprogram<CUfunction> SUB_MATRIX;
    private static final Subprogram<CUfunction> SUB_SCALAR;
    private static final Subprogram<CUfunction> SUB_C_VECTOR;
    private static final Subprogram<CUfunction> SUB_R_VECTOR;

    private static final Subprogram<CUfunction> RSUB_SCALAR;
    private static final Subprogram<CUfunction> RSUB_C_VECTOR;
    private static final Subprogram<CUfunction> RSUB_R_VECTOR;

    private static final Subprogram<CUfunction> DIV_MATRIX;
    private static final Subprogram<CUfunction> DIV_SCALAR;
    private static final Subprogram<CUfunction> DIV_C_VECTOR;
    private static final Subprogram<CUfunction> DIV_R_VECTOR;

    private static final Subprogram<CUfunction> RDIV_SCALAR;
    private static final Subprogram<CUfunction> RDIV_C_VECTOR;
    private static final Subprogram<CUfunction> RDIV_R_VECTOR;

    private static final Subprogram<CUfunction> SET_ZERO;
    private static final Subprogram<CUfunction> SET_ONE;
    private static final Subprogram<CUfunction> COPY_MATRIX;

    static {
        CudaFloatFunctionBuilder builder = new CudaFloatFunctionBuilder();
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

        
        AFunctionObject zero = new Value(0.0);

        SET_ZERO = buildPredefinedFunction(builder, zero);

        
        AFunctionObject one = new Value(1.0);

        SET_ONE = buildPredefinedFunction(builder, one);
       
        
        AFunctionObject copy = new Arg(0);
        
        COPY_MATRIX = buildPredefinedFunction(builder, copy, ArgumentType.MATRIX);

        
        for (Subprogram<CUfunction> subprogram : CudaPredefined.kernels.values()) {
            CORE.loadFromGeneratedFunction(subprogram);
        }
    }
    
    private static Subprogram<CUfunction> buildPredefinedFunction(AFunctionBuilder<CUfunction> builder, AFunctionObject functionObject, ArgumentType... argumentTypes) {
        Subprogram<CUfunction> subprogram = builder.buildFunction(functionObject, argumentTypes);
        subprogram.setCustom(false);
        CORE.loadFromGeneratedFunction(subprogram);
        return subprogram;
    }


    /**
     * dirty allocation
     * 
     * @param rows
     * @param columns
     */
    public CudaFloatMatrix(int rows, int columns) {
        super(rows, columns);
        this.dataPointer = CORE.malloc(this.length);
     }

    public CudaFloatMatrix(int rows, int columns, float[] values) {
       super(rows, columns);

       if (rows * columns != values.length) throw new IllegalArgumentException(
               "rows times columns " + (rows * columns) + " != " + "data length = " + values.length);

       this.dataPointer = CORE.malloc(values);
    }
 
    
    
    // ---------------------------------- utility methods -------------------------------------

	@Override
	public Context getContext() {
		return Context.createOpenCLSinglePrecisionContext();
	}
	
    @Override
    public FloatMatrix readRowMajor(float[] values) {
        if (getRows() * getColumns() != values.length)
            throw new IllegalArgumentException("Array's length is not the size of rows times columns.");
        CORE.getData(dataPointer, values);
        return this;
    }
    
	@Override
	public String toString() {
		return toString1D();
	}
    
	@Override
	public FloatMatrix dup(FloatMatrix source, FloatMatrix destination) {
	   	CudaFloatMatrix src = (CudaFloatMatrix)source;
    	CudaFloatMatrix dst = (CudaFloatMatrix)destination;
		checkSameSize(src, dst);
	    CORE.execute(COPY_MATRIX.getProgramName(), src.rows, src.columns, dst.dataPointer, src.dataPointer);
	    return destination;
	}

	@Override
    public FloatMatrix transpose(FloatMatrix matrix, FloatMatrix transposed) {
	   	CudaFloatMatrix mat = (CudaFloatMatrix)matrix;
    	CudaFloatMatrix result = (CudaFloatMatrix)transposed;
        CORE.transpose(mat.dataPointer, result.dataPointer, mat.rows, mat.columns);
        return transposed;
    }

	
	// ---------------------------------- inplace methods -------------------------------------
	
	@Override
    public FloatMatrix setOne() {
        CORE.execute(SET_ONE.getProgramName(), this.rows, this.columns, dataPointer);
        return this;
    }

	@Override
    public FloatMatrix setZero() {
        CORE.execute(SET_ZERO.getProgramName(), this.rows, this.columns, dataPointer);
        return this;
    }

	@Override
    public FloatMatrix randi() {
        CORE.rand(dataPointer, length);
        return this;
    }

	@Override
    public FloatMatrix randni() {
        CORE.randn(dataPointer, length);
        return this;
    }
	
	
    // --------------------------------------- add methods ----------------------------------------
    
    @Override
    public FloatMatrix add(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	CudaFloatMatrix a = (CudaFloatMatrix)matrixA;
    	CudaFloatMatrix b = (CudaFloatMatrix)matrixB;
    	CudaFloatMatrix r = (CudaFloatMatrix)result;
        checkSameSize(a, b, r);
        CORE.execute(ADD_MATRIX.getProgramName(), r.rows, r.columns, r.dataPointer, a.dataPointer, b.dataPointer);
      	return result;
    }

    @Override
    public FloatMatrix add(FloatMatrix matrix, float scalar, FloatMatrix result) {
		CudaFloatMatrix a = (CudaFloatMatrix)matrix;
		CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	CudaFloatMatrix r = (CudaFloatMatrix)result;
    	CORE.execute(ADD_SCALAR.getProgramName(), r.rows, r.columns, r.dataPointer, a.dataPointer, b.dataPointer);
        b.free();
    	return result;
    }

    public static void addColumnVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        CORE.execute(ADD_C_VECTOR.getProgramName(), result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
    }

    public static void addRowVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(ADD_R_VECTOR.getProgramName(), result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
    }
    	
    
    
    // --------------------------------------- sub methods ----------------------------------------
    
    @Override
    public FloatMatrix sub(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	CudaFloatMatrix a = (CudaFloatMatrix)matrixA;
    	CudaFloatMatrix b = (CudaFloatMatrix)matrixB;
    	CudaFloatMatrix r = (CudaFloatMatrix)result;
        checkSameSize(a, b, r);
        CORE.execute(SUB_MATRIX.getProgramName(), r.rows, r.columns, r.dataPointer, a.dataPointer, b.dataPointer);
      	return result;
    }

    @Override
    public FloatMatrix sub(FloatMatrix matrix, float scalar, FloatMatrix result) {
		CudaFloatMatrix a = (CudaFloatMatrix)matrix;
		CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	CudaFloatMatrix r = (CudaFloatMatrix)result;
    	CORE.execute(SUB_SCALAR.getProgramName(), r.rows, r.columns, r.dataPointer, a.dataPointer, b.dataPointer);
        b.free();
    	return result;
    }

    public static void subColumnVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        CORE.execute(SUB_C_VECTOR.getProgramName(), result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
    }

    public static void subRowVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(SUB_R_VECTOR.getProgramName(), result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
    }
    
    public static void rsub(CudaFloatMatrix a, float x, CudaFloatMatrix result) {
        CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { x });

        CORE.execute(RSUB_SCALAR.getProgramName(), result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
        b.free();
    }

    public static void rsubColumnVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        CORE.execute(RSUB_C_VECTOR.getProgramName(), result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }

    public static void rsubRowVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(RSUB_R_VECTOR.getProgramName(), result.rows, result.columns, result.dataPointer,
                a.dataPointer, b.dataPointer);
    }
    
    
    
    // --------------------------------------- mul methods ----------------------------------------
    
    @Override
    public FloatMatrix mul(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	CudaFloatMatrix a = (CudaFloatMatrix)matrixA;
    	CudaFloatMatrix b = (CudaFloatMatrix)matrixB;
    	CudaFloatMatrix r = (CudaFloatMatrix)result;
        checkSameSize(a, b, r);
        CORE.execute(MUL_MATRIX.getProgramName(), r.rows, r.columns, r.dataPointer, a.dataPointer, b.dataPointer);
      	return result;
    }

    @Override
    public FloatMatrix mul(FloatMatrix matrix, float scalar, FloatMatrix result) {
		CudaFloatMatrix a = (CudaFloatMatrix)matrix;
		CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	CudaFloatMatrix r = (CudaFloatMatrix)result;
    	CORE.execute(MUL_SCALAR.getProgramName(), r.rows, r.columns, r.dataPointer, a.dataPointer, b.dataPointer);
        b.free();
    	return result;
    }

    public static void mulColumnVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        CORE.execute(MUL_C_VECTOR.getProgramName(), result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
    }

    public static void mulRowVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(MUL_R_VECTOR.getProgramName(), result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
    }
    
    
    
    // --------------------------------------- div methods ----------------------------------------
    
    @Override
    public FloatMatrix div(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	CudaFloatMatrix a = (CudaFloatMatrix)matrixA;
    	CudaFloatMatrix b = (CudaFloatMatrix)matrixB;
    	CudaFloatMatrix r = (CudaFloatMatrix)result;
        checkSameSize(a, b, r);
        CORE.execute(DIV_MATRIX.getProgramName(), r.rows, r.columns, r.dataPointer, a.dataPointer, b.dataPointer);
      	return result;
    }

    @Override
    public FloatMatrix div(FloatMatrix matrix, float scalar, FloatMatrix result) {
		CudaFloatMatrix a = (CudaFloatMatrix)matrix;
		CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { scalar });
    	CudaFloatMatrix r = (CudaFloatMatrix)result;
    	CORE.execute(DIV_SCALAR.getProgramName(), r.rows, r.columns, r.dataPointer, a.dataPointer, b.dataPointer);
        b.free();
    	return result;
    }

    public static void divColumnVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        CORE.execute(DIV_C_VECTOR.getProgramName(), result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
    }

    public static void divRowVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(DIV_R_VECTOR.getProgramName(), result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
    }
	
    public static void rdiv(CudaFloatMatrix a, float x, CudaFloatMatrix result) {
        CudaFloatMatrix b = new CudaFloatMatrix(1, 1, new float[] { x });

        CORE.execute(RDIV_SCALAR.getProgramName(), result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
        b.free();
    }

    public static void rdivColumnVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkColumnVectorSize(a, b, result);
        CORE.execute(RDIV_C_VECTOR.getProgramName(), result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
    }

    public static void rdivRowVector(CudaFloatMatrix a, CudaFloatMatrix b, CudaFloatMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(RDIV_R_VECTOR.getProgramName(), result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
    }
    
    
    // --------------------------------------- mathematical functions ----------------------------------------
	
    @Override
    public FloatMatrix exp(FloatMatrix matrix, FloatMatrix result) {
    	throw new NotImplementedException();
	}

    @Override
    public FloatMatrix neg(FloatMatrix matrix, FloatMatrix result) {
    	throw new NotImplementedException();
	}
	
    @Override
	public FloatMatrix sigmoid(FloatMatrix matrix, FloatMatrix result) {
    	throw new NotImplementedException();
	}
    
    // --------------------------------------- greater than ----------------------------------------
    
    @Override
    public FloatMatrix gt(FloatMatrix matrixA, FloatMatrix matrixB, FloatMatrix result) {
    	throw new NotImplementedException();
    }
    
    @Override
    public FloatMatrix gt(FloatMatrix matrix, float scalar, FloatMatrix result) {
    	throw new NotImplementedException();
    }
    
    
    
    
    // --------------------------------------- matrix multiplication ----------------------------------------    

    @Override
    public FloatMatrix mmul(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
    	CudaFloatMatrix matrixA = (CudaFloatMatrix) a;
    	CudaFloatMatrix matrixB = (CudaFloatMatrix) b;
    	CudaFloatMatrix matrixR = (CudaFloatMatrix) result;
    	CORE.mmul(matrixA.dataPointer, matrixA.rows, matrixA.columns, matrixB.dataPointer, matrixB.columns, matrixR.dataPointer);
        return result;
    }

    @Override
    public FloatMatrix mmulTN(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
    	CudaFloatMatrix matrixA = (CudaFloatMatrix) a;
    	CudaFloatMatrix matrixB = (CudaFloatMatrix) b;
    	CudaFloatMatrix matrixR = (CudaFloatMatrix) result;
    	CORE.mmulTransposeA(matrixA.dataPointer, matrixA.rows, matrixA.columns, matrixB.dataPointer, matrixB.columns, matrixR.dataPointer);
        return result;
    }

    @Override
    public FloatMatrix mmulNT(FloatMatrix a, FloatMatrix b, FloatMatrix result) {
    	CudaFloatMatrix matrixA = (CudaFloatMatrix) a;
    	CudaFloatMatrix matrixB = (CudaFloatMatrix) b;
    	CudaFloatMatrix matrixR = (CudaFloatMatrix) result;
    	CORE.mmulTransposeB(matrixA.dataPointer, matrixA.rows, matrixA.columns, matrixB.dataPointer, matrixB.rows, matrixR.dataPointer);
        return result;
    }
    
    // --------------------------------------- reduction methods ----------------------------------------

    @Override
    public float sum(FloatMatrix matrix) {
    	CudaFloatMatrix mat = (CudaFloatMatrix) matrix;
        return CORE.reduce("sumFloats", mat.dataPointer, mat.length, 0);
    }

    public float mean(FloatMatrix matrix) {
    	CudaFloatMatrix mat = (CudaFloatMatrix) matrix;
        return sum(matrix) / mat.length;
    }

    public float prod(FloatMatrix matrix) {
    	CudaFloatMatrix mat = (CudaFloatMatrix) matrix;
        return CORE.reduce("productFloats", mat.dataPointer, mat.length, 1);
    }

    public float max(FloatMatrix matrix) {
    	CudaFloatMatrix mat = (CudaFloatMatrix) matrix;
        return CORE.reduce("maxFloats", mat.dataPointer, mat.length, Float.NEGATIVE_INFINITY);
    }

    public float min(FloatMatrix matrix) {
    	CudaFloatMatrix mat = (CudaFloatMatrix) matrix;
        return CORE.reduce("minFloats", mat.dataPointer, mat.length, Float.POSITIVE_INFINITY);
    }


    
    // --------------------------------------- row reduction methods ----------------------------------------

    public static void rowSums(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceRows("rowSumsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, 0);
    }

    public static void rowMeans(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        rowSums(matrix, result);
        result.div(result, matrix.columns, result);
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

    

    // --------------------------------------- column reduction methods ----------------------------------------

    public static void columnSums(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        CORE.reduceColumns("columnSumsFloats", matrix.dataPointer, result.dataPointer, matrix.rows, matrix.columns, 0);
    }

    public static void columnMeans(CudaFloatMatrix matrix, CudaFloatMatrix result) {
        columnSums(matrix, result);
        result.div(result, matrix.rows, result);
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
    
    
    // --------------------------------------- getter and setter methods ----------------------------------------

    public FloatMatrix getSubMatrix(FloatMatrix source, FloatMatrix destination, int rowOffset, int columnOffset) {
    	CudaFloatMatrix src = (CudaFloatMatrix) source;
    	CudaFloatMatrix dst = (CudaFloatMatrix) destination;    	
        CORE.getSubMatrix(src.dataPointer, dst.dataPointer, dst.rows, dst.columns, src.rows, rowOffset, columnOffset);
        return destination;
    }
    
    public FloatMatrix setSubMatrix(FloatMatrix source, FloatMatrix destination, int offsetRow, int offsetColumn) {
    	CudaFloatMatrix src = (CudaFloatMatrix) source;
    	CudaFloatMatrix dst = (CudaFloatMatrix) destination;   
        CORE.setSubMatrix(dst.dataPointer, src.dataPointer, src.rows, src.rows, dst.rows, offsetRow, offsetColumn);
        return destination;
    }
}
