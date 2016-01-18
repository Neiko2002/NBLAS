package org.nblas.cuda.blas;

import org.nblas.cuda.CudaBLASBase;
import org.nblas.cuda.CudaMatrix;
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
import org.nblas.generic.Subprogram;

import jcuda.driver.CUfunction;

/**
 * Level 1 BLAS operations typically take linear time, O(n)
 * 
 * @author Nico
 *
 */
public class CudaLevel1 extends CudaBLASBase {

	private final Subprogram<CUfunction> ADD_MATRIX;
    private final Subprogram<CUfunction> ADD_SCALAR;
    private final Subprogram<CUfunction> ADD_C_VECTOR;
    private final Subprogram<CUfunction> ADD_R_VECTOR;
	
    private final Subprogram<CUfunction> MUL_MATRIX;
    private final Subprogram<CUfunction> MUL_SCALAR;
    private final Subprogram<CUfunction> MUL_C_VECTOR;
    private final Subprogram<CUfunction> MUL_R_VECTOR;

    private final Subprogram<CUfunction> SUB_MATRIX;
    private final Subprogram<CUfunction> SUB_SCALAR;
    private final Subprogram<CUfunction> SUB_C_VECTOR;
    private final Subprogram<CUfunction> SUB_R_VECTOR;

    private final Subprogram<CUfunction> RSUB_SCALAR;
    private final Subprogram<CUfunction> RSUB_C_VECTOR;
    private final Subprogram<CUfunction> RSUB_R_VECTOR;

    private final Subprogram<CUfunction> DIV_MATRIX;
    private final Subprogram<CUfunction> DIV_SCALAR;
    private final Subprogram<CUfunction> DIV_C_VECTOR;
    private final Subprogram<CUfunction> DIV_R_VECTOR;

    private final Subprogram<CUfunction> RDIV_SCALAR;
    private final Subprogram<CUfunction> RDIV_C_VECTOR;
    private final Subprogram<CUfunction> RDIV_R_VECTOR;
    
    // greater than
    private final Subprogram<CUfunction> GT_MATRIX;
    private final Subprogram<CUfunction> GT_SCALAR;
    private final Subprogram<CUfunction> GT_C_VECTOR;
    private final Subprogram<CUfunction> GT_R_VECTOR;
    
    // greater than or equal
    private final Subprogram<CUfunction> GE_MATRIX;
    private final Subprogram<CUfunction> GE_SCALAR;
    private final Subprogram<CUfunction> GE_C_VECTOR;
    private final Subprogram<CUfunction> GE_R_VECTOR;
    
    // lower than
    private final Subprogram<CUfunction> LT_MATRIX;
    private final Subprogram<CUfunction> LT_SCALAR;
    private final Subprogram<CUfunction> LT_C_VECTOR;
    private final Subprogram<CUfunction> LT_R_VECTOR;
    
    // lower than or equal
    private final Subprogram<CUfunction> LE_MATRIX;
    private final Subprogram<CUfunction> LE_SCALAR;
    private final Subprogram<CUfunction> LE_C_VECTOR;
    private final Subprogram<CUfunction> LE_R_VECTOR;
    
    // equal
    private final Subprogram<CUfunction> EQ_MATRIX;
    private final Subprogram<CUfunction> EQ_SCALAR;
    private final Subprogram<CUfunction> EQ_C_VECTOR;
    private final Subprogram<CUfunction> EQ_R_VECTOR;
    
    // not equal
    private final Subprogram<CUfunction> NE_MATRIX;
    private final Subprogram<CUfunction> NE_SCALAR;
    private final Subprogram<CUfunction> NE_C_VECTOR;
    private final Subprogram<CUfunction> NE_R_VECTOR;
    
    private final Subprogram<CUfunction> SET_ONE;
    private final Subprogram<CUfunction> SET_ZERO;
    private final Subprogram<CUfunction> DUP;   

    // special functions  
    private final Subprogram<CUfunction> EXP;
    private final Subprogram<CUfunction> NEG;
    private final Subprogram<CUfunction> SIGMOID;
    
    public CudaLevel1(AFunctionBuilder<CUfunction> builder) {
		super(builder);
		
        // add Functions
        AFunctionObject add = new Add(new Arg(0), new Arg(1));
        ADD_MATRIX = buildPredefinedFunction("add", add, ArgumentType.MATRIX, ArgumentType.MATRIX);
        ADD_SCALAR = buildPredefinedFunction("add", add, ArgumentType.MATRIX, ArgumentType.SCALAR);
        ADD_R_VECTOR = buildPredefinedFunction("add", add, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        ADD_C_VECTOR = buildPredefinedFunction("add", add, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);
        
        AFunctionObject mul = new Mul(new Arg(0), new Arg(1));
        MUL_MATRIX = buildPredefinedFunction("mul", mul, ArgumentType.MATRIX, ArgumentType.MATRIX);
        MUL_SCALAR = buildPredefinedFunction("mul", mul, ArgumentType.MATRIX, ArgumentType.SCALAR);
        MUL_R_VECTOR = buildPredefinedFunction("mul", mul, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        MUL_C_VECTOR = buildPredefinedFunction("mul", mul, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);

        AFunctionObject sub = new Sub(new Arg(0), new Arg(1));
        SUB_MATRIX = buildPredefinedFunction("sub", sub, ArgumentType.MATRIX, ArgumentType.MATRIX);
        SUB_SCALAR = buildPredefinedFunction("sub", sub, ArgumentType.MATRIX, ArgumentType.SCALAR);
        SUB_R_VECTOR = buildPredefinedFunction("sub", sub, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        SUB_C_VECTOR = buildPredefinedFunction("sub", sub, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);

        AFunctionObject rsub = new Sub(new Arg(1), new Arg(0));
        RSUB_SCALAR = buildPredefinedFunction("rsub", rsub, ArgumentType.MATRIX, ArgumentType.SCALAR);
        RSUB_R_VECTOR = buildPredefinedFunction("rsub", rsub, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        RSUB_C_VECTOR = buildPredefinedFunction("rsub", rsub, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);

        AFunctionObject div = new Div(new Arg(0), new Arg(1));
        DIV_MATRIX = buildPredefinedFunction("div", div, ArgumentType.MATRIX, ArgumentType.MATRIX);
        DIV_SCALAR = buildPredefinedFunction("div", div, ArgumentType.MATRIX, ArgumentType.SCALAR);
        DIV_R_VECTOR = buildPredefinedFunction("div", div, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        DIV_C_VECTOR = buildPredefinedFunction("div", div, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);

        AFunctionObject rdiv = new Div(new Arg(1), new Arg(0));
        RDIV_SCALAR = buildPredefinedFunction("rdiv", rdiv, ArgumentType.MATRIX, ArgumentType.SCALAR);
        RDIV_R_VECTOR = buildPredefinedFunction("rdiv", rdiv, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        RDIV_C_VECTOR = buildPredefinedFunction("rdiv", rdiv, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);
        
        AFunctionObject greaterThan = new Comparator(">", new Arg(0), new Arg(1));
        GT_MATRIX = buildPredefinedFunction("gt", greaterThan, ArgumentType.MATRIX, ArgumentType.MATRIX);
        GT_SCALAR = buildPredefinedFunction("gt", greaterThan, ArgumentType.MATRIX, ArgumentType.SCALAR);
        GT_R_VECTOR = buildPredefinedFunction("gt", greaterThan, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        GT_C_VECTOR = buildPredefinedFunction("gt", greaterThan, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);                
        
        AFunctionObject greaterEqual = new Comparator(">=", new Arg(0), new Arg(1));
        GE_MATRIX = buildPredefinedFunction("ge", greaterEqual, ArgumentType.MATRIX, ArgumentType.MATRIX);
        GE_SCALAR = buildPredefinedFunction("ge", greaterEqual, ArgumentType.MATRIX, ArgumentType.SCALAR);
        GE_R_VECTOR = buildPredefinedFunction("ge", greaterEqual, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        GE_C_VECTOR = buildPredefinedFunction("ge", greaterEqual, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);
        
        AFunctionObject lowerThan = new Comparator("<", new Arg(0), new Arg(1));
        LT_MATRIX = buildPredefinedFunction("lt", lowerThan, ArgumentType.MATRIX, ArgumentType.MATRIX);
        LT_SCALAR = buildPredefinedFunction("lt", lowerThan, ArgumentType.MATRIX, ArgumentType.SCALAR);
        LT_R_VECTOR = buildPredefinedFunction("lt", lowerThan, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        LT_C_VECTOR = buildPredefinedFunction("lt", lowerThan, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);                
        
        AFunctionObject lowerEqual = new Comparator("<=", new Arg(0), new Arg(1));
        LE_MATRIX = buildPredefinedFunction("le", lowerEqual, ArgumentType.MATRIX, ArgumentType.MATRIX);
        LE_SCALAR = buildPredefinedFunction("le", lowerEqual, ArgumentType.MATRIX, ArgumentType.SCALAR);
        LE_R_VECTOR = buildPredefinedFunction("le", lowerEqual, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        LE_C_VECTOR = buildPredefinedFunction("le", lowerEqual, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);
        
        AFunctionObject equal = new Comparator("==", new Arg(0), new Arg(1));
        EQ_MATRIX = buildPredefinedFunction("eq", equal, ArgumentType.MATRIX, ArgumentType.MATRIX);
        EQ_SCALAR = buildPredefinedFunction("eq", equal, ArgumentType.MATRIX, ArgumentType.SCALAR);
        EQ_R_VECTOR = buildPredefinedFunction("eq", equal, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        EQ_C_VECTOR = buildPredefinedFunction("eq", equal, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);                
        
        AFunctionObject notEqual = new Comparator("!=", new Arg(0), new Arg(1));
        NE_MATRIX = buildPredefinedFunction("ne", notEqual, ArgumentType.MATRIX, ArgumentType.MATRIX);
        NE_SCALAR = buildPredefinedFunction("ne", notEqual, ArgumentType.MATRIX, ArgumentType.SCALAR);
        NE_R_VECTOR = buildPredefinedFunction("ne", notEqual, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        NE_C_VECTOR = buildPredefinedFunction("ne", notEqual, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);
        
        AFunctionObject one = new Value(1.0);
        SET_ONE = buildPredefinedFunction("one", one);
       
        AFunctionObject zero = new Value(0.0);
        SET_ZERO = buildPredefinedFunction("zero", zero);
        
        AFunctionObject exp = new Exp(new Arg(0));
        EXP = buildPredefinedFunction("epx", exp, ArgumentType.MATRIX);
          
        AFunctionObject negate = new Negate(new Arg(0));
        NEG = buildPredefinedFunction("neg", negate, ArgumentType.MATRIX);
        
        AFunctionObject sigmoid = MatrixFunctions.sigmoid(new Arg(0));
        SIGMOID = buildPredefinedFunction("sigmoid", sigmoid, ArgumentType.MATRIX);
        
        AFunctionObject copy = new Arg(0);
        DUP = buildPredefinedFunction("dup", copy, ArgumentType.MATRIX);
    }
    
    
    // ADD

    public void add(CudaMatrix matrixA, CudaMatrix matrixB, CudaMatrix result) {
    	runMatrixMatrixElementWiseOperation(ADD_MATRIX, matrixA, matrixB, result);
    }

    public void addScalar(CudaMatrix matrix, CudaMatrix scalar, CudaMatrix result) {
    	runMatrixScalarElementWiseOperation(ADD_SCALAR, matrix, scalar, result);
    }

    public void addColumnVector(CudaMatrix matrix, CudaMatrix columnVector, CudaMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(ADD_C_VECTOR, matrix, columnVector, result);
    }

    public void addRowVector(CudaMatrix matrix, CudaMatrix rowVector, CudaMatrix result) {
    	runMatrixRowVectorElementWiseOperation(ADD_R_VECTOR, matrix, rowVector, result);
    }
    
    
    // MUL

    public void mul(CudaMatrix matrixA, CudaMatrix matrixB, CudaMatrix result) {
    	runMatrixMatrixElementWiseOperation(MUL_MATRIX, matrixA, matrixB, result);
    }

    public void mulScalar(CudaMatrix matrix, CudaMatrix scalar, CudaMatrix result) {
    	runMatrixScalarElementWiseOperation(MUL_SCALAR, matrix, scalar, result);
    }

    public void mulColumnVector(CudaMatrix matrix, CudaMatrix columnVector, CudaMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(MUL_C_VECTOR, matrix, columnVector, result);
    }

    public void mulRowVector(CudaMatrix matrix, CudaMatrix rowVector, CudaMatrix result) {
    	runMatrixRowVectorElementWiseOperation(MUL_R_VECTOR, matrix, rowVector, result);
    }


    
    // SUB

    public void sub(CudaMatrix matrixA, CudaMatrix matrixB, CudaMatrix result) {
    	runMatrixMatrixElementWiseOperation(SUB_MATRIX, matrixA, matrixB, result);
    }

    public void subScalar(CudaMatrix matrix, CudaMatrix scalar, CudaMatrix result) {
    	runMatrixScalarElementWiseOperation(SUB_SCALAR, matrix, scalar, result);
    }

    public void subColumnVector(CudaMatrix matrix, CudaMatrix columnVector, CudaMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(SUB_C_VECTOR, matrix, columnVector, result);
    }

    public void subRowVector(CudaMatrix matrix, CudaMatrix rowVector, CudaMatrix result) {
    	runMatrixRowVectorElementWiseOperation(SUB_R_VECTOR, matrix, rowVector, result);
    }

    public void rsubScalar(CudaMatrix matrix, CudaMatrix scalar, CudaMatrix result) {
      	runMatrixScalarElementWiseOperation(RSUB_SCALAR, matrix, scalar, result);
    }

    public void rsubColumnVector(CudaMatrix matrix, CudaMatrix columnVector, CudaMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(RSUB_C_VECTOR, matrix, columnVector, result);
    }

    public void rsubRowVector(CudaMatrix matrix, CudaMatrix rowVector, CudaMatrix result) {
    	runMatrixRowVectorElementWiseOperation(RSUB_R_VECTOR, matrix, rowVector, result);
    }


    // DIV

    public void div(CudaMatrix matrixA, CudaMatrix matrixB, CudaMatrix result) {
    	runMatrixMatrixElementWiseOperation(DIV_MATRIX, matrixA, matrixB, result);
    }

    public void divScalar(CudaMatrix matrix, CudaMatrix scalar, CudaMatrix result) {
		runMatrixScalarElementWiseOperation(DIV_SCALAR, matrix, scalar, result);
    }

    public void divColumnVector(CudaMatrix matrix, CudaMatrix columnVector, CudaMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(DIV_C_VECTOR, matrix, columnVector, result);
    }

    public void divRowVector(CudaMatrix matrix, CudaMatrix rowVector, CudaMatrix result) {
    	runMatrixRowVectorElementWiseOperation(DIV_R_VECTOR, matrix, rowVector, result);
    }

    public void rdivScalar(CudaMatrix matrix, CudaMatrix scalar, CudaMatrix result) {
		runMatrixScalarElementWiseOperation(RDIV_SCALAR, matrix, scalar, result);
    }

    public void rdivColumnVector(CudaMatrix matrix, CudaMatrix columnVector, CudaMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(RDIV_C_VECTOR, matrix, columnVector, result);
    }

    public void rdivRowVector(CudaMatrix matrix, CudaMatrix rowVector, CudaMatrix result) {
    	runMatrixRowVectorElementWiseOperation(RDIV_R_VECTOR, matrix, rowVector, result);
    }
    
    
    // GREATER THAN
    
    public void gtScalar(CudaMatrix matrix, CudaMatrix scalar, CudaMatrix result) {
		runMatrixScalarElementWiseOperation(GT_SCALAR, matrix, scalar, result);
    }

    public void gt(CudaMatrix matrixA, CudaMatrix matrixB, CudaMatrix result) {
    	runMatrixMatrixElementWiseOperation(GT_MATRIX, matrixA, matrixB, result);
    }

    public void gtColumnVector(CudaMatrix matrix, CudaMatrix columnVector, CudaMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(GT_C_VECTOR, matrix, columnVector, result);
    }

    public void gtRowVector(CudaMatrix matrix, CudaMatrix rowVector, CudaMatrix result) {
    	runMatrixRowVectorElementWiseOperation(GT_R_VECTOR, matrix, rowVector, result);
    }   
    
    
    
    // GREATER THAN OR EQUAL
    
    public void geScalar(CudaMatrix matrix, CudaMatrix scalar, CudaMatrix result) {
		runMatrixScalarElementWiseOperation(GE_SCALAR, matrix, scalar, result);
    }

    public void ge(CudaMatrix matrixA, CudaMatrix matrixB, CudaMatrix result) {
    	runMatrixMatrixElementWiseOperation(GE_MATRIX, matrixA, matrixB, result);
    }

    public void geColumnVector(CudaMatrix matrix, CudaMatrix columnVector, CudaMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(GE_C_VECTOR, matrix, columnVector, result);
    }

    public void geRowVector(CudaMatrix matrix, CudaMatrix rowVector, CudaMatrix result) {
    	runMatrixRowVectorElementWiseOperation(GE_R_VECTOR, matrix, rowVector, result);
    }   
    
    
    
    // LOWER THAN
    
    public void ltScalar(CudaMatrix matrix, CudaMatrix scalar, CudaMatrix result) {
		runMatrixScalarElementWiseOperation(LT_SCALAR, matrix, scalar, result);
    }

    public void lt(CudaMatrix matrixA, CudaMatrix matrixB, CudaMatrix result) {
    	runMatrixMatrixElementWiseOperation(LT_MATRIX, matrixA, matrixB, result);
    }

    public void ltColumnVector(CudaMatrix matrix, CudaMatrix columnVector, CudaMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(LT_C_VECTOR, matrix, columnVector, result);
    }

    public void ltRowVector(CudaMatrix matrix, CudaMatrix rowVector, CudaMatrix result) {
    	runMatrixRowVectorElementWiseOperation(LT_R_VECTOR, matrix, rowVector, result);
    }   
    
    
    
    // LOWER THAN OR EQUAL
    
    public void leScalar(CudaMatrix matrix, CudaMatrix scalar, CudaMatrix result) {
		runMatrixScalarElementWiseOperation(LE_SCALAR, matrix, scalar, result);
    }

    public void le(CudaMatrix matrixA, CudaMatrix matrixB, CudaMatrix result) {
    	runMatrixMatrixElementWiseOperation(LE_MATRIX, matrixA, matrixB, result);
    }

    public void leColumnVector(CudaMatrix matrix, CudaMatrix columnVector, CudaMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(LE_C_VECTOR, matrix, columnVector, result);
    }

    public void leRowVector(CudaMatrix matrix, CudaMatrix rowVector, CudaMatrix result) {
    	runMatrixRowVectorElementWiseOperation(LE_R_VECTOR, matrix, rowVector, result);
    } 
    
    
    
    // EQUAL
    
    public void eqScalar(CudaMatrix matrix, CudaMatrix scalar, CudaMatrix result) {
		runMatrixScalarElementWiseOperation(EQ_SCALAR, matrix, scalar, result);
    }

    public void eq(CudaMatrix matrixA, CudaMatrix matrixB, CudaMatrix result) {
    	runMatrixMatrixElementWiseOperation(EQ_MATRIX, matrixA, matrixB, result);
    }

    public void eqColumnVector(CudaMatrix matrix, CudaMatrix columnVector, CudaMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(EQ_C_VECTOR, matrix, columnVector, result);
    }

    public void eqRowVector(CudaMatrix matrix, CudaMatrix rowVector, CudaMatrix result) {
    	runMatrixRowVectorElementWiseOperation(EQ_R_VECTOR, matrix, rowVector, result);
    }   
    
    
    
    // NOT EQUAL
    
    public void neScalar(CudaMatrix matrix, CudaMatrix scalar, CudaMatrix result) {
		runMatrixScalarElementWiseOperation(NE_SCALAR, matrix, scalar, result);
    }

    public void ne(CudaMatrix matrixA, CudaMatrix matrixB, CudaMatrix result) {
    	runMatrixMatrixElementWiseOperation(NE_MATRIX, matrixA, matrixB, result);
    }

    public void neColumnVector(CudaMatrix matrix, CudaMatrix columnVector, CudaMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(NE_C_VECTOR, matrix, columnVector, result);
    }

    public void neRowVector(CudaMatrix matrix, CudaMatrix rowVector, CudaMatrix result) {
    	runMatrixRowVectorElementWiseOperation(NE_R_VECTOR, matrix, rowVector, result);
    } 
    
    public void setOne(CudaMatrix matrix) {
        runMatrixOperation(SET_ONE, matrix);
    }
    
    public void setZero(CudaMatrix matrix) {
        runMatrixOperation(SET_ZERO, matrix);
    }
    
    public void dup(CudaMatrix matrix, CudaMatrix result) {
		runMatrixElementWiseOperation(DUP, matrix, result);
	}
	
    public void exp(CudaMatrix matrix, CudaMatrix result) {
		runMatrixElementWiseOperation(EXP, matrix, result);
	}

    public void neg(CudaMatrix matrix, CudaMatrix result) {
		runMatrixElementWiseOperation(NEG, matrix, result);
	}
	
	public void sigmoid(CudaMatrix matrix, CudaMatrix result) {
		runMatrixElementWiseOperation(SIGMOID, matrix, result);
	}
}
