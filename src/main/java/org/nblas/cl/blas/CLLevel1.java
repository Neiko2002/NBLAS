package org.nblas.cl.blas;

import org.jocl.cl_kernel;
import org.nblas.cl.CLBLASBase;
import org.nblas.cl.CLMatrix;
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

/**
 * Level 1 BLAS operations typically take linear time, O(n)
 * 
 * TODO: jedes Level sollte sein eigenes cl_program haben. 
 * 
 * 
 * @author Nico
 *
 */
public class CLLevel1 extends CLBLASBase {

	private final Subprogram<cl_kernel> ADD_MATRIX;
    private final Subprogram<cl_kernel> ADD_SCALAR;
    private final Subprogram<cl_kernel> ADD_C_VECTOR;
    private final Subprogram<cl_kernel> ADD_R_VECTOR;
	
    private final Subprogram<cl_kernel> MUL_MATRIX;
    private final Subprogram<cl_kernel> MUL_SCALAR;
    private final Subprogram<cl_kernel> MUL_C_VECTOR;
    private final Subprogram<cl_kernel> MUL_R_VECTOR;

    private final Subprogram<cl_kernel> SUB_MATRIX;
    private final Subprogram<cl_kernel> SUB_SCALAR;
    private final Subprogram<cl_kernel> SUB_C_VECTOR;
    private final Subprogram<cl_kernel> SUB_R_VECTOR;

    private final Subprogram<cl_kernel> RSUB_SCALAR;
    private final Subprogram<cl_kernel> RSUB_C_VECTOR;
    private final Subprogram<cl_kernel> RSUB_R_VECTOR;

    private final Subprogram<cl_kernel> DIV_MATRIX;
    private final Subprogram<cl_kernel> DIV_SCALAR;
    private final Subprogram<cl_kernel> DIV_C_VECTOR;
    private final Subprogram<cl_kernel> DIV_R_VECTOR;

    private final Subprogram<cl_kernel> RDIV_SCALAR;
    private final Subprogram<cl_kernel> RDIV_C_VECTOR;
    private final Subprogram<cl_kernel> RDIV_R_VECTOR;
    
    // greater than
    private final Subprogram<cl_kernel> GT_MATRIX;
    private final Subprogram<cl_kernel> GT_SCALAR;
    private final Subprogram<cl_kernel> GT_C_VECTOR;
    private final Subprogram<cl_kernel> GT_R_VECTOR;
    
    // greater than or equal
    private final Subprogram<cl_kernel> GE_MATRIX;
    private final Subprogram<cl_kernel> GE_SCALAR;
    private final Subprogram<cl_kernel> GE_C_VECTOR;
    private final Subprogram<cl_kernel> GE_R_VECTOR;
    
    // lower than
    private final Subprogram<cl_kernel> LT_MATRIX;
    private final Subprogram<cl_kernel> LT_SCALAR;
    private final Subprogram<cl_kernel> LT_C_VECTOR;
    private final Subprogram<cl_kernel> LT_R_VECTOR;
    
    // lower than or equal
    private final Subprogram<cl_kernel> LE_MATRIX;
    private final Subprogram<cl_kernel> LE_SCALAR;
    private final Subprogram<cl_kernel> LE_C_VECTOR;
    private final Subprogram<cl_kernel> LE_R_VECTOR;
    
    // equal
    private final Subprogram<cl_kernel> EQ_MATRIX;
    private final Subprogram<cl_kernel> EQ_SCALAR;
    private final Subprogram<cl_kernel> EQ_C_VECTOR;
    private final Subprogram<cl_kernel> EQ_R_VECTOR;
    
    // not equal
    private final Subprogram<cl_kernel> NE_MATRIX;
    private final Subprogram<cl_kernel> NE_SCALAR;
    private final Subprogram<cl_kernel> NE_C_VECTOR;
    private final Subprogram<cl_kernel> NE_R_VECTOR;
    
    private final Subprogram<cl_kernel> SET_ONE;
    private final Subprogram<cl_kernel> DUP;   

    // special functions  
    private final Subprogram<cl_kernel> EXP;
    private final Subprogram<cl_kernel> NEG;
    private final Subprogram<cl_kernel> SIGMOID;
    
    public CLLevel1(AFunctionBuilder<cl_kernel> builder) {
		super(builder);
		
        // add Functions
        AFunctionObject add = new Add(new Arg(0), new Arg(1));
        ADD_MATRIX = buildPredefinedFunction(add, ArgumentType.MATRIX, ArgumentType.MATRIX);
        ADD_SCALAR = buildPredefinedFunction(add, ArgumentType.MATRIX, ArgumentType.SCALAR);
        ADD_R_VECTOR = buildPredefinedFunction(add, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        ADD_C_VECTOR = buildPredefinedFunction(add, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);
        
        AFunctionObject mul = new Mul(new Arg(0), new Arg(1));
        MUL_MATRIX = buildPredefinedFunction(mul, ArgumentType.MATRIX, ArgumentType.MATRIX);
        MUL_SCALAR = buildPredefinedFunction(mul, ArgumentType.MATRIX, ArgumentType.SCALAR);
        MUL_R_VECTOR = buildPredefinedFunction(mul, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        MUL_C_VECTOR = buildPredefinedFunction(mul, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);

        AFunctionObject sub = new Sub(new Arg(0), new Arg(1));
        SUB_MATRIX = buildPredefinedFunction(sub, ArgumentType.MATRIX, ArgumentType.MATRIX);
        SUB_SCALAR = buildPredefinedFunction(sub, ArgumentType.MATRIX, ArgumentType.SCALAR);
        SUB_R_VECTOR = buildPredefinedFunction(sub, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        SUB_C_VECTOR = buildPredefinedFunction(sub, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);

        AFunctionObject rsub = new Sub(new Arg(1), new Arg(0));
        RSUB_SCALAR = buildPredefinedFunction(rsub, ArgumentType.MATRIX, ArgumentType.SCALAR);
        RSUB_R_VECTOR = buildPredefinedFunction(rsub, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        RSUB_C_VECTOR = buildPredefinedFunction(rsub, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);

        AFunctionObject div = new Div(new Arg(0), new Arg(1));
        DIV_MATRIX = buildPredefinedFunction(div, ArgumentType.MATRIX, ArgumentType.MATRIX);
        DIV_SCALAR = buildPredefinedFunction(div, ArgumentType.MATRIX, ArgumentType.SCALAR);
        DIV_R_VECTOR = buildPredefinedFunction(div, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        DIV_C_VECTOR = buildPredefinedFunction(div, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);

        AFunctionObject rdiv = new Div(new Arg(1), new Arg(0));
        RDIV_SCALAR = buildPredefinedFunction(rdiv, ArgumentType.MATRIX, ArgumentType.SCALAR);
        RDIV_R_VECTOR = buildPredefinedFunction(rdiv, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        RDIV_C_VECTOR = buildPredefinedFunction(rdiv, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);
        
        AFunctionObject greaterThan = new Comparator(">", new Arg(0), new Arg(1));
        GT_MATRIX = buildPredefinedFunction(greaterThan, ArgumentType.MATRIX, ArgumentType.MATRIX);
        GT_SCALAR = buildPredefinedFunction(greaterThan, ArgumentType.MATRIX, ArgumentType.SCALAR);
        GT_R_VECTOR = buildPredefinedFunction(greaterThan, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        GT_C_VECTOR = buildPredefinedFunction(greaterThan, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);                
        
        AFunctionObject greaterEqual = new Comparator(">=", new Arg(0), new Arg(1));
        GE_MATRIX = buildPredefinedFunction(greaterEqual, ArgumentType.MATRIX, ArgumentType.MATRIX);
        GE_SCALAR = buildPredefinedFunction(greaterEqual, ArgumentType.MATRIX, ArgumentType.SCALAR);
        GE_R_VECTOR = buildPredefinedFunction(greaterEqual, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        GE_C_VECTOR = buildPredefinedFunction(greaterEqual, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);
        
        AFunctionObject lowerThan = new Comparator("<", new Arg(0), new Arg(1));
        LT_MATRIX = buildPredefinedFunction(lowerThan, ArgumentType.MATRIX, ArgumentType.MATRIX);
        LT_SCALAR = buildPredefinedFunction(lowerThan, ArgumentType.MATRIX, ArgumentType.SCALAR);
        LT_R_VECTOR = buildPredefinedFunction(lowerThan, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        LT_C_VECTOR = buildPredefinedFunction(lowerThan, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);                
        
        AFunctionObject lowerEqual = new Comparator("<=", new Arg(0), new Arg(1));
        LE_MATRIX = buildPredefinedFunction(lowerEqual, ArgumentType.MATRIX, ArgumentType.MATRIX);
        LE_SCALAR = buildPredefinedFunction(lowerEqual, ArgumentType.MATRIX, ArgumentType.SCALAR);
        LE_R_VECTOR = buildPredefinedFunction(lowerEqual, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        LE_C_VECTOR = buildPredefinedFunction(lowerEqual, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);
        
        AFunctionObject equal = new Comparator("==", new Arg(0), new Arg(1));
        EQ_MATRIX = buildPredefinedFunction(equal, ArgumentType.MATRIX, ArgumentType.MATRIX);
        EQ_SCALAR = buildPredefinedFunction(equal, ArgumentType.MATRIX, ArgumentType.SCALAR);
        EQ_R_VECTOR = buildPredefinedFunction(equal, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        EQ_C_VECTOR = buildPredefinedFunction(equal, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);                
        
        AFunctionObject notEqual = new Comparator("!=", new Arg(0), new Arg(1));
        NE_MATRIX = buildPredefinedFunction(notEqual, ArgumentType.MATRIX, ArgumentType.MATRIX);
        NE_SCALAR = buildPredefinedFunction(notEqual, ArgumentType.MATRIX, ArgumentType.SCALAR);
        NE_R_VECTOR = buildPredefinedFunction(notEqual, ArgumentType.MATRIX, ArgumentType.ROW_VECTOR);
        NE_C_VECTOR = buildPredefinedFunction(notEqual, ArgumentType.MATRIX, ArgumentType.COLUMN_VECTOR);
        
        AFunctionObject one = new Value(1.0);
        SET_ONE = buildPredefinedFunction(one);
        
        AFunctionObject exp = new Exp(new Arg(0));
        EXP = buildPredefinedFunction(exp, ArgumentType.MATRIX);
          
        AFunctionObject negate = new Negate(new Arg(0));
        NEG = buildPredefinedFunction(negate, ArgumentType.MATRIX);
        
        AFunctionObject sigmoid = MatrixFunctions.sigmoid(new Arg(0));
        SIGMOID = buildPredefinedFunction(sigmoid, ArgumentType.MATRIX);
        
        AFunctionObject copy = new Arg(0);
        DUP = buildPredefinedFunction(copy, ArgumentType.MATRIX);
    }
    
    
    // ADD

    public void add(CLMatrix matrixA, CLMatrix matrixB, CLMatrix result) {
    	runMatrixMatrixElementWiseOperation(ADD_MATRIX, matrixA, matrixB, result);
    }

    public void addScalar(CLMatrix matrix, CLMatrix scalar, CLMatrix result) {
    	runMatrixScalarElementWiseOperation(ADD_SCALAR, matrix, scalar, result);
    }

    public void addColumnVector(CLMatrix matrix, CLMatrix columnVector, CLMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(ADD_C_VECTOR, matrix, columnVector, result);
    }

    public void addRowVector(CLMatrix matrix, CLMatrix rowVector, CLMatrix result) {
    	runMatrixRowVectorElementWiseOperation(ADD_R_VECTOR, matrix, rowVector, result);
    }
    
    
    // MUL

    public void mul(CLMatrix matrixA, CLMatrix matrixB, CLMatrix result) {
    	runMatrixMatrixElementWiseOperation(MUL_MATRIX, matrixA, matrixB, result);
    }

    public void mulScalar(CLMatrix matrix, CLMatrix scalar, CLMatrix result) {
    	runMatrixScalarElementWiseOperation(MUL_SCALAR, matrix, scalar, result);
    }

    public void mulColumnVector(CLMatrix matrix, CLMatrix columnVector, CLMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(MUL_C_VECTOR, matrix, columnVector, result);
    }

    public void mulRowVector(CLMatrix matrix, CLMatrix rowVector, CLMatrix result) {
    	runMatrixRowVectorElementWiseOperation(MUL_R_VECTOR, matrix, rowVector, result);
    }


    
    // SUB

    public void sub(CLMatrix matrixA, CLMatrix matrixB, CLMatrix result) {
    	runMatrixMatrixElementWiseOperation(SUB_MATRIX, matrixA, matrixB, result);
    }

    public void subScalar(CLMatrix matrix, CLMatrix scalar, CLMatrix result) {
    	runMatrixScalarElementWiseOperation(SUB_SCALAR, matrix, scalar, result);
    }

    public void subColumnVector(CLMatrix matrix, CLMatrix columnVector, CLMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(SUB_C_VECTOR, matrix, columnVector, result);
    }

    public void subRowVector(CLMatrix matrix, CLMatrix rowVector, CLMatrix result) {
    	runMatrixRowVectorElementWiseOperation(SUB_R_VECTOR, matrix, rowVector, result);
    }

    public void rsubScalar(CLMatrix matrix, CLMatrix scalar, CLMatrix result) {
      	runMatrixScalarElementWiseOperation(RSUB_SCALAR, matrix, scalar, result);
    }

    public void rsubColumnVector(CLMatrix matrix, CLMatrix columnVector, CLMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(RSUB_C_VECTOR, matrix, columnVector, result);
    }

    public void rsubRowVector(CLMatrix matrix, CLMatrix rowVector, CLMatrix result) {
    	runMatrixRowVectorElementWiseOperation(RSUB_R_VECTOR, matrix, rowVector, result);
    }


    // DIV

    public void div(CLMatrix matrixA, CLMatrix matrixB, CLMatrix result) {
    	runMatrixMatrixElementWiseOperation(DIV_MATRIX, matrixA, matrixB, result);
    }

    public void divScalar(CLMatrix matrix, CLMatrix scalar, CLMatrix result) {
		runMatrixScalarElementWiseOperation(DIV_SCALAR, matrix, scalar, result);
    }

    public void divColumnVector(CLMatrix matrix, CLMatrix columnVector, CLMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(DIV_C_VECTOR, matrix, columnVector, result);
    }

    public void divRowVector(CLMatrix matrix, CLMatrix rowVector, CLMatrix result) {
    	runMatrixRowVectorElementWiseOperation(DIV_R_VECTOR, matrix, rowVector, result);
    }

    public void rdivScalar(CLMatrix matrix, CLMatrix scalar, CLMatrix result) {
		runMatrixScalarElementWiseOperation(RDIV_SCALAR, matrix, scalar, result);
    }

    public void rdivColumnVector(CLMatrix matrix, CLMatrix columnVector, CLMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(RDIV_C_VECTOR, matrix, columnVector, result);
    }

    public void rdivRowVector(CLMatrix matrix, CLMatrix rowVector, CLMatrix result) {
    	runMatrixRowVectorElementWiseOperation(RDIV_R_VECTOR, matrix, rowVector, result);
    }
    
    
    // GREATER THAN
    
    public void gtScalar(CLMatrix matrix, CLMatrix scalar, CLMatrix result) {
		runMatrixScalarElementWiseOperation(GT_SCALAR, matrix, scalar, result);
    }

    public void gt(CLMatrix matrixA, CLMatrix matrixB, CLMatrix result) {
    	runMatrixMatrixElementWiseOperation(GT_MATRIX, matrixA, matrixB, result);
    }

    public void gtColumnVector(CLMatrix matrix, CLMatrix columnVector, CLMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(GT_C_VECTOR, matrix, columnVector, result);
    }

    public void gtRowVector(CLMatrix matrix, CLMatrix rowVector, CLMatrix result) {
    	runMatrixRowVectorElementWiseOperation(GT_R_VECTOR, matrix, rowVector, result);
    }   
    
    
    
    // GREATER THAN OR EQUAL
    
    public void geScalar(CLMatrix matrix, CLMatrix scalar, CLMatrix result) {
		runMatrixScalarElementWiseOperation(GE_SCALAR, matrix, scalar, result);
    }

    public void ge(CLMatrix matrixA, CLMatrix matrixB, CLMatrix result) {
    	runMatrixMatrixElementWiseOperation(GE_MATRIX, matrixA, matrixB, result);
    }

    public void geColumnVector(CLMatrix matrix, CLMatrix columnVector, CLMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(GE_C_VECTOR, matrix, columnVector, result);
    }

    public void geRowVector(CLMatrix matrix, CLMatrix rowVector, CLMatrix result) {
    	runMatrixRowVectorElementWiseOperation(GE_R_VECTOR, matrix, rowVector, result);
    }   
    
    
    
    // LOWER THAN
    
    public void ltScalar(CLMatrix matrix, CLMatrix scalar, CLMatrix result) {
		runMatrixScalarElementWiseOperation(LT_SCALAR, matrix, scalar, result);
    }

    public void lt(CLMatrix matrixA, CLMatrix matrixB, CLMatrix result) {
    	runMatrixMatrixElementWiseOperation(LT_MATRIX, matrixA, matrixB, result);
    }

    public void ltColumnVector(CLMatrix matrix, CLMatrix columnVector, CLMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(LT_C_VECTOR, matrix, columnVector, result);
    }

    public void ltRowVector(CLMatrix matrix, CLMatrix rowVector, CLMatrix result) {
    	runMatrixRowVectorElementWiseOperation(LT_R_VECTOR, matrix, rowVector, result);
    }   
    
    
    
    // LOWER THAN OR EQUAL
    
    public void leScalar(CLMatrix matrix, CLMatrix scalar, CLMatrix result) {
		runMatrixScalarElementWiseOperation(LE_SCALAR, matrix, scalar, result);
    }

    public void le(CLMatrix matrixA, CLMatrix matrixB, CLMatrix result) {
    	runMatrixMatrixElementWiseOperation(LE_MATRIX, matrixA, matrixB, result);
    }

    public void leColumnVector(CLMatrix matrix, CLMatrix columnVector, CLMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(LE_C_VECTOR, matrix, columnVector, result);
    }

    public void leRowVector(CLMatrix matrix, CLMatrix rowVector, CLMatrix result) {
    	runMatrixRowVectorElementWiseOperation(LE_R_VECTOR, matrix, rowVector, result);
    } 
    
    
    
    // EQUAL
    
    public void eqScalar(CLMatrix matrix, CLMatrix scalar, CLMatrix result) {
		runMatrixScalarElementWiseOperation(EQ_SCALAR, matrix, scalar, result);
    }

    public void eq(CLMatrix matrixA, CLMatrix matrixB, CLMatrix result) {
    	runMatrixMatrixElementWiseOperation(EQ_MATRIX, matrixA, matrixB, result);
    }

    public void eqColumnVector(CLMatrix matrix, CLMatrix columnVector, CLMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(EQ_C_VECTOR, matrix, columnVector, result);
    }

    public void eqRowVector(CLMatrix matrix, CLMatrix rowVector, CLMatrix result) {
    	runMatrixRowVectorElementWiseOperation(EQ_R_VECTOR, matrix, rowVector, result);
    }   
    
    
    
    // NOT EQUAL
    
    public void neScalar(CLMatrix matrix, CLMatrix scalar, CLMatrix result) {
		runMatrixScalarElementWiseOperation(NE_SCALAR, matrix, scalar, result);
    }

    public void ne(CLMatrix matrixA, CLMatrix matrixB, CLMatrix result) {
    	runMatrixMatrixElementWiseOperation(NE_MATRIX, matrixA, matrixB, result);
    }

    public void neColumnVector(CLMatrix matrix, CLMatrix columnVector, CLMatrix result) {
    	runMatrixColumnVectorElementWiseOperation(NE_C_VECTOR, matrix, columnVector, result);
    }

    public void neRowVector(CLMatrix matrix, CLMatrix rowVector, CLMatrix result) {
    	runMatrixRowVectorElementWiseOperation(NE_R_VECTOR, matrix, rowVector, result);
    } 
    
    public void setOne(CLMatrix matrix) {
        runMatrixOperation(SET_ONE, matrix);
    }
    
    public void dup(CLMatrix matrix, CLMatrix result) {
		runMatrixElementWiseOperation(DUP, matrix, result);
	}
	
    public void exp(CLMatrix matrix, CLMatrix result) {
		runMatrixElementWiseOperation(EXP, matrix, result);
	}

    public void neg(CLMatrix matrix, CLMatrix result) {
		runMatrixElementWiseOperation(NEG, matrix, result);
	}
	
	public void sigmoid(CLMatrix matrix, CLMatrix result) {
		runMatrixElementWiseOperation(SIGMOID, matrix, result);
	}
}
