package org.nblas.cl.blas;

import org.jocl.cl_kernel;
import org.nblas.cl.CLBLASBase;
import org.nblas.cl.CLFloatMatrix;
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
 * TODO: jedes Level sollte sein eigenes CL Program haben und compilieren. 
 * Dadurch kann auch setup() weg und jedes level kann einselnd verwendet werden. 
 * 
 * @author Nico
 *
 */
public class CLLevel1 extends CLBLASBase {

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
    
    private static final Subprogram<cl_kernel> SET_ONE;
    private static final Subprogram<cl_kernel> DUP;   

    // special functions  
    private static final Subprogram<cl_kernel> EXP;
    private static final Subprogram<cl_kernel> NEG;
    private static final Subprogram<cl_kernel> SIGMOID;
    
    static {
        
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

    /**
     * Kann aufgerufen werden um den Static Constructur zu callen 
     */
    public static void setup() {
    	
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
    
    public static void setOne(CLFloatMatrix matrix) {
        runMatrixOperation(SET_ONE, matrix);
    }
    
    public static void dup(CLFloatMatrix matrix, CLFloatMatrix result) {
		runMatrixElementWiseOperation(DUP, matrix, result);
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
}
