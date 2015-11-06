package org.nblas.cl;

import org.jocl.cl_kernel;
import org.nblas.function.ArgumentType;
import org.nblas.function.generic.AFunctionObject;
import org.nblas.generic.Subprogram;

/**
 * BLAS functionality is categorized into three sets of routines called "levels", 
 * which correspond to both the chronological order of definition and publication, 
 * as well as the degree of the polynomial in the complexities of algorithms; 
 * Level 1 BLAS operations typically take linear time, O(n), 
 * Level 2 operations quadratic time and 
 * Level 3 operations cubic time.
 * 
 * TODO hier sollten nur ANativeCLMatrix verwendet werden
 * 
 * @author Nico
 *
 */
public abstract class CLBLASBase {
	
	private static final CLCore CORE = CLCore.getCore();
	private static final CLFloatFunctionBuilder builder = new CLFloatFunctionBuilder();
    
	protected static Subprogram<cl_kernel> buildPredefinedFunction(AFunctionObject functionObject, ArgumentType... argumentTypes) {
		Subprogram<cl_kernel> subprogram = builder.buildFunction(functionObject, argumentTypes);
		subprogram.setCustom(false);
		CORE.loadFromGeneratedSubprogram(subprogram);
		return subprogram;
	}
	
	/**
     * Führe ein OpenCL Programm auf einer Matrix aus.
     * 
     * @param subprogram
     * @param a
     */
	protected static void runMatrixOperation(Subprogram<cl_kernel> subprogram, CLFloatMatrix a) {
		CLFloatMatrix.runMatrixOperation(subprogram, a);
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
		CLFloatMatrix.runMatrixMatrixElementWiseOperation(subprogram, a, b, result);
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
		CLFloatMatrix.runMatrixElementWiseOperation(subprogram, a, result);
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
	    CLFloatMatrix.runMatrixScalarElementWiseOperation(subprogram, a, scalar, result);
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
		CLFloatMatrix.runMatrixRowVectorElementWiseOperation(subprogram, a, b, result);
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
		CLFloatMatrix.runMatrixColumnVectorElementWiseOperation(subprogram, a, b, result);
	}	
}
