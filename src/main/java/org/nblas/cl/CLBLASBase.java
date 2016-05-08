package org.nblas.cl;

import org.jocl.cl_kernel;
import org.nblas.cl.model.CLScalar;
import org.nblas.function.AFunctionBuilder;
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
 * 
 * @author Nico
 *
 */
public abstract class CLBLASBase {
	    
	private final AFunctionBuilder<cl_kernel> builder;
	private final CLCore CORE;
	
    protected CLBLASBase(CLContext context, AFunctionBuilder<cl_kernel> builder) {
		this.builder = builder;
		this.CORE = CLCore.getCore(context.getDeviceId());
	}
	
    /**
     * Läd und kompiliert eine Kernel
     * 
     * @param functionObject
     * @param argumentTypes
     * @return
     */
	protected Subprogram<cl_kernel> buildPredefinedFunction(String name, AFunctionObject functionObject, ArgumentType... argumentTypes) {
		Subprogram<cl_kernel> subprogram = builder.buildFunction(name, functionObject, argumentTypes);
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
	protected void runMatrixOperation(Subprogram<cl_kernel> subprogram, CLMatrix a) {
		a.runMatrixOperation(subprogram);
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
	protected void runMatrixMatrixElementWiseOperation(Subprogram<cl_kernel> subprogram, CLMatrix a, CLMatrix b, CLMatrix result) {
		a.runMatrixMatrixElementWiseOperation(subprogram, b, result);
	}
	
	/**
     * Führe ein OpenCL Programm auf einer Matrix durch,
     * das Resultat wird in eine zweite Matrix gespeichert
     * 
	 * @param programId
	 * @param matrix
	 * @param result
	 */
	protected void runMatrixElementWiseOperation(Subprogram<cl_kernel> subprogram, CLMatrix a, CLMatrix result) {
		a.runMatrixElementWiseOperation(subprogram, result);
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
	protected void runMatrixScalarElementWiseOperation(Subprogram<cl_kernel> subprogram, CLMatrix a, CLScalar scalar, CLMatrix result) {
		a.runMatrixScalarElementWiseOperation(subprogram, scalar, result);
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
	protected void runMatrixRowVectorElementWiseOperation(Subprogram<cl_kernel> subprogram, CLMatrix a, CLMatrix b, CLMatrix result) {
		a.runMatrixRowVectorElementWiseOperation(subprogram, b, result);
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
	protected void runMatrixColumnVectorElementWiseOperation(Subprogram<cl_kernel> subprogram, CLMatrix a, CLMatrix b, CLMatrix result) {
		a.runMatrixColumnVectorElementWiseOperation(subprogram, b, result);
	}	
}
