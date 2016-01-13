package org.nblas.cuda;

import org.nblas.function.AFunctionBuilder;
import org.nblas.function.ArgumentType;
import org.nblas.function.generic.AFunctionObject;
import org.nblas.generic.Subprogram;

import jcuda.driver.CUfunction;

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
public abstract class CudaBLASBase {
	
	private static final CudaCore CORE = CudaCore.getCore();
    
	private final AFunctionBuilder<CUfunction> builder;
	
    protected CudaBLASBase(AFunctionBuilder<CUfunction> builder) {
		this.builder = builder;
	}
	
    /**
     * Läd und kompiliert eine Kernel
     * 
     * @param functionObject
     * @param argumentTypes
     * @return
     */
	protected Subprogram<CUfunction> buildPredefinedFunction(AFunctionObject functionObject, ArgumentType... argumentTypes) {
		Subprogram<CUfunction> subprogram = builder.buildFunction(functionObject, argumentTypes);
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
	protected void runMatrixOperation(Subprogram<CUfunction> subprogram, CudaMatrix a) {
		CudaMatrix.runMatrixOperation(subprogram, a);
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
	protected void runMatrixMatrixElementWiseOperation(Subprogram<CUfunction> subprogram, CudaMatrix a, CudaMatrix b, CudaMatrix result) {
		CudaMatrix.runMatrixMatrixElementWiseOperation(subprogram, a, b, result);
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
	protected void runMatrixElementWiseOperation(Subprogram<CUfunction> subprogram, CudaMatrix a, CudaMatrix result) {
		CudaMatrix.runMatrixElementWiseOperation(subprogram, a, result);
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
	protected void runMatrixScalarElementWiseOperation(Subprogram<CUfunction> subprogram, CudaMatrix a, CudaMatrix scalar, CudaMatrix result) {
		CudaMatrix.runMatrixScalarElementWiseOperation(subprogram, a, scalar, result);
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
	protected void runMatrixRowVectorElementWiseOperation(Subprogram<CUfunction> subprogram, CudaMatrix a, CudaMatrix b, CudaMatrix result) {
		CudaMatrix.runMatrixRowVectorElementWiseOperation(subprogram, a, b, result);
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
	protected void runMatrixColumnVectorElementWiseOperation(Subprogram<CUfunction> subprogram, CudaMatrix a, CudaMatrix b, CudaMatrix result) {
		CudaMatrix.runMatrixColumnVectorElementWiseOperation(subprogram, a, b, result);
	}	
}
