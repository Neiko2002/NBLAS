package org.nblas.cuda;

import org.nblas.function.AFunctionBuilder;
import org.nblas.function.ArgumentType;
import org.nblas.function.generic.AFunctionObject;
import org.nblas.generic.AMatrix;
import org.nblas.generic.Subprogram;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

/**
 * 
 * @author Nico
 *
 */
public class CudaMatrix extends AMatrix {

	protected static final CudaCore CORE = CudaCore.getCore();
	
	protected static Subprogram<CUfunction> buildPredefinedFunction(AFunctionBuilder<CUfunction> builder, AFunctionObject functionObject, ArgumentType... argumentTypes) {
        Subprogram<CUfunction> subprogram = builder.buildFunction(functionObject, argumentTypes);
        subprogram.setCustom(false);
        CORE.loadFromGeneratedSubprogram(subprogram);
        return subprogram;
    }
		
	protected Pointer dataPointer;
	  
	protected CudaMatrix(int rows, int columns) {
		super(rows, columns);
	}
	
    @Override
    public void free() {
        CORE.free(dataPointer);
		released = true;
    }

    
  // --------------------------------------- helper methods ----------------------------------------
    
    /**
     * Führe ein CUDA Programm auf einer Matrix aus.
     * 
     * @param subprogram
     * @param a
     */
	protected static void runMatrixOperation(Subprogram<CUfunction> subprogram, CudaMatrix a) {
		CORE.execute(subprogram, a.rows, a.columns, a.dataPointer);
	}
	
    
    /**
     * Führe ein CUDA Programm auf zwei gleich große Matrizen durch,
     * das Resultat wird in eine dritte Matrix gespeichert
     *  
     * @param programId
     * @param a
     * @param b
     * @param result
     */
	protected static void runMatrixMatrixElementWiseOperation(Subprogram<CUfunction> subprogram, CudaMatrix a, CudaMatrix b, CudaMatrix result) {
		checkSameSize(a, b, result);
        CORE.execute(subprogram, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
	}
	
	
	/**
     * Führe ein CUDA Programm auf einer Matrix durch,
     * das Resultat wird in eine zweite Matrix gespeichert
     * 
	 * @param programId
	 * @param matrix
	 * @param scalar
	 * @param result
	 */
	protected static void runMatrixElementWiseOperation(Subprogram<CUfunction> subprogram, CudaMatrix a, CudaMatrix result) {
		checkSameSize(a, result);
        CORE.execute(subprogram, result.rows, result.columns, result.dataPointer, a.dataPointer);
	}
	
	/**
     * Führe ein CUDA Programm auf einer Matrix und einem Scalar durch,
     * das Resultat wird in eine zweite Matrix gespeichert
     * 
	 * @param programId
	 * @param matrix
	 * @param scalar
	 * @param result
	 */
	protected static void runMatrixScalarElementWiseOperation(Subprogram<CUfunction> subprogram, CudaMatrix a, CudaMatrix scalar, CudaMatrix result) {
        CORE.execute(subprogram, result.rows, result.columns, result.dataPointer, a.dataPointer, scalar.dataPointer);
	}
	
	/**
     * Führe ein CUDA Programm auf einer Matrix und einem Row Vector durch,
     * das Resultat wird in eine zweite Matrix gespeichert
     * 
	 * @param programId
	 * @param matrix 
	 * @param row vector
	 * @param result
	 */
	protected static void runMatrixRowVectorElementWiseOperation(Subprogram<CUfunction> subprogram, CudaMatrix a, CudaMatrix b, CudaMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(subprogram, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
	}	
	
	/**
     * Führe ein CUDA Programm auf einer Matrix und einem Column Vector durch,
     * das Resultat wird in eine zweite Matrix gespeichert
     * 
	 * @param programId
	 * @param matrix
	 * @param column vector
	 * @param result
	 */
	protected static void runMatrixColumnVectorElementWiseOperation(Subprogram<CUfunction> subprogram, CudaMatrix a, CudaMatrix b, CudaMatrix result) {
		checkColumnVectorSize(a, b, result);
		CORE.execute(subprogram, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
	}	
}
