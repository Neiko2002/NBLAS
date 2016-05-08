package org.nblas.cuda;

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

	protected CudaCore CORE;
	
	protected Pointer dataPointer;
	  
	protected CudaMatrix(int rows, int columns, CudaCore core) {
		super(rows, columns);
		this.CORE = core;
	}
	
    @Override
    public void release() {
        CORE.release(dataPointer);
		released = true;
    }

    
  // --------------------------------------- helper methods ----------------------------------------
    
    /**
     * Führe ein CUDA Programm auf einer Matrix aus.
     * 
     * @param subprogram
     * @param a
     */
	protected void runMatrixOperation(Subprogram<CUfunction> subprogram) {
		CORE.execute(subprogram, this.rows, this.columns, this.dataPointer);
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
	protected void runMatrixMatrixElementWiseOperation(Subprogram<CUfunction> subprogram, CudaMatrix b, CudaMatrix result) {
		checkSameSize(this, b, result);
        CORE.execute(subprogram, result.rows, result.columns, result.dataPointer, this.dataPointer, b.dataPointer);
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
	protected void runMatrixElementWiseOperation(Subprogram<CUfunction> subprogram, CudaMatrix result) {
		checkSameSize(this, result);
        CORE.execute(subprogram, result.rows, result.columns, result.dataPointer, this.dataPointer);
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
	protected void runMatrixScalarElementWiseOperation(Subprogram<CUfunction> subprogram, CudaMatrix scalar, CudaMatrix result) {
        CORE.execute(subprogram, result.rows, result.columns, result.dataPointer, this.dataPointer, scalar.dataPointer);
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
	protected void runMatrixRowVectorElementWiseOperation(Subprogram<CUfunction> subprogram, CudaMatrix b, CudaMatrix result) {
        checkRowVectorSize(this, b, result);
        CORE.execute(subprogram, result.rows, result.columns, result.dataPointer, this.dataPointer, b.dataPointer);
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
	protected void runMatrixColumnVectorElementWiseOperation(Subprogram<CUfunction> subprogram, CudaMatrix b, CudaMatrix result) {
		checkColumnVectorSize(this, b, result);
		CORE.execute(subprogram, result.rows, result.columns, result.dataPointer, this.dataPointer, b.dataPointer);
	}	
}
