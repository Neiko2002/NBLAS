package org.nblas.cl;

import java.util.Optional;

import org.jblas.util.Random;
import org.jocl.Pointer;
import org.jocl.cl_kernel;
import org.nblas.cl.model.CLMemory;
import org.nblas.cl.model.CLScalar;
import org.nblas.cl.model.CLStorage;
import org.nblas.generic.AMatrix;
import org.nblas.generic.Subprogram;

/**
 * 
 * @author Nico
 *
 */
public abstract class CLMatrix extends AMatrix implements CLStorage {
		
	protected CLCore CORE;

	// memory on the device
	protected CLMemory clMemory;    
    protected int clRows, clColumns;
    protected Optional<CLMemory> randomDataPointer;
    
	public CLMatrix(int rows, int columns, CLCore core) {
		super(rows, columns);
		
		// Backend 
		this.CORE = core;
	
		// row or column vector else matrix
		this.clRows = getValidSize(rows, CORE.getThreadCountX());			
		this.clColumns = getValidSize(columns, CORE.getThreadCountY());

		this.randomDataPointer = Optional.empty();
	}
	
	protected int getValidSize(int size, int divisor) {
		if(size == 1) return 1;
		return (int) Math.ceil((double)size / divisor) * divisor;
	}
	
    protected void initRandom() {
        if (!randomDataPointer.isPresent()) {
            int[] initRandom = new int[CORE.getThreadCountY() * CORE.getThreadCountX() * 4];
            for (int i = 0; i < initRandom.length; i++) {
                initRandom[i] = Random.nextInt(Integer.MAX_VALUE - 1234) + 1234;
            }
            randomDataPointer = Optional.of(CORE.malloc(initRandom));
        }
    }
	
    @Override
    public Pointer getPointer() {
    	return clMemory.getPointer();
    }
    
    @Override
    public int getSizeof() {
    	return clMemory.getSizeof();
    }
    
    @Override
    public void release() {
        clMemory.release();
        if (randomDataPointer.isPresent()) 
            randomDataPointer.get().release();
        released = true;
    }  
    


    // --------------------------------------- helper methods ----------------------------------------

    /**
     * Führe ein OpenCL Programm auf einer Matrix aus.
     * 
     * @param subprogram
     * @param a
     */
	protected void runMatrixOperation(Subprogram<cl_kernel> subprogram) {
		CORE.execute(subprogram, this.clRows, this.clColumns, this, CLScalar.of(this.rows), CLScalar.of(this.columns));
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
	protected void runMatrixMatrixElementWiseOperation(Subprogram<cl_kernel> subprogram, CLMatrix b, CLMatrix result) {
		checkSameSize(this, b, result);
        CORE.execute(subprogram, this.clRows, this.clColumns, result, CLScalar.of(result.rows), CLScalar.of(result.columns), this, b);
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
	protected void runMatrixElementWiseOperation(Subprogram<cl_kernel> subprogram, CLMatrix result) {
		checkSameSize(this, result);
        CORE.execute(subprogram, this.clRows, this.clColumns, result, CLScalar.of(result.rows), CLScalar.of(result.columns), this);
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
	protected void runMatrixScalarElementWiseOperation(Subprogram<cl_kernel> subprogram, CLScalar scalar, CLMatrix result) {
        CORE.execute(subprogram, this.clRows, this.clColumns, result, CLScalar.of(result.rows), CLScalar.of(result.columns), this, scalar);
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
	protected void runMatrixRowVectorElementWiseOperation(Subprogram<cl_kernel> subprogram, CLMatrix b, CLMatrix result) {
        checkRowVectorSize(this, b, result);
        CORE.execute(subprogram, this.clRows, this.clColumns, result, CLScalar.of(result.rows), CLScalar.of(result.columns), this, b);
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
	protected void runMatrixColumnVectorElementWiseOperation(Subprogram<cl_kernel> subprogram, CLMatrix b, CLMatrix result) {
		checkColumnVectorSize(this, b, result);
		CORE.execute(subprogram, this.clRows, this.clColumns, result, CLScalar.of(result.rows), CLScalar.of(result.columns), this, b);
	}	

	/**
	 * Warte so lange bis alle anstehenden Operationen auf der GPU durchgeführt wurden
	 */
    public void waitOnComplete() {
        CORE.waitOnComplete();
    }

}
