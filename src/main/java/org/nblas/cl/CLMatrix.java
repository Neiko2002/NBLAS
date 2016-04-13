package org.nblas.cl;

import java.util.Optional;

import org.jblas.util.Random;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.nblas.cl.model.CLMemory;
import org.nblas.cl.model.CLPointer;
import org.nblas.generic.AMatrix;
import org.nblas.generic.Subprogram;

/**
 * 
 * @author Nico
 *
 */
public abstract class CLMatrix extends AMatrix {
	
	protected static final CLCore CORE = CLCore.getCore();

    protected CLMemory dataPointer;
    protected int clRows, clColumns;
    protected Optional<CLMemory> randomDataPointer;

    
	public CLMatrix(int rows, int columns) {
		super(rows, columns);

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
    public void release() {
        dataPointer.release();
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
	protected static void runMatrixOperation(Subprogram<cl_kernel> subprogram, CLMatrix a) {
		CORE.execute(subprogram, a.clRows, a.clColumns, a.rows, a.columns, a.dataPointer);
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
	protected static void runMatrixMatrixElementWiseOperation(Subprogram<cl_kernel> subprogram, CLMatrix a, CLMatrix b, CLMatrix result) {
		checkSameSize(a, b, result);
        CORE.execute(subprogram, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
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
	protected static void runMatrixElementWiseOperation(Subprogram<cl_kernel> subprogram, CLMatrix a, CLMatrix result) {
		checkSameSize(a, result);
        CORE.execute(subprogram, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer);
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
	protected static void runMatrixScalarElementWiseOperation(Subprogram<cl_kernel> subprogram, CLMatrix a, CLPointer scalar, CLMatrix result) {
        CORE.execute(subprogram, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, scalar, a.dataPointer);
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
	protected static void runMatrixRowVectorElementWiseOperation(Subprogram<cl_kernel> subprogram, CLMatrix a, CLMatrix b, CLMatrix result) {
        checkRowVectorSize(a, b, result);
        CORE.execute(subprogram, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
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
	protected static void runMatrixColumnVectorElementWiseOperation(Subprogram<cl_kernel> subprogram, CLMatrix a, CLMatrix b, CLMatrix result) {
		checkColumnVectorSize(a, b, result);
		CORE.execute(subprogram, a.clRows, a.clColumns, result.rows, result.columns, result.dataPointer, a.dataPointer, b.dataPointer);
	}	

	/**
	 * Warte so lange bis alle anstehenden Operationen auf der GPU durchgeführt wurden
	 */
    public static void waitOnComplete() {
        CORE.waitOnComplete();
    }

}
