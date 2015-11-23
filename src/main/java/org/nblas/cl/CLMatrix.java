package org.nblas.cl;

import java.util.Optional;

import org.jocl.cl_mem;
import org.nblas.cl.blas.CLLevel1;
import org.nblas.generic.AMatrix;

/**
 * 
 * @author Nico
 *
 */
public class CLMatrix extends AMatrix {
	
	protected static final CLCore CORE = CLCore.getCore();

	static {
		CLLevel1.setup();
		CORE.compileMatrixFunctions();
	}

    protected cl_mem dataPointer;
    protected Optional<cl_mem> randomDataPointer;
    protected int clRows, clColumns, clLength;
    
	public CLMatrix(int rows, int columns) {
		super(rows, columns);

		this.clColumns = (int) Math.ceil(columns / (double) CORE.getThreadCount_Y()) * CORE.getThreadCount_Y();
		this.clRows = (int) Math.ceil(rows / (double) CORE.getThreadCount_X()) * CORE.getThreadCount_X();
		this.clLength = clColumns * clRows;

		randomDataPointer = Optional.empty();
	}

	
    @Override
    public void free() {
        CORE.free(dataPointer);
        if (randomDataPointer.isPresent()) {
            CORE.free(randomDataPointer.get());
        }
        released = true;
    }  
}
