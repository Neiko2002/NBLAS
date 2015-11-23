package org.nblas.cuda;

import org.nblas.generic.AMatrix;

import jcuda.Pointer;

/**
 * 
 * @author Nico
 *
 */
public class CUDAMatrix extends AMatrix {

	protected static final CudaCore CORE = CudaCore.getCore();
	  
	
	protected Pointer dataPointer;
	  
	public CUDAMatrix(int rows, int columns) {
		super(rows, columns);
	}
   
    @Override
    public void free() {
        CORE.free(dataPointer);
    }

}
