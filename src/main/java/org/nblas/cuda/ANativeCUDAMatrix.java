package org.nblas.cuda;

import org.nblas.generic.ANativeMatrix;

/**
 * 
 * @author Nico
 *
 */
public abstract class ANativeCUDAMatrix extends ANativeMatrix {

	public ANativeCUDAMatrix(int rows, int columns) {
		super(rows, columns);
	}
   

}
