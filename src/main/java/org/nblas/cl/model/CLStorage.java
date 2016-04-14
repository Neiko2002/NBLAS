package org.nblas.cl.model;

import org.jocl.Pointer;

public interface CLStorage {
	
	public Pointer getPointer();
	public int getSizeof();
}
