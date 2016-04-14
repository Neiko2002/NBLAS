package org.nblas.cl.model;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

public class CLMemory extends CLScalar {

	protected cl_mem memory;
	
	public CLMemory(cl_mem memory, int sizeof, int length) {
		super(Pointer.to(memory), sizeof);
		this.memory = memory;
	}

	@Override
	public int getSizeof() {
		return Sizeof.cl_mem;
	}
	
	public cl_mem getMemory() {
		return memory;
	}
	
	public void release() {
		CL.clReleaseMemObject(memory);
	}
}
