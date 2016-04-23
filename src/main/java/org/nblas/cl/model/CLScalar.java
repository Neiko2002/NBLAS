package org.nblas.cl.model;

import org.jocl.Pointer;
import org.jocl.Sizeof;

public class CLScalar implements CLStorage {
	
	protected Pointer pointer;
	protected int sizeof;
	
	protected CLScalar(Pointer pointer, int sizeof) {
		this.pointer = pointer;
		this.sizeof = sizeof;
	}
	
	public Pointer getPointer() {
		return pointer;
	}

	public int getSizeof() {
		return sizeof;
	}

	public static CLScalar of(float val) {
		return new CLScalar(Pointer.to(new float[] {val}), Sizeof.cl_float);
	}
	
	public static CLScalar of(int val) {
		return new CLScalar(Pointer.to(new int[] {val}), Sizeof.cl_int);
	}
	
	public static CLScalar of(double val) {
		return new CLScalar(Pointer.to(new double[] {val}), Sizeof.cl_double);
	}
}
