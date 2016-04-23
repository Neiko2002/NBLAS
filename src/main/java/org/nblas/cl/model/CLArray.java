package org.nblas.cl.model;

import org.jocl.Pointer;
import org.jocl.Sizeof;

public class CLArray extends CLScalar {
	
	protected int length;
	
	protected CLArray(Pointer pointer, int sizeof, int length) {
		super(pointer, sizeof);
		this.length = length;
	}	
	
	public int getLength() {
		return length;
	}
	
	@Override
	public int getSizeof() {
		return sizeof*length;
	}

	public static CLArray of(float[] val) {
		return new CLArray(Pointer.to(val), Sizeof.cl_float, val.length);
	}
	
	public static CLArray of(int[] val) {
		return new CLArray(Pointer.to(val), Sizeof.cl_int, val.length);
	}
	
	public static CLArray of(double[] val) {
		return new CLArray(Pointer.to(val), Sizeof.cl_double, val.length);
	}
	
	
	public static CLArray ofFloat(int length) {
		return new CLArray(null, Sizeof.cl_float, length);
	}
	
	public static CLArray ofInt(int length) {
		return new CLArray(null, Sizeof.cl_int, length);
	}
	
	public static CLArray ofDouble(int length) {
		return new CLArray(null, Sizeof.cl_double, length);
	}
}
