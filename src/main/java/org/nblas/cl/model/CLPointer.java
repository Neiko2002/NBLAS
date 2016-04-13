package org.nblas.cl.model;

import org.jocl.Pointer;
import org.jocl.Sizeof;

public class CLPointer {
	
	protected Pointer pointer;
	protected int sizeof;
	protected int length;
	
	public CLPointer(Pointer pointer, int sizeof, int length) {
		this.pointer = pointer;
		this.sizeof = sizeof;
		this.length = length;
	}
	
	public Pointer getPointer() {
		return pointer;
	}

	public int getSizeof() {
		return sizeof;
	}
	
	public int getLength() {
		return length;
	}

	public static CLPointer of(float val) {
		return new CLPointer(Pointer.to(new float[] {val}), Sizeof.cl_float, 1);
	}
	
	public static CLPointer of(int val) {
		return new CLPointer(Pointer.to(new int[] {val}), Sizeof.cl_int, 1);
	}
	
	public static CLPointer of(double val) {
		return new CLPointer(Pointer.to(new double[] {val}), Sizeof.cl_double, 1);
	}
}
