package org.nblas.impl;

import java.util.HashMap;
import java.util.Map;

import org.nblas.Context;
import org.nblas.cl.CLContext;
import org.nblas.cuda.CudaContext;
import org.nblas.jblas.JBlasContext;

public abstract class ContextDefault implements Context {

    /**
     * all active context
     */
    protected static final Map<Integer, Context> activeContext = new HashMap<>();
    
    protected static int createHashCode(Precision precision, Backend backend, int deviceId) {
    	return (precision.ordinal() << 16) |  (backend.ordinal() << 8) |  deviceId;
    }
    
    public static Context createContext(Precision precision, Backend backend, int deviceId) {
    	int hashCode = createHashCode(precision, backend, deviceId);
    	Context context = activeContext.get(hashCode);
    	if(context == null) {    		
    		switch (backend) {
				case CUDA:
					context = CudaContext.create(precision, deviceId);
					break;
				case MKL:
					break;
				case OpenCL:
					context = CLContext.create(precision, deviceId);
					break;
				case JBLAS:
				default:
					context = JBlasContext.create(precision, deviceId);
					break;		
			}
    		activeContext.put(hashCode, context);
    	}
    	return context;
    }
	
    protected int deviceId;
    protected Precision precision;
    
    protected ContextDefault(Precision precision, int deviceId) {
		this.deviceId = deviceId;
		this.precision = precision;
	}
    
	@Override
	public Precision getPrecision() {
		return precision;
	}
	
	@Override
	public int getDeviceId() {
		return deviceId;
	}
	
	@Override
	public int hashCode() {		
		return createHashCode(getPrecision(), getBackend(), getDeviceId());
	}
	
    @Override
    public boolean equals(Object obj) {
    	return obj.hashCode() == hashCode();
    }
}
