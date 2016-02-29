package org.nblas;

/**
 * Created by Moritz on 4/27/2015.
 */
public final class Context {

	public enum DeviceInterface { CUDA, OpenCL };
	public enum Precision { SINGLE, DOUBLE };
	public enum Host { CPU, GPU };
	
	protected DeviceInterface deviceInterface;
    protected Precision precision;
    protected Host host;

    private Context(Precision precision, Host host, DeviceInterface deviceInterface) {
    	this.deviceInterface = deviceInterface;
    	this.precision = precision;
    	this.host = host;
    }

//    public static Context createCudaDoublePrecisionContext() {
//        return new Context(Precision.DOUBLE, Host.GPU, DeviceInterface.CUDA);
//    }
//
//    public static Context createOpenCLDoublePrecisionContext() {
//        return new Context(Precision.DOUBLE, Host.GPU, DeviceInterface.OpenCL);
//    }
//
//    public static Context createJBLASDoublelePrecisionContext() {
//        return new Context(Precision.DOUBLE, Host.CPU, DeviceInterface.OpenCL);
//    }

    public boolean isDoublePrecision() {
        return precision == Context.Precision.DOUBLE;
    }
    
    public boolean isSinglePrecision() {
        return precision == Context.Precision.SINGLE;
    }

    public boolean isCPU() {
        return host == Context.Host.CPU;
    }
    
    public boolean isGPU() {
        return host == Context.Host.GPU;
    }

    public boolean isCUDA() {
        return deviceInterface == Context.DeviceInterface.CUDA;
    }
    
    public boolean isOpenCL() {
        return deviceInterface == Context.DeviceInterface.OpenCL;
    }
    
    @Override
    public boolean equals(Object obj) {
    	if(obj instanceof Context) {
    		Context otherContext = (Context) obj;
    		if(host != otherContext.host) return false;
    		if(deviceInterface != otherContext.deviceInterface) return false;
    		if(precision != otherContext.precision) return false;
    		return true;
    	}
    	return false;
    }
    
    
    

    public static Context CudaSinglePrecisionContext = new Context(Precision.SINGLE, Host.GPU, DeviceInterface.CUDA);
    public static Context OpenCLSinglePrecisionContext = new Context(Precision.SINGLE, Host.GPU, DeviceInterface.OpenCL);
    public static Context JBLASSinglePrecisionContext = new Context(Precision.SINGLE, Host.CPU, DeviceInterface.OpenCL);
}
