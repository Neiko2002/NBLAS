package org.nblas;

import org.nblas.impl.ContextDefault;

/**
 * Created by Moritz on 4/27/2015.
 * 
 * Kein Host mehr. Context zu Backend umbenennen.
 * nur noch Cuda, OpenCL, Java, MKL, CBLAS
 * 
 */
public interface Context {

	public enum Backend { CUDA, OpenCL, MKL, JBLAS };	
	public enum Precision { HALF, SINGLE, DOUBLE };
    
    public static Context createCudaSinglePrecisionContext() {
    	return createContext(Precision.SINGLE, Backend.CUDA, -1);
    }
    
    public static Context createOpenCLSinglePrecisionContext() {
    	return createContext(Precision.SINGLE, Backend.OpenCL, -1);
    }
    
    public static Context createJBlasSinglePrecisionContext() {
    	return createContext(Precision.SINGLE, Backend.JBLAS, -1);
    }
    
    public static Context createContext(Precision precision, Backend backend) {
    	return createContext(precision, backend, -1);
    }
    
    /**
     * Eine deviceId von -1 ermittelt das best geeigneste Device und verwendet dieses.
     * 
     * @param precision
     * @param backend
     * @param deviceId
     * @return
     */
    public static Context createContext(Precision precision, Backend backend, int deviceId) {
    	return ContextDefault.createContext(precision, backend, deviceId);
    }
    

    public default boolean isDoublePrecision() {
        return getPrecision() == Context.Precision.DOUBLE;
    }
    
    public default boolean isSinglePrecision() {
        return getPrecision() == Context.Precision.SINGLE;
    }
    
    public default boolean isHalfPrecision() {
        return getPrecision() == Context.Precision.HALF;
    }
    
    public Precision getPrecision();
    
    
    
    public default boolean isCUDA() {
        return getBackend() == Context.Backend.CUDA;
    }
    
    public default boolean isOpenCL() {
        return getBackend() == Context.Backend.OpenCL;
    }
    
    public default boolean isMKL() {
        return getBackend() == Context.Backend.MKL;
    }
    
    public default boolean isJBLAS() {
        return getBackend() == Context.Backend.JBLAS;
    }
    
	public Backend getBackend();
	
	public int getDeviceId();
	

}