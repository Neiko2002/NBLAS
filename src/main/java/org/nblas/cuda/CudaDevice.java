package org.nblas.cuda;

import java.util.ArrayList;
import java.util.List;

import jcuda.driver.CUcontext;
import jcuda.driver.CUctx_flags;
import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

/**
 * http://www.pcper.com/reviews/Graphics-Cards/NVIDIA-Reveals-GK110-GPU-Kepler-71B-Transistors-15-SMX-Units
 * GK110 SMX Units: 192 single-precision units, 64 double-precision units, 32 special function units and 32 load/store units
 * GM108 SMM Units: 128 single-precision units, 64 double-precision units, 32 special function units and 32 load/store units
 *  
 * Tesla K20c
 * https://www.techpowerup.com/gpudb/564/tesla-k20c.html
 * SMX count: 13 
 * SPUs: 2496
 * DPUs: 832
 * SFUs: 416
 * LD/ST: 416
 * 
 * https://devtalk.nvidia.com/default/topic/745504/comparing-cpu-and-gpu-theoretical-gflops/
 * https://devtalk.nvidia.com/default/topic/522964/calculatin-flops-of-gpu/
 * Theoretische TFLops = GPU Clock * Shading Units * 2 
 * Shading Units = multiprocessorCount * (SP units oder DP units)
 * Theoretische TFLops = clockRate * multiprocessorCount * (SP Core oder DP Unit) * 2 
 * 
 * @author Nico
 *
 */
class CudaDevice {

	private CUcontext contextPointer;
	private CUdevice devicePointer;
	
	private int cudaVersion;
	private cudaDeviceProp props;
		
	private CudaDevice(CUdevice devicePointer, CUcontext contextPointer) {
		this.devicePointer = devicePointer;
		this.contextPointer = contextPointer;
	}
	
	public int getCudaVersion() {
		return cudaVersion;
	}

	public int getMaxThreadPerBlock() {
		return props.maxThreadsPerBlock;
	}
	
	/**
	 * multiProcessor * clockRate * 2
	 * @return
	 */
	public int getTheoreticalComputingPower() {
		return props.multiProcessorCount * props.clockRate * 2;
	}
	
	public void use() {
		
//		JCuda.cudacre
//		JCuda.cudaSetDeviceFlags(JCuda.cudaDeviceScheduleBlockingSync);
//		JCudaDriver.cu
        
		 JCudaDriver.cuCtxCreate(contextPointer, CUctx_flags.CU_CTX_SCHED_AUTO, devicePointer);
//		 JCudaDriver.cuCtxCreate(contextPointer, CUctx_flags.CU_CTX_SCHED_BLOCKING_SYNC, devicePointer); // Blockt CPU beim synchonisieren
//		 JCudaDriver.cuCtxCreate(contextPointer, CUctx_flags.CU_CTX_SCHED_YIELD, devicePointer); // TODO performanter!?
	}
	
	public void release() {
		 JCudaDriver.cuCtxDestroy(contextPointer);
	}
	
	@Override
	public String toString() {
		return props.toFormattedString();
	}

	private static CudaDevice getDevice(int ordinal) {
		
		CUcontext contextPointer = new CUcontext();
		CUdevice devicePointer = new CUdevice();
		JCudaDriver.cuDeviceGet(devicePointer, 0);

		CudaDevice device = new CudaDevice(devicePointer, contextPointer);
		
		// lese ein paar Eigenschaften aus
		int[] valueBuffer = new int[1];
        JCuda.cudaRuntimeGetVersion(valueBuffer);
        device.cudaVersion = valueBuffer[0];        
        
        cudaDeviceProp prop = device.props = new cudaDeviceProp();
        JCuda.cudaGetDeviceProperties(prop, ordinal);
		
		return device;
	}
	
	public static CudaDevice[] getDevices() {
		
        int[] value = new int[1];
        JCudaDriver.cuDeviceGetCount(value);
        int deviceCount = value[0];
        
		List<CudaDevice> devices = new ArrayList<>();
		for (int i = 0; i < deviceCount; i++) 
			devices.add(getDevice(i));	

		return devices.toArray(new CudaDevice[devices.size()]);
	}
}
