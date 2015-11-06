package org.nblas.cl;

import org.jocl.cl_platform_id;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_device_id;

public class CLDevice {

	public enum CLDeviceType { CPU, GPU };
	
	private CLPlatform platform;
	private cl_device_id device_id;
	private CLDeviceType deviceType;
	private String deviceName;
	private boolean available;
	private boolean ecc;
	private String extensions;
	private long globalCacheSize;
	private int globalCachelineSize;
	private long globalMemorySize;
	private boolean unifiedMemory;
	private long localMemorySize;
	private int maxClockFrequency;
	private int computeUnits;
	private int maxConstantArgs;
	private long maxConstantBufferSize;
	private long maxAllocationSize;
	private int maxParameterSize;
	private int maxWorkGroupSize;
	private int maxWorkItemDimensions;
	private long[] maxWorkItemSizes;
	private String openclVersion;
	private String profile;
	private String version;
	
	private CLDevice(cl_device_id device_id) {
		this.device_id = device_id;
	}


	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("deviceName: "+deviceName+"("+deviceType.toString()+")\n");
		sb.append("ECC: "+ecc+"\n");
		sb.append("globalCacheSize: "+globalCacheSize+"\n");
		sb.append("globalCachelineSize: "+globalCachelineSize+"\n");
		sb.append("globalMemorySize: "+globalMemorySize+"\n");
		sb.append("unifiedMemory: "+unifiedMemory+"\n");
		sb.append("localMemorySize: "+localMemorySize+"\n");
		sb.append("maxClockFrequency: "+maxClockFrequency+"\n");
		sb.append("computeUnits: "+computeUnits+"\n");
		sb.append("maxConstantArgs: "+maxConstantArgs+"\n");
		sb.append("maxConstantBufferSize: "+maxConstantBufferSize+"\n");
		sb.append("maxAllocationSize: "+maxAllocationSize+"\n");
		sb.append("maxParameterSize: "+maxParameterSize+"\n");
		sb.append("maxWorkGroupSize: "+maxWorkGroupSize+"\n");
		sb.append("maxWorkItemDimensions: "+maxWorkItemDimensions+"\n");
		sb.append("maxWorkItemSizes: "+Arrays.toString(maxWorkItemSizes)+"\n");
		
		return sb.toString();
	}

	public int getTheoreticalComputingPower() {
		return maxClockFrequency * computeUnits;
	}
	
	public int getMaxWorkGroupSize() {
		return maxWorkGroupSize;
	}
	
	public int getComputeUnits() {
		return computeUnits;
	}

	public CLPlatform getPlatform() {
		return platform;
	}

	public cl_device_id getId() {
		return device_id;
	}
	
	
	
	
	
	
	
	
	

	private static String getString(cl_device_id device, int paramName) {
        // Obtain the length of the string that will be queried
        long[] size = new long[1];
        CL.clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte[] buffer = new byte[(int) size[0]];
        CL.clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length - 1);
    }
    
    private static long getLong(cl_device_id device, int paramName) {
        return getLongs(device, paramName, 1)[0];
    }
    
    private static long[] getLongs(cl_device_id device, int paramName, int numValues) {
        long[] values = new long[numValues];
        CL.clGetDeviceInfo(device, paramName, Sizeof.cl_long * numValues, Pointer.to(values), null);
        return values;
    }

    private static boolean getBoolean(cl_device_id device, int paramName) {
        return getBooleans(device, paramName, 1)[0];
    }

    private static boolean[] getBooleans(cl_device_id device, int paramName, int numValues) {
    	int[] values = new int[numValues];
        CL.clGetDeviceInfo(device, paramName, Sizeof.cl_int + numValues, Pointer.to(values), null);
        
        boolean[] result = new boolean[numValues];
        for (int i = 0; i < values.length; i++)
        	result[i] = (values[i] == 1);
        return result;
    }
    
    private static int getInt(cl_device_id device, int paramName) {
        return getInts(device, paramName, 1)[0];
    }

    private static int[] getInts(cl_device_id device, int paramName, int numValues) {
        int[] values = new int[numValues];
        CL.clGetDeviceInfo(device, paramName, Sizeof.cl_int * numValues, Pointer.to(values), null);
        return values;
    }
    
    private static long getSize(cl_device_id device, int paramName) {
        return getSizes(device, paramName, 1)[0];
    }


    private static long[] getSizes(cl_device_id device, int paramName, int numValues) {
        // The size of the returned data has to depend on
        // the size of a size_t, which is handled here
        ByteBuffer buffer = ByteBuffer.allocate(numValues * Sizeof.size_t).order(ByteOrder.nativeOrder());
        CL.clGetDeviceInfo(device, paramName, Sizeof.size_t * numValues, Pointer.to(buffer), null);
        long[] values = new long[numValues];
        if (Sizeof.size_t == 4) {
            for (int i = 0; i < numValues; i++) {
                values[i] = buffer.getInt(i * Sizeof.size_t);
            }
        } else {
            for (int i = 0; i < numValues; i++) {
                values[i] = buffer.getLong(i * Sizeof.size_t);
            }
        }
        return values;
    }
    
	private static CLDevice getDevice(cl_device_id device_id, CLPlatform platform, CLDeviceType deviceType) {
		
    	CLDevice device = new CLDevice(device_id);
    	
    	device.platform = platform;
    	device.deviceName = getString(device_id, CL.CL_DEVICE_NAME);
    	device.available = getBoolean(device_id, CL.CL_DEVICE_AVAILABLE);
    	device.ecc = getBoolean(device_id, CL.CL_DEVICE_ERROR_CORRECTION_SUPPORT);
    	device.extensions = getString(device_id, CL.CL_DEVICE_EXTENSIONS);
    	device.globalCacheSize = getLong(device_id, CL.CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
    	device.globalCachelineSize = getInt(device_id, CL.CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE);
    	device.globalMemorySize = getLong(device_id, CL.CL_DEVICE_GLOBAL_MEM_SIZE);
    	device.unifiedMemory = getBoolean(device_id, CL.CL_DEVICE_HOST_UNIFIED_MEMORY);
    	device.localMemorySize = getLong(device_id, CL.CL_DEVICE_LOCAL_MEM_SIZE);
      	device.maxClockFrequency = getInt(device_id, CL.CL_DEVICE_MAX_CLOCK_FREQUENCY );
      	device.computeUnits = getInt(device_id, CL.CL_DEVICE_MAX_COMPUTE_UNITS);
       	device.maxConstantArgs = getInt(device_id, CL.CL_DEVICE_MAX_CONSTANT_ARGS);
       	device.maxConstantBufferSize = getLong(device_id, CL.CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
     	device.maxAllocationSize = getLong(device_id, CL.CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    	device.maxParameterSize = (int) getSize(device_id, CL.CL_DEVICE_MAX_PARAMETER_SIZE);
        device.maxWorkGroupSize = (int) getSize(device_id, CL.CL_DEVICE_MAX_WORK_GROUP_SIZE);
        device.maxWorkItemDimensions = getInt(device_id, CL.CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
        device.maxWorkItemSizes = getSizes(device_id, CL.CL_DEVICE_MAX_WORK_ITEM_SIZES, device.maxWorkItemDimensions);
        device.openclVersion = getString(device_id, CL.CL_DEVICE_OPENCL_C_VERSION);
        device.profile  = getString(device_id, CL.CL_DEVICE_PROFILE);
        device.version = getString(device_id, CL.CL_DEVICE_VERSION);
    	device.deviceType = deviceType;
    	 
		return device;
	}
	
	public static CLDevice[] getDevices(CLPlatform platform) {
		  
		CLDeviceType[] clDeviceTypes = new CLDeviceType[] { CLDeviceType.CPU, CLDeviceType.GPU };
		long[] deviceTypeIds = new long[] { CL.CL_DEVICE_TYPE_CPU, CL.CL_DEVICE_TYPE_GPU };
		
		List<CLDevice> devices = new ArrayList<>(); 
		for (int i = 0; i < deviceTypeIds.length; i++) {
			CLDeviceType clDeviceType = clDeviceTypes[i];
			long deviceTypeId = deviceTypeIds[i];
			
	        int numDevicesPointer[] = new int[1];
	        CL.clGetDeviceIDs(platform.getId(), deviceTypeId, 0, null, numDevicesPointer);
	        int numDevices = numDevicesPointer[0];
	        
	        if(numDevices == 0) continue;
	        
	        cl_device_id device_ids[] = new cl_device_id[numDevices];        
	        CL.clGetDeviceIDs(platform.getId(), deviceTypeId, numDevices, device_ids, null);
	        
	        for (cl_device_id cl_device_id : device_ids)				
	        	devices.add(CLDevice.getDevice(cl_device_id, platform, clDeviceType));
		}
  
		return devices.toArray(new CLDevice[0]);
	}

}
