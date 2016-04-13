package org.nblas.cl.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.cl_platform_id;

public class CLPlatform {

	private CLDevice[] devices;
	private cl_platform_id platform_id;
	
	private String version;
	private String vendor;
	private String name;
	private String profile;
	private String extension;
	
	private CLPlatform(cl_platform_id platform_id) {
		this.platform_id = platform_id;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(name+"\n");
		sb.append(vendor+"\n");
		sb.append(profile+"\n");
		sb.append(version+"\n");
		sb.append(extension+"\n\n");
		for (int i = 0; i < devices.length; i++) {
			sb.append("\t"+(i+1)+". Device:\n");
			String[] parts = devices[i].toString().split("\\n");
			for (int p = 0; p < parts.length; p++)
				sb.append("\t"+parts[p]+"\n");
			sb.append("\n");
		}
		return sb.toString();
	}

	public cl_platform_id getId() {
		return platform_id;
	}
	
	public  CLDevice[] getDevices() {
		return devices;
	}
	
	public CLDevice getFastestDevice() {
		final Comparator<CLDevice> performanceComperator = (c1, c2) -> Integer.compare( c1.getTheoreticalComputingPower(), c2.getTheoreticalComputingPower());
		return Arrays.stream(devices).max(performanceComperator).get();
    }
	
	public CLDevice getFastestGPU() {
		final Comparator<CLDevice> performanceComperator = (c1, c2) -> Integer.compare( c1.getTheoreticalComputingPower(), c2.getTheoreticalComputingPower());
		return Arrays.stream(devices).filter(CLDevice::isGPU).max(performanceComperator).orElse(null);
    }
	
	private static String getString(cl_platform_id platform, int paramName)
    {
        // Obtain the length of the string that will be queried
        long size[] = new long[1];
        CL.clGetPlatformInfo(platform, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int)size[0]];
        CL.clGetPlatformInfo(platform, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }
	
	private static CLPlatform getPlatform(cl_platform_id platform_id) {
		
    	CLPlatform platform = new CLPlatform(platform_id);
    	
    	try {
    		platform.extension = getString(platform_id, CL.CL_PLATFORM_EXTENSIONS);
    		platform.version = getString(platform_id, CL.CL_PLATFORM_VERSION);
    		platform.name = getString(platform_id, CL.CL_PLATFORM_NAME);
    		platform.vendor = getString(platform_id, CL.CL_PLATFORM_VENDOR);
    		platform.profile = getString(platform_id, CL.CL_PLATFORM_PROFILE);
    		platform.devices = CLDevice.getDevices(platform);
			
		} catch (Exception e) {
			return null;
		}
    	
		return platform;
	}

	public static CLPlatform[] getPlatforms() {
		
        int numPlatformsPointer[] = new int[1];
        CL.clGetPlatformIDs(0, null, numPlatformsPointer);
        int numPlatforms = numPlatformsPointer[0];

        if(numPlatforms == 0)
        	return new CLPlatform[0];
        
        cl_platform_id platform_ids[] = new cl_platform_id[numPlatforms];
        CL.clGetPlatformIDs(numPlatforms, platform_ids, null);
        
        List<CLPlatform> platforms = new ArrayList<>();
        for (int i = 0; i < numPlatforms; i++) {
        	CLPlatform platform = CLPlatform.getPlatform(platform_ids[i]);
        	if(platform != null && platform.getDevices().length > 0)
        		platforms.add(platform);
        }
    	
		return platforms.toArray(new CLPlatform[0]);
	}
}
