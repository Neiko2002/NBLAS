package org.nblas.cl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.IntStream;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_event;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_program;
import org.nblas.cl.model.CLDevice;
import org.nblas.cl.model.CLMemory;
import org.nblas.cl.model.CLPlatform;
import org.nblas.cl.model.CLStorage;
import org.nblas.generic.Subprogram;

/**
 *   
 * @author Nico
 *
 */
class CLCore {

	private static final Map<Integer, CLCore> activeCores = new HashMap<>();
	
	/**
	 * Setup the CLCore for a specific device. 
	 * Device id -1 means the fastest avaiable device.
	 * 
	 * @param deviceId
	 * @return
	 */
	protected static int setup(int deviceId) {
		
		CL.setExceptionsEnabled(true);
		
		// get a list of all OpenCL devices
		final CLDevice[] devices = Arrays.stream(CLPlatform.getPlatforms())
					   			 .filter(Objects::nonNull)
					   			 .map(CLPlatform::getDevices)
					   			 .filter(Objects::nonNull)
					   			 .flatMap(Arrays::stream)
					   			 .toArray(CLDevice[]::new);
		
		// find the fastest device
		if(deviceId == -1 && devices.length > 0) {
			deviceId = IntStream.range(0, devices.length).boxed().max((idx1, idx2) -> {
				return Integer.compare(devices[idx1].getTheoreticalSpeed(), devices[idx2].getTheoreticalSpeed());
			}).get();
		}
		
		// already setup?
    	CLCore core = activeCores.get(deviceId);
    	if(core == null) {
    		
    		// get device for the new core
    		CLDevice device = devices[deviceId];
    		if(device == null) {
    			System.out.println("Could not find device with id "+deviceId);
    			return -1;
    		}
    		
    		// create new core
			core = new CLCore(device);    		
			System.out.println("Use OpenCL device: \n"+device.toString());
			activeCores.put(deviceId, core);
    	}
    	
    	return deviceId;
	}
		
	/**
	 * OpenCL Core for a specific device. Creates a new core if there
	 * is no core for the device, otherwise returns the existing one.
	 * 
	 * the device 
	 * @param index
	 * @return
	 */
    protected static CLCore getCore(int deviceId) {
    	return activeCores.get(deviceId);
    }
 
    private cl_context context;
    private cl_command_queue commandQueue;
    
    private List<Subprogram<cl_kernel>> customSubprograms;
    private cl_program customProgram = null;
    
    private List<Subprogram<cl_kernel>> matrixSubprograms;
    private cl_program matrixProgram = null;

    private int threadCount_X;
    private int threadCount_Y;
    private int threadCount;
    private int computeUnits;
    
    private float[] sharedData;
    private CLMemory sharedBuffer;

    private CLCore(CLDevice device) {
        
        computeUnits = device.getComputeUnits();
        threadCount = device.getMaxWorkGroupSize();
        int logBlockSize = (int) Math.round(Math.log(threadCount) / Math.log(2));
        int logBlockSizeX = logBlockSize / 2;
        int logBlockSizeY = (logBlockSize % 2 == 0) ? logBlockSizeX : logBlockSizeX + 1;
        threadCount_X = (int) Math.pow(2.0, logBlockSizeX);
        threadCount_Y = (int) Math.pow(2.0, logBlockSizeY);

        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, device.getPlatform().getId());
        context = CL.clCreateContext(contextProperties, 1, new cl_device_id[]{device.getId()}, null, null, null);
        commandQueue = CL.clCreateCommandQueue(context, device.getId(), 0, null);

        customSubprograms = new ArrayList<>();
        matrixSubprograms = new ArrayList<>();
        
        sharedData = new float[computeUnits];
        sharedBuffer = mallocSinglePrecision(computeUnits);
    }



   

    public int getThreadCount() {
        return threadCount;
    }
    
    public int getThreadCountX() {
        return threadCount_X;
    }

    public int getThreadCountY() {
        return threadCount_Y;
    }

    public float[] getData(CLMemory buffer, float[] n) {
    	return getData(buffer, n, 0);
    }
    
    public float[] getData(CLMemory buffer, float[] n, int offset) {
        cl_event event = new cl_event();
        waitOnComplete();
        CL.clEnqueueReadBuffer(commandQueue, buffer.getMemory(), CL.CL_TRUE, offset, n.length * Sizeof.cl_float, Pointer.to(n), 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
        return n;
    }

    /** 
     * TODO: Baut und released mehrfach die custom Programme. Bessere unterscheidung zwischen custom und matrix funktionen
     * 
     * @param subprogram
     */
    public void loadFromGeneratedSubprogram(Subprogram<cl_kernel> subprogram) {
        if (subprogram.isCustom() == false) {
        	matrixSubprograms.add(subprogram);
        } else {
            // remove kernels to recompile;
            if (customProgram != null) {
            	
            	for (Subprogram<cl_kernel> customSubprogram : customSubprograms) {
            		CL.clReleaseKernel(customSubprogram.getKernel());
            		customSubprogram.setKernel(null);
				}
                CL.clReleaseProgram(customProgram);
                customProgram = null;
            }
            // get all kernels
            customSubprograms.add(subprogram);

            StringBuilder builder = new StringBuilder();
            for (Subprogram<cl_kernel> customSubprogram : customSubprograms)
            	builder.append(customSubprogram.getSourceCode());

            // create program source from all custom kernels
            String programSource = builder.toString();

            customProgram = CL.clCreateProgramWithSource(context, 1, new String[]{programSource}, null, null);
            CL.clBuildProgram(customProgram, 0, null, null, null, null);
            
            for (Subprogram<cl_kernel> customSubprogram : customSubprograms) {
            	String kernelName = customSubprogram.getProgramName();
            	customSubprogram.setKernel(CL.clCreateKernel(customProgram, kernelName, null));
            }
        }
    }

    public void compileMatrixFunctions() {

        // Verbinde den Sourcecode aller Kernels
        StringBuilder builder = new StringBuilder();
        for (Subprogram<cl_kernel> subprogram : matrixSubprograms) {
	        builder.append(subprogram.getSourceCode());
	        builder.append('\n');
		}

        // create program source from all kernels
        String programSource = builder.toString();
        matrixProgram = CL.clCreateProgramWithSource(context, 1, new String[]{programSource}, null, null);
        CL.clBuildProgram(matrixProgram, 0, null, null, null, null);

        for (Subprogram<cl_kernel> subprogram : matrixSubprograms) {
	        String kernelName = subprogram.getProgramName();
	        subprogram.setKernel(CL.clCreateKernel(matrixProgram, kernelName, null));
        }
    }
    
    /**
     * Kernel die einmal über die ganze Matrix laufen:
     * <i>function(__global <type> storages[0], ... , __global <type> storages[length-1])</i>
     * 
     * @param subprogram
     * @param clRows
     * @param clColumns
     * @param result
     */
    public void execute(Subprogram<cl_kernel> subprogram, int length, CLStorage ... storages) {
    	cl_kernel kernel = subprogram.getKernel();
        for (int i = 0; i < storages.length; i++) {
        	CLStorage storage = storages[i];
            CL.clSetKernelArg(kernel, i, storage.getSizeof(), storage.getPointer());
        }
        enqueue1DRangeKernel(kernel, length, 0);
    }
    
	protected void enqueue1DRangeKernel(cl_kernel kernel, int length, int offset) {

		long[] global_work_offset = new long[] { offset };
		long[] global_work_size = new long[] { length };
		long[] local_work_size = new long[] { Math.min(length, threadCount) };

		CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, global_work_offset, global_work_size, local_work_size, 0, null, null);
	}
	
    /**
     * Kernel die einmal über die ganze Matrix laufen:
     * <i>function(__global <type> storages[0], ... , __global <type> storages[length-1])</i>
     * Der Kernel wird mit clRows und clColumns großen Work-Groups ausgeführt.
     * 
     * @param subprogram
     * @param clRows
     * @param clColumns
     * @param result
     */
    public void execute(Subprogram<cl_kernel> subprogram, int clRows, int clColumns, CLStorage ... storages) {
    	cl_kernel kernel = subprogram.getKernel();
        for (int i = 0; i < storages.length; i++) {
        	CLStorage storage = storages[i];
            CL.clSetKernelArg(kernel, i, storage.getSizeof(), storage.getPointer());
        }
        enqueue2DRangeKernel(kernel, clRows, clColumns, 0, 0);
    }
    
	protected void enqueue2DRangeKernel(cl_kernel kernel, int clRows, int clColumns, int rowOffset, int columnOffset) {

		long[] global_work_offset = new long[] { columnOffset, rowOffset };
		long[] global_work_size = new long[] { clColumns, clRows };
		long[] local_work_size = new long[] { Math.min(clColumns, threadCount_X), Math.min(clRows, threadCount_Y) };

		CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, global_work_offset, global_work_size, local_work_size, 0, null, null);
	}
	
	
	public void enqueue2DRangeKernelTest(cl_kernel kernel, int clRows, int clColumns, int localRows, int localCols) {

		long[] global_work_offset = new long[] { 0,0 };
		long[] global_work_size = new long[] { clColumns, clRows };
		long[] local_work_size = new long[] { localCols, localRows };

		CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, global_work_offset, global_work_size, local_work_size, 0, null, null);
	}
	
	
    // -----------------------------------------------------------------------------------------
    // ----------------------------------- reduce methods --------------------------------------
    // -----------------------------------------------------------------------------------------
	
    private void copyColumnMajor(CLStorage data, CLStorage result, int n) {

        int size = (int) Math.ceil((double) n / threadCount) * threadCount;
//        cl_event event = new cl_event();
        cl_kernel kernel = CLPredefined.getSubprogram("copyColumnMajor").getKernel();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, data.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, result.getPointer());
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                new long[]{size},
                new long[]{threadCount}, 0, null, null);

//        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    private void copyRowMajor(CLStorage data, CLStorage result, int n, int clRows) {

        int size = (int) Math.ceil((double) n / threadCount) * threadCount;
//        cl_event event = new cl_event();
        cl_kernel kernel = CLPredefined.getSubprogram("copyRowMajor").getKernel();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, data.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, result.getPointer());
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{clRows}));
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                new long[]{size},
                new long[]{threadCount}, 0, null, null);

//        CL.clWaitForEvents(1, new cl_event[]{event});
    }
    
    @Deprecated
    public float reduce(String reductionName, CLStorage data, int n, float initValue) {

    	// alle vorherigen Operationen müssen abgeschlossen sein
    	waitOnComplete();
    	
        int tempSize = (int) Math.ceil((double) n / threadCount);
        int size = tempSize * threadCount;

        CLMemory temp = mallocSinglePrecision(tempSize);
        cl_kernel kernel = CLPredefined.getSubprogram(reductionName).getKernel();
        reduceCall(kernel, data, temp, n, initValue, size);

        while (tempSize > 1) {
            size = tempSize;
            tempSize = (int) Math.ceil((double) tempSize / threadCount);
            reduceCall(kernel, temp, temp, size, initValue, tempSize * threadCount);
        }

        float[] result = new float[1];
        getData(temp, result);
        temp.release();

        return result[0];
    }

    @Deprecated
    private void reduceCall(cl_kernel kernel, CLStorage data, CLStorage temp, int n, float initValue, int size) {
    	
    	// alle vorherigen Operationen müssen abgeschlossen sein
    	waitOnComplete();
    	
//        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, data.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, temp.getPointer());
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_float * threadCount, null);
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_float, Pointer.to(new float[]{initValue}));
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                new long[]{size},
                new long[]{threadCount}, 0, null, null);

//        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public float reduce2D(String reductionName, CLStorage data, int rows, int columns, float initValue) {
    	 
        int tempSizeX = (int) Math.ceil((double) rows / threadCount_X);
        int tempSizeY = (int) Math.ceil((double) columns / threadCount_Y);
        int sizeX = tempSizeX * threadCount_X;
        int sizeY = tempSizeY * threadCount_Y;

//        cl_mem temp = malloc(tempSizeY * tempSizeX);
        CLMemory temp = malloc(new float[tempSizeY * tempSizeX]);
        cl_kernel kernel = CLPredefined.getSubprogram(reductionName).getKernel();
        reduceCall(kernel, data, temp, rows, columns, initValue, sizeX, sizeY);

        while (tempSizeX > 1 || tempSizeY > 1) {
            sizeX = tempSizeX;
            sizeY = tempSizeY;
            tempSizeX = (int) Math.ceil((double) tempSizeX / threadCount_X);
            tempSizeY = (int) Math.ceil((double) tempSizeY / threadCount_Y);
            reduceCall(kernel, temp, temp, sizeX, sizeY, initValue, tempSizeX * threadCount_X, tempSizeY * threadCount_Y);
        }

        float[] reduced = new float[1];
        getData(temp, reduced);
        temp.release();
        
        return reduced[0];
    }
  
    
    public float sum(float[] arr) {
    	float sum = 0;
    	for (float f : arr) {
			sum+=f;
		}
    	
    	return sum;
    }

    public void reduce2D(String reductionName, CLStorage data, CLStorage result, int rows, int columns, int tempSizeX, int tempSizeY, float initValue) {

        int sizeX = tempSizeX * threadCount_X;
        int sizeY = tempSizeY * threadCount_Y;

        cl_kernel kernel = CLPredefined.getSubprogram(reductionName).getKernel();
        reduceCall(kernel, data, result, rows, columns, initValue, sizeX, sizeY);
    }

    public void reduceColumns(String reductionName, CLStorage data, CLStorage result, int rows, int columns, int clResultRows, float initValue) {

        int tempSizeX = (int) Math.ceil((double) rows / threadCount_X);
        int tempSizeY = (int) Math.ceil((double) columns / threadCount_Y);
        int sizeX = tempSizeX * threadCount_X;
        int sizeY = tempSizeY * threadCount_Y;

        CLMemory temp = mallocSinglePrecision(columns * tempSizeX);
        cl_kernel kernel = CLPredefined.getSubprogram(reductionName).getKernel();
        reduceCall(kernel, data, temp, rows, columns, initValue, sizeX, sizeY);

        while (tempSizeX > 1) {
            sizeX = tempSizeX;
            tempSizeX = (int) Math.ceil((double) tempSizeX / threadCount_X);
            reduceCall(kernel, temp, temp, sizeX, columns, initValue, tempSizeX * threadCount_X, sizeY);
        }
        
        copyRowMajor(temp, result, columns, clResultRows);
        temp.release();
    }


    public void reduceRows(String reductionName, CLStorage data, CLStorage result, int rows, int columns, float initValue) {

        int tempSizeX = (int) Math.ceil((double) rows / threadCount_X);
        int tempSizeY = (int) Math.ceil((double) columns / threadCount_Y);
        int sizeX = tempSizeX * threadCount_X;
        int sizeY = tempSizeY * threadCount_Y;

        CLMemory temp = mallocSinglePrecision(rows * tempSizeY);
        cl_kernel kernel = CLPredefined.getSubprogram(reductionName).getKernel();
        reduceCall(kernel, data, temp, rows, columns, initValue, sizeX, sizeY);

        while (tempSizeY > 1) {
            sizeY = tempSizeY;
            tempSizeY = (int) Math.ceil((double) tempSizeY / threadCount_Y);
            reduceCall(kernel, temp, temp, rows, sizeY, initValue, sizeX, tempSizeY * threadCount_Y);
        }
   
        copyColumnMajor(temp, result, rows);
        temp.release();
    }

    private void reduceCall(cl_kernel kernel, CLStorage data, CLStorage result, int rows, int columns, float initValue, int sizeX, int sizeY) {
    	
//        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, data.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, result.getPointer());
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_float * threadCount, null);
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{columns}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_float, Pointer.to(new float[]{initValue}));
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null,
                new long[]{sizeX, sizeY},
                new long[]{threadCount_X, threadCount_Y}, 0, null, null);

//        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    
    public float reduce1D(String reductionName, CLStorage data, int dataLength) {

    	cl_kernel kernel = CLPredefined.getSubprogram(reductionName).getKernel();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, data.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, sharedBuffer.getPointer());
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_float * threadCount, null);
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{dataLength}));
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                new long[]{computeUnits * threadCount},
                new long[]{threadCount}, 0, null, null);

        getData(sharedBuffer, sharedData);
        return sum(sharedData);
    }
    
    
    
    // -----------------------------------------------------------------------------------------
    // ----------------------------- allocation and release methods ----------------------------
    // -----------------------------------------------------------------------------------------
    public CLMemory malloc(float[] values) {    
        return malloc(Pointer.to(values), Sizeof.cl_float, values.length);
    }

    public CLMemory malloc(int[] values) {
        return malloc(Pointer.to(values), Sizeof.cl_uint4, values.length);
    }
    
    /**
     * allocate and fill the memory with the data in the pointer
     *  
     * @param pointer
     * @param size in byte
     * @return
     */
    public CLMemory malloc(Pointer pointer, int sizeof, int length) {

        cl_mem memory = CL.clCreateBuffer(context,
                CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR,
                sizeof * length, pointer, null);
        return new CLMemory(memory, sizeof, length);
    }

    public CLMemory mallocSinglePrecision(int length) {
    	cl_mem memory = CL.clCreateBuffer(context,
                CL.CL_MEM_READ_WRITE,
                Sizeof.cl_float * length, null, null);
        return new CLMemory(memory, Sizeof.cl_float, length);
    }
       
    /**
     * wait for all previous tasks to complete
     */
    public void waitOnComplete() {
    	CL.clFinish(commandQueue);
    } 
}
