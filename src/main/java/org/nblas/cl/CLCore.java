package org.nblas.cl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

import org.jocl.CL;
import org.jocl.CLException;
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
 * Matrix funktionen wie transpose() gehören in eine allgemeine CLMatrix Klasse und sgemm_nn sogar in die CLFloatMatrix.
 *   
 * @author Nico
 *
 */
class CLCore {

    private static final CLCore CORE = new CLCore();

    public static CLCore getCore() {
        return CORE;
    }
    
    
    
    private CLDevice device;
    
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

    private CLCore() {

        device = setupDevice();
        
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



    private CLDevice setupDevice() {
    	
    	CL.setExceptionsEnabled(true);
    	
    	CLPlatform[] platforms = CLPlatform.getPlatforms();
        if (platforms.length == 0) {
            throw new CLException("No OpenCL-Device found!\n " +
                    "Please reconsider that all OpenCL-Drivers and OpenCL-Platforms are installed properly.");
        }
        
    	final Comparator<CLDevice> performanceComperator = (c1, c2) -> Integer.compare( c1.getTheoreticalComputingPower(), c2.getTheoreticalComputingPower());
    	CLDevice fastestDevice = Arrays.stream(platforms).map(CLPlatform::getFastestGPU).filter(Objects::nonNull).max(performanceComperator).get();
        System.out.println("Use OpenCL device: \n"+fastestDevice.toString());
        
        return fastestDevice;
	}

    public int getThreadCountX() {
        return threadCount_X;
    }

    public int getThreadCountY() {
        return threadCount_Y;
    }

    public void copy(CLStorage input, CLStorage copy, int clRows, int clColumns) {
        cl_kernel kernel = CLPredefined.getSubprogram("copy").getKernel();

        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, input.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, copy.getPointer());

        enqueue2DRangeKernel(kernel, clRows, clColumns, 0, 0);
    }

    public void repmat(CLStorage source, CLStorage result, int clRows, int clColumns, int outputRows, int outputColumns, int inputRows, int inputColumns, int stride) {
        cl_kernel kernel = CLPredefined.getSubprogram("repmat").getKernel();
        
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, source.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, result.getPointer());
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, Pointer.to(new int[]{outputRows}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{outputColumns}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{inputRows}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{inputColumns}));
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_uint, Pointer.to(new int[]{stride}));

        enqueue2DRangeKernel(kernel, clRows, clColumns, 0, 0);
    }

    
    public void setSubMatrix(CLStorage source, CLStorage destination, int srcCLRows, int srcCLColumns, int srcRows, int srcColumns, int offsetRows, int offsetColumns, int dstStride) {
    	cl_kernel kernel = CLPredefined.getSubprogram("setSubMatrix").getKernel();

        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, source.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, destination.getPointer());
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, Pointer.to(new int[]{srcRows}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{srcColumns}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{offsetRows}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{offsetColumns}));
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_uint, Pointer.to(new int[]{dstStride}));

        enqueue2DRangeKernel(kernel, srcCLRows, srcCLColumns, 0, 0);
    }
    

    public void getSubMatrix(CLStorage source, CLStorage destination, int dstCLRows, int dstCLColumns, int dstRows, int dstColumns, int offsetRows, int offsetColumns, int srcStride) {
    	cl_kernel kernel = CLPredefined.getSubprogram("getSubMatrix").getKernel();
    	
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, source.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, destination.getPointer());
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, Pointer.to(new int[]{offsetRows+dstRows}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{offsetColumns+dstColumns}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{offsetRows}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{offsetColumns}));
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_uint, Pointer.to(new int[]{srcStride}));

        enqueue2DRangeKernel(kernel, dstCLRows, dstCLColumns, 0, 0);
    }
    
    public void getCustom(Subprogram<cl_kernel> subprogram,CLStorage source, CLStorage destination, int dstCLRows, int dstCLColumns, int dstRows, int dstColumns, int offsetRows, int offsetColumns, int srcStride) {
    	cl_kernel kernel = subprogram.getKernel();
    	  
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, source.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, destination.getPointer());
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, Pointer.to(new int[]{offsetRows+dstRows}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{offsetColumns+dstColumns}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{offsetRows}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{offsetColumns}));
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_uint, Pointer.to(new int[]{srcStride}));

        enqueue2DRangeKernel(kernel, dstCLRows, dstCLColumns, 0, 0);
    }
    
    public void transpose(CLStorage in, CLStorage out, int clRows, int clColumns, int rows, int columns) {
        cl_kernel kernel = CLPredefined.getSubprogram("transpose").getKernel();

//        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, in.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, out.getPointer());
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_float * threadCount, null);
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{columns}));

        enqueue2DRangeKernel(kernel, clRows, clColumns, 0, 0);
//        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{clRows, clColumns}, new long[]{threadCount_X, threadCount_Y}, 0, null, null);
//        CL.clWaitForEvents(1, new cl_event[]{event});
    }
    
    
    public void sgemm_nn_custom(Subprogram<cl_kernel> subprogram, CLStorage a, CLStorage b, CLStorage result, int clM, int clN, int clK) {
    	cl_kernel kernel = subprogram.getKernel();
    	 
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, a.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, b.getPointer());
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, result.getPointer());
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_float * threadCount, null);
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_float * threadCount, null);
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{clM}));
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_int, Pointer.to(new int[]{clN}));
        CL.clSetKernelArg(kernel, 7, Sizeof.cl_int, Pointer.to(new int[]{clK}));

        enqueue2DRangeKernel(kernel, clM, clN, 0, 0);
    }

    public void sgemm_nn(CLStorage a, CLStorage b, CLStorage result, int clM, int clN, int clK) {
        Subprogram<cl_kernel> subprogram = CLPredefined.getSubprogram("sgemm_nn");
        sgemmCall(a, b, result, clM, clN, clK, subprogram.getKernel());
    }

    public void sgemm_nt(CLStorage a, CLStorage b, CLStorage result, int clM, int clN, int clK) {
        Subprogram<cl_kernel> subprogram = CLPredefined.getSubprogram("sgemm_nt");
        sgemmCall(a, b, result, clM, clN, clK, subprogram.getKernel());
    }

    public void sgemm_tn(CLStorage a, CLStorage b, CLStorage result, int clM, int clN, int clK) {
        Subprogram<cl_kernel> subprogram = CLPredefined.getSubprogram("sgemm_tn");
        sgemmCall(a, b, result, clM, clN, clK, subprogram.getKernel());
    }


    private void sgemmCall(CLStorage a, CLStorage b, CLStorage result, int clM, int clN, int clK, cl_kernel kernel) {
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, a.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, b.getPointer());
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, result.getPointer());
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_float * threadCount, null);
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_float * threadCount, null);
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{clM}));
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_int, Pointer.to(new int[]{clN}));
        CL.clSetKernelArg(kernel, 7, Sizeof.cl_int, Pointer.to(new int[]{clK}));

        enqueue2DRangeKernel(kernel, clM, clN, 0, 0);
    }

    public void boxMuller(CLStorage dataPointer, CLStorage random, int clRows, int clColumns, int rows, int columns) {
        cl_kernel kernel = CLPredefined.getSubprogram("boxmuller").getKernel();
        random(dataPointer, random, clRows, rows, columns, kernel);
    }

    public void uniform(CLStorage dataPointer, CLStorage random, int clRows, int clColumns, int rows, int columns) {
        cl_kernel kernel = CLPredefined.getSubprogram("auniform").getKernel();
        random(dataPointer, random, clRows, rows, columns, kernel);
    }

    private void random(CLStorage dataPointer, CLStorage random, int clRows, int rows, int columns, cl_kernel kernel) {
//        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, random.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, dataPointer.getPointer());
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{clRows}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{columns}));
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{threadCount_X, threadCount_Y},
                new long[]{threadCount_X, threadCount_Y}, 0, null, null);
//        CL.clWaitForEvents(1, new cl_event[]{event});
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

    
    /**
     * Kernel die einmal über die ganze Matrix laufen:
     * <i>function(__global <type> storages[0], ... , __global <type> storages[length-1])</i>
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
	
	
	
    // -----------------------------------------------------------------------------------------
    // ----------------------------------- reduce methods --------------------------------------
    // -----------------------------------------------------------------------------------------
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
     * Release the memory from the device
     * 
     * @param memory
     */
    public void release(cl_mem memory) {
		CL.clReleaseMemObject(memory);
	}
    
    /**
     * wait for all previous tasks to complete
     */
    public void waitOnComplete() {
    	CL.clFinish(commandQueue);
    } 
}
