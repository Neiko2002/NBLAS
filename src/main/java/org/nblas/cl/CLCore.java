package org.nblas.cl;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.jocl.CL;
import org.jocl.CLException;
import org.jocl.NativePointerObject;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_event;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;
import org.nblas.generic.Subprogram;

class CLCore {

    private static final CLCore CORE = new CLCore();
    private Field nativePointerField;
    private CLPlatform platform;
    private CLDevice device;
    
    private cl_context_properties contextProperties;
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
    private cl_mem sharedBuffer;

    private CLCore() {

        setup();
        
        computeUnits = device.getComputeUnits();
        threadCount = device.getMaxWorkGroupSize();
        int logBlockSize = (int) Math.round(Math.log(threadCount) / Math.log(2));
        int logBlockSizeX = logBlockSize / 2;
        int logBlockSizeY = (logBlockSize % 2 == 0) ? logBlockSizeX : logBlockSizeX + 1;
        threadCount_X = (int) Math.pow(2.0, logBlockSizeX);
        threadCount_Y = (int) Math.pow(2.0, logBlockSizeY);

        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platform.getId());
        context = CL.clCreateContext(contextProperties, 1, new cl_device_id[]{device.getId()}, null, null, null);
        commandQueue = CL.clCreateCommandQueue(context, device.getId(), 0, null);

        customSubprograms = new ArrayList<>();
        matrixSubprograms = new ArrayList<>();
        
        try {
            this.nativePointerField = NativePointerObject.class.getDeclaredField("nativePointer");
            this.nativePointerField.setAccessible(true);
//            clblas = new CLBLAS(getNativePointer(this.platform), getNativePointer(this.device), getNativePointer(this.context), getNativePointer(this.commandQueue));
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        // lade alle Predefined Kernels
        for (Subprogram<cl_kernel> subprogram : CLPredefined.getAllSubPrograms()) {
        	loadFromGeneratedSubprogram(subprogram);
        }
        

        sharedData = new float[computeUnits];
        sharedBuffer = malloc(computeUnits);
    }



    private void setup() {
    	
    	CL.setExceptionsEnabled(true);
    	
    	CLPlatform[] platforms = CLPlatform.getPlatforms();        
        if (platforms.length == 0) {
            throw new CLException("No OpenCL-Device found!\n " +
                    "Please reconsider that all OpenCL-Drivers and OpenCL-Platforms are installed properly.");
        }
        
    	final Comparator<CLDevice> performanceComperator = (p1, p2) -> Integer.compare( p1.getTheoreticalComputingPower(), p2.getTheoreticalComputingPower());
    	CLDevice fastestDevice = Arrays.stream(platforms).map(CLPlatform::getFastestGPU).max(performanceComperator).get();
        
        this.device = fastestDevice;
        this.platform = fastestDevice.getPlatform();
        System.out.println("Use device: \n"+this.device.toString());    	
	}



	public long getNativePointer(NativePointerObject nativePointerObject) throws IllegalAccessException {
        return nativePointerField.getLong(nativePointerObject);
    }

    public static CLCore getCore() {
        return CORE;
    }

    public int getThreadCount_X() {
        return threadCount_X;
    }

    public int getThreadCount_Y() {
        return threadCount_Y;
    }

    public void copy(cl_mem input, cl_mem copy, int m, int n) {

        cl_kernel kernel = CLPredefined.getSubprogram("copy").getKernel();
//        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(input));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(copy));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, null);
//        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void repmat(cl_mem source, cl_mem result, int m, int n, int outputRows, int outputColumns, int inputRows, int inputColumns, int stride) {

        cl_kernel kernel = CLPredefined.getSubprogram("repmat").getKernel();
//        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(source));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(result));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, Pointer.to(new int[]{outputRows}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{outputColumns}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{inputRows}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{inputColumns}));
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_uint, Pointer.to(new int[]{stride}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, null);
//        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void setSubMatrix(cl_mem source, cl_mem result, int m, int n, int sourceRows, int sourceColumns, int offsetRows, int offsetColumns, int resultStride) {

        cl_kernel kernel = CLPredefined.getSubprogram("setSubMatrix").getKernel();
//        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(source));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(result));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, Pointer.to(new int[]{sourceRows}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{sourceColumns}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{offsetRows}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{offsetColumns}));
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_uint, Pointer.to(new int[]{resultStride}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, null);
//        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void getSubMatrix(cl_mem source, cl_mem result, int m, int n, int resultRows, int resultColumns, int offsetRows, int offsetColumns, int sourceStride) {

        cl_kernel kernel = CLPredefined.getSubprogram("getSubMatrix").getKernel();
//        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(source));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(result));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, Pointer.to(new int[]{resultRows}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{resultColumns}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{offsetRows}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{offsetColumns}));
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_uint, Pointer.to(new int[]{sourceStride}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, null);
//        CL.clWaitForEvents(1, new cl_event[]{event});
    }


    public void transpose(cl_mem in, cl_mem out, int clRows, int clColumns, int rows, int columns) {
        cl_kernel kernel = CLPredefined.getSubprogram("transpose").getKernel();

//        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(in));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(out));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_float * threadCount, null);
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{columns}));


        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{clRows, clColumns}, new long[]{threadCount_X, threadCount_Y}, 0, null, null);
//        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void sgemm_nn(cl_mem a, cl_mem b, cl_mem result, int clM, int clN, int clK) {
        Subprogram<cl_kernel> subprogram = CLPredefined.getSubprogram("sgemm_nn");
        if(subprogram.isBuild())
        	sgemmCall(a, b, result, clM, clN, clK, subprogram.getKernel());
    }

    public void sgemm_nt(cl_mem a, cl_mem b, cl_mem result, int clM, int clN, int clK) {
        Subprogram<cl_kernel> subprogram = CLPredefined.getSubprogram("sgemm_nt");
        if(subprogram.isBuild())
        	sgemmCall(a, b, result, clM, clN, clK, subprogram.getKernel());
    }

    public void sgemm_tn(cl_mem a, cl_mem b, cl_mem result, int clM, int clN, int clK) {
        Subprogram<cl_kernel> subprogram = CLPredefined.getSubprogram("sgemm_tn");
        if(subprogram.isBuild())
        	sgemmCall(a, b, result, clM, clN, clK, subprogram.getKernel());
    }


    private void sgemmCall(cl_mem a, cl_mem b, cl_mem result, int clM, int clN, int clK, cl_kernel kernel) {
//        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(result));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_float * threadCount, null);
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_float * threadCount, null);
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{clM}));
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_int, Pointer.to(new int[]{clN}));
        CL.clSetKernelArg(kernel, 7, Sizeof.cl_int, Pointer.to(new int[]{clK}));

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{clM, clN}, new long[]{threadCount_X, threadCount_Y}, 0, null, null);
//        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void boxMuller(cl_mem dataPointer, cl_mem random, int clRows, int clColumns, int rows, int columns) {
        cl_kernel kernel = CLPredefined.getSubprogram("boxmuller").getKernel();
        random(dataPointer, random, clRows, rows, columns, kernel);
    }

    public void uniform(cl_mem dataPointer, cl_mem random, int clRows, int clColumns, int rows, int columns) {
        cl_kernel kernel = CLPredefined.getSubprogram("auniform").getKernel();
        random(dataPointer, random, clRows, rows, columns, kernel);
    }

    private void random(cl_mem dataPointer, cl_mem random, int clRows, int rows, int columns, cl_kernel kernel) {
//        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(random));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(dataPointer));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{clRows}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{columns}));
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{threadCount_X, threadCount_Y},
                new long[]{threadCount_X, threadCount_Y}, 0, null, null);
//        CL.clWaitForEvents(1, new cl_event[]{event});
    }


    public String loadSource(String path) {
        try {
            return new String(Files.readAllBytes(Paths.get(path)));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    private long[] getLocalWorkSize(int m, int n) {
        long[] local_work_size = {threadCount_X, threadCount_Y};
        if (m == 1) {
            local_work_size[0] = 1;
        }
        if (n == 1) {
            local_work_size[1] = 1;
        }
        return local_work_size;
    }

    public cl_context getContext() {
        return context;
    }


    public float[] getData(cl_mem buffer, float[] n) {
        Pointer pointer = Pointer.to(n);
        cl_event event = new cl_event();
        waitOnComplete();
        CL.clEnqueueReadBuffer(commandQueue, buffer, CL.CL_TRUE, 0, n.length * Sizeof.cl_float, pointer, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
        return n;
    }

    public void loadFromGeneratedSubprogram(Subprogram<cl_kernel> subprogram) {
        if (subprogram.isCustom() == false) {
        	matrixSubprograms.add(subprogram);
        } else {
            // remove kernels to recompile;
            if (customSubprograms != null) {
            	
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

            customProgram = CL.clCreateProgramWithSource(context,
                    1, new String[]{programSource}, null, null);
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
        int result = CL.clBuildProgram(matrixProgram, 0, null, null, null, null);

        for (Subprogram<cl_kernel> subprogram : matrixSubprograms) {
	        String kernelName = subprogram.getProgramName();
	        subprogram.setKernel(CL.clCreateKernel(matrixProgram, kernelName, null));
        }
    }
    
    

    private void copy1d(cl_mem data, cl_mem result, int n) {

        int size = (int) Math.ceil((double) n / threadCount) * threadCount;
//        cl_event event = new cl_event();
        cl_kernel kernel = CLPredefined.getSubprogram("copy1D").getKernel();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(data));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(result));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                new long[]{size},
                new long[]{threadCount}, 0, null, null);

//        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    
    public void execute(Subprogram<cl_kernel> subprogram, int clRows, int clColumns, int rows, int columns, cl_mem result, cl_mem... dataPointer) {
        cl_kernel kernel = subprogram.getKernel();
//        cl_event event = new cl_event();
        
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(result));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[]{columns}));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{rows}));
        for (int i = 0; i < dataPointer.length; i++) {
            CL.clSetKernelArg(kernel, i + 3, Sizeof.cl_mem, Pointer.to(dataPointer[i]));
        }
        
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null,
                new long[]{clRows, clColumns},
                new long[]{threadCount_X, threadCount_Y}, 0, null, null);
//        CL.clWaitForEvents(1, new cl_event[]{event});
    }
    
    public void execute(Subprogram<cl_kernel> subprogram, int clRows, int clColumns, cl_mem result) {
    	cl_kernel kernel = subprogram.getKernel();
//    	cl_event event = new cl_event();
        
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(result));
        
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null,
                new long[]{clRows, clColumns},
                new long[]{threadCount_X, threadCount_Y}, 0, null, null);
//        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    @Deprecated
    public float reduce(String reductionName, cl_mem data, int n, float initValue) {

    	// alle vorherigen Operationen müssen abgeschlossen sein
    	waitOnComplete();
    	
        int tempSize = (int) Math.ceil((double) n / threadCount);
        int size = tempSize * threadCount;

        cl_mem temp = malloc(tempSize);
        cl_kernel kernel = CLPredefined.getSubprogram(reductionName).getKernel();
        reduceCall(kernel, data, temp, n, initValue, size);

        while (tempSize > 1) {
            size = tempSize;
            tempSize = (int) Math.ceil((double) tempSize / threadCount);
            reduceCall(kernel, temp, temp, size, initValue, tempSize * threadCount);
        }

        float[] result = new float[1];
        getData(temp, result);
        free(temp);

        return result[0];
    }

    @Deprecated
    private void reduceCall(cl_kernel kernel, cl_mem data, cl_mem temp, int n, float initValue, int size) {
    	
    	// alle vorherigen Operationen müssen abgeschlossen sein
    	waitOnComplete();
    	
//        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(data));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(temp));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_float * threadCount, null);
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_float, Pointer.to(new float[]{initValue}));
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                new long[]{size},
                new long[]{threadCount}, 0, null, null);

//        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public float reduce2D(String reductionName, cl_mem data, int rows, int columns, float initValue) {
    	 
        int tempSizeX = (int) Math.ceil((double) rows / threadCount_X);
        int tempSizeY = (int) Math.ceil((double) columns / threadCount_Y);
        int sizeX = tempSizeX * threadCount_X;
        int sizeY = tempSizeY * threadCount_Y;

//        cl_mem temp = malloc(tempSizeY * tempSizeX);
        cl_mem temp = malloc(new float[tempSizeY * tempSizeX]);
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
        free(temp);
        
        return reduced[0];
    }
  
    
    public float sum(float[] arr) {
    	float sum = 0;
    	for (float f : arr) {
			sum+=f;
		}
    	
    	return sum;
    }

    public void reduce2D(String reductionName, cl_mem data, cl_mem result, int rows, int columns, int tempSizeX, int tempSizeY, float initValue) {

        int sizeX = tempSizeX * threadCount_X;
        int sizeY = tempSizeY * threadCount_Y;

        cl_kernel kernel = CLPredefined.getSubprogram(reductionName).getKernel();
        reduceCall(kernel, data, result, rows, columns, initValue, sizeX, sizeY);
    }

    public void reduceColumns(String reductionName, cl_mem data, cl_mem result, int rows, int columns, float initValue) {

        int tempSizeX = (int) Math.ceil((double) rows / threadCount_X);
        int tempSizeY = (int) Math.ceil((double) columns / threadCount_Y);
        int sizeX = tempSizeX * threadCount_X;
        int sizeY = tempSizeY * threadCount_Y;

        cl_mem temp = malloc(columns * tempSizeX);
        cl_kernel kernel = CLPredefined.getSubprogram(reductionName).getKernel();
        reduceCall(kernel, data, temp, rows, columns, initValue, sizeX, sizeY);

        while (tempSizeX > 1) {
            sizeX = tempSizeX;
            tempSizeX = (int) Math.ceil((double) tempSizeX / threadCount_X);
            reduceCall(kernel, temp, temp, sizeX, columns, initValue, tempSizeX * threadCount_X, sizeY);
        }
        copy1d(temp, result, columns);
        free(temp);

    }


    public void reduceRows(String reductionName, cl_mem data, cl_mem result, int rows, int columns, float initValue) {

        int tempSizeX = (int) Math.ceil((double) rows / threadCount_X);
        int tempSizeY = (int) Math.ceil((double) columns / threadCount_Y);
        int sizeX = tempSizeX * threadCount_X;
        int sizeY = tempSizeY * threadCount_Y;

        cl_mem temp = malloc(rows * tempSizeY);
        cl_kernel kernel = CLPredefined.getSubprogram(reductionName).getKernel();
        reduceCall(kernel, data, temp, rows, columns, initValue, sizeX, sizeY);

        while (tempSizeY > 1) {
            sizeY = tempSizeY;
            tempSizeY = (int) Math.ceil((double) tempSizeY / threadCount_Y);
            reduceCall(kernel, temp, temp, rows, sizeY, initValue, sizeX, tempSizeY * threadCount_Y);
        }
        copy1d(temp, result, rows);
        free(temp);
    }

    private void reduceCall(cl_kernel kernel, cl_mem data, cl_mem result, int rows, int columns, float initValue, int sizeX, int sizeY) {
    	
//        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(data));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(result));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_float * threadCount, null);
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{columns}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_float, Pointer.to(new float[]{initValue}));
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null,
                new long[]{sizeX, sizeY},
                new long[]{threadCount_X, threadCount_Y}, 0, null, null);

//        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    
    public float reduce1D(String reductionName, cl_mem data, int dataLength) {

    	cl_kernel kernel = CLPredefined.getSubprogram(reductionName).getKernel();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(data));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(sharedBuffer));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_float * threadCount, null);
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{dataLength}));
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                new long[]{computeUnits * threadCount},
                new long[]{threadCount}, 0, null, null);

        getData(sharedBuffer, sharedData);
        return sum(sharedData);
    }
    
    public cl_mem malloc(float[] values) {

        Pointer pointer = Pointer.to(values);

        cl_mem cl_mem = CL.clCreateBuffer(getContext(),
                CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * values.length, pointer, null);
        return cl_mem;
    }


    public cl_mem malloc(int[] values) {

        Pointer pointer = Pointer.to(values);

        cl_mem cl_mem = CL.clCreateBuffer(getContext(),
                CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_uint4 * values.length / 4, pointer, null);
        return cl_mem;
    }

    public cl_mem malloc(int length) {
        return CL.clCreateBuffer(getContext(),
                CL.CL_MEM_READ_WRITE,
                Sizeof.cl_float * length, null, null);
    }

    public void free(cl_mem buffer) {
        CL.clReleaseMemObject(buffer);
    }  
    
    public void waitOnComplete() {
    	CL.clFinish(commandQueue);
    } 
  

}
