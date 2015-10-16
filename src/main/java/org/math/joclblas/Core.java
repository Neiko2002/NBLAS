package org.math.joclblas;

import org.jocl.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;

import static org.jocl.CL.*;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateKernel;

class Core {

    private static final Core CORE = new Core();
    private long deviceType;
    private cl_context_properties contextProperties;
    private cl_platform_id platform;
    private cl_device_id device;
    private cl_context context;
    private cl_command_queue commandQueue;
    private cl_program matrixProgram;
    private cl_program randomProgram;
    private cl_program setgetProgram;
    private cl_program customProgram = null;
    private HashMap<String, cl_kernel> matrixKernels;
    private HashMap<String, cl_kernel> randomKernels;
    private HashMap<String, cl_kernel> setgetKernels;
    private HashMap<String, cl_kernel> customKernels;

    private HashMap<String, String> customKernelsSource;

    private int blockSizeX;
    private int blockSizeY;
    private int blockSize;

    private Core() {

        CL.setExceptionsEnabled(true);
        deviceType = CL_DEVICE_TYPE_GPU;
        if (!getFastestDevice(deviceType)) {
            deviceType = CL_DEVICE_TYPE_CPU;
            if (!getFastestDevice(deviceType)) {
                deviceType = CL_DEVICE_TYPE_ALL;
                if (!getFastestDevice(deviceType)) {
                    throw new CLException("No OpenCL-Device found!\n " +
                            "Please reconsider that all OpenCL-Drivers and OpenCL-Platforms are installed properly.");
                }
            }
        }

        // CL_DEVICE_NAME
        String deviceName = getString(device, CL_DEVICE_NAME);
        System.out.printf("Using Device: \t%s\n", deviceName);


        blockSize = (int) getSize(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
        int logBlockSize = (int) Math.round(Math.log(blockSize) / Math.log(2));
        int logBlockSizeX = logBlockSize / 2;
        int logBlockSizeY = (logBlockSize % 2 == 0) ? logBlockSizeX : logBlockSizeX + 1;
        blockSizeX = (int) Math.pow(2.0, logBlockSizeX);
        blockSizeY = (int) Math.pow(2.0, logBlockSizeY);


        contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
        context = clCreateContext(
                contextProperties, 1, new cl_device_id[]{device},
                null, null, null);
        commandQueue = clCreateCommandQueue(context, device, 0, null);


        matrixProgram = clCreateProgramWithSource(context,
                2, new String[]{loadSource("cl/matrix.cl"), loadSource("cl/arithmetic.cl")}, null, null);
        randomProgram = clCreateProgramWithSource(context,
                1, new String[]{loadSource("cl/random.cl")}, null, null);
        setgetProgram = clCreateProgramWithSource(context,
                1, new String[]{loadSource("cl/setget.cl")}, null, null);

        clBuildProgram(matrixProgram, 0, null, null, null, null);
        clBuildProgram(randomProgram, 0, null, null, null, null);
        clBuildProgram(setgetProgram, 0, null, null, null, null);


        customKernelsSource = new HashMap<>();

        matrixKernels = new HashMap<>();
        customKernels = new HashMap<>();
        randomKernels = new HashMap<>();
        setgetKernels = new HashMap<>();

        // arithmetic kernels
        matrixKernels.put("addMatrix", clCreateKernel(matrixProgram, "addMatrix", null));
        matrixKernels.put("addScalar", clCreateKernel(matrixProgram, "addScalar", null));
        matrixKernels.put("addColumnVector", clCreateKernel(matrixProgram, "addColumnVector", null));
        matrixKernels.put("addRowVector", clCreateKernel(matrixProgram, "addRowVector", null));


        matrixKernels.put("mulMatrix", clCreateKernel(matrixProgram, "mulMatrix", null));
        matrixKernels.put("mulScalar", clCreateKernel(matrixProgram, "mulScalar", null));
        matrixKernels.put("mulColumnVector", clCreateKernel(matrixProgram, "mulColumnVector", null));
        matrixKernels.put("mulRowVector", clCreateKernel(matrixProgram, "mulRowVector", null));


        matrixKernels.put("subMatrix", clCreateKernel(matrixProgram, "subMatrix", null));
        matrixKernels.put("subScalar", clCreateKernel(matrixProgram, "subScalar", null));
        matrixKernels.put("subColumnVector", clCreateKernel(matrixProgram, "subColumnVector", null));
        matrixKernels.put("subRowVector", clCreateKernel(matrixProgram, "subRowVector", null));

        matrixKernels.put("rsubScalar", clCreateKernel(matrixProgram, "rsubScalar", null));
        matrixKernels.put("rsubColumnVector", clCreateKernel(matrixProgram, "rsubColumnVector", null));
        matrixKernels.put("rsubRowVector", clCreateKernel(matrixProgram, "rsubRowVector", null));


        matrixKernels.put("divMatrix", clCreateKernel(matrixProgram, "divMatrix", null));
        matrixKernels.put("divScalar", clCreateKernel(matrixProgram, "divScalar", null));
        matrixKernels.put("divColumnVector", clCreateKernel(matrixProgram, "divColumnVector", null));
        matrixKernels.put("divRowVector", clCreateKernel(matrixProgram, "divRowVector", null));

        matrixKernels.put("rdivScalar", clCreateKernel(matrixProgram, "rdivScalar", null));
        matrixKernels.put("rdivColumnVector", clCreateKernel(matrixProgram, "rdivColumnVector", null));
        matrixKernels.put("rdivRowVector", clCreateKernel(matrixProgram, "rdivRowVector", null));


        // matrix and reduce
        matrixKernels.put("sgemm", clCreateKernel(matrixProgram, "sgemm", null));
        matrixKernels.put("transpose", clCreateKernel(matrixProgram, "transpose", null));
        matrixKernels.put("sum", clCreateKernel(matrixProgram, "sum", null));

        // random
        randomKernels.put("xorshift", clCreateKernel(randomProgram, "xorshift", null));
        randomKernels.put("uniform", clCreateKernel(randomProgram, "uniform", null));
        randomKernels.put("boxmuller", clCreateKernel(randomProgram, "boxmuller", null));

        // setter and getter
        setgetKernels.put("copy", clCreateKernel(setgetProgram, "copy", null));
        setgetKernels.put("setZero", clCreateKernel(setgetProgram, "setZero", null));
        setgetKernels.put("setOne", clCreateKernel(setgetProgram, "setOne", null));
        setgetKernels.put("repmat", clCreateKernel(setgetProgram, "repmat", null));
        setgetKernels.put("setSubMatrix", clCreateKernel(setgetProgram, "setSubMatrix", null));
        setgetKernels.put("getSubMatrix", clCreateKernel(setgetProgram, "getSubMatrix", null));

    }

    public static Core getInstance() {
        return CORE;
    }


    public void add(cl_mem a, cl_mem b, cl_mem c, int m, int n) {

        cl_kernel kernel = matrixKernels.get("addMatrix");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));

        long[] local_work_size = getLocalWorkSize(m, n);
        long workSize = local_work_size[0] * local_work_size[1];

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{m * n}, new long[]{workSize}, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void addColumnVector(cl_mem a, cl_mem b, cl_mem c, int m, int n) {

        cl_kernel kernel = matrixKernels.get("addColumnVector");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{m}));

        long[] local_work_size = getLocalWorkSize(m, n);
        long workSize = local_work_size[0] * local_work_size[1];

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{m * n}, new long[]{workSize}, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void addRowVector(cl_mem a, cl_mem b, cl_mem c, int m, int n) {

        cl_kernel kernel = matrixKernels.get("addRowVector");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{m}));

        long[] local_work_size = getLocalWorkSize(m, n);
        long workSize = local_work_size[0] * local_work_size[1];

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{m * n}, new long[]{workSize}, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }


    public void addScalar(cl_mem a, float b, cl_mem c, int m, int n, int rows, int columns) {

        cl_kernel kernel = matrixKernels.get("addScalar");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[]{b}));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{columns}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }


    public void mul(cl_mem a, cl_mem b, cl_mem c, int m, int n) {

        cl_kernel kernel = matrixKernels.get("mulMatrix");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));

        long[] local_work_size = getLocalWorkSize(m, n);
        long workSize = local_work_size[0] * local_work_size[1];

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{m * n}, new long[]{workSize}, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void mulColumnVector(cl_mem a, cl_mem b, cl_mem c, int m, int n) {

        cl_kernel kernel = matrixKernels.get("mulColumnVector");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{m}));

        long[] local_work_size = getLocalWorkSize(m, n);
        long workSize = local_work_size[0] * local_work_size[1];

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{m * n}, new long[]{workSize}, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void mulRowVector(cl_mem a, cl_mem b, cl_mem c, int m, int n) {

        cl_kernel kernel = matrixKernels.get("mulRowVector");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{m}));

        long[] local_work_size = getLocalWorkSize(m, n);
        long workSize = local_work_size[0] * local_work_size[1];

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{m * n}, new long[]{workSize}, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void mulScalar(cl_mem a, float b, cl_mem c, int m, int n) {

        cl_kernel kernel = matrixKernels.get("mulScalar");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[]{b}));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));

        long[] local_work_size = getLocalWorkSize(m, n);
        long workSize = local_work_size[0] * local_work_size[1];

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{m * n}, new long[]{workSize}, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }


    public void sub(cl_mem a, cl_mem b, cl_mem c, int m, int n) {

        cl_kernel kernel = matrixKernels.get("subMatrix");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));

        long[] local_work_size = getLocalWorkSize(m, n);
        long workSize = local_work_size[0] * local_work_size[1];

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{m * n}, new long[]{workSize}, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void subColumnVector(cl_mem a, cl_mem b, cl_mem c, int m, int n) {

        cl_kernel kernel = matrixKernels.get("subColumnVector");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{m}));

        long[] local_work_size = getLocalWorkSize(m, n);
        long workSize = local_work_size[0] * local_work_size[1];

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{m * n}, new long[]{workSize}, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void subRowVector(cl_mem a, cl_mem b, cl_mem c, int m, int n) {

        cl_kernel kernel = matrixKernels.get("subRowVector");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{m}));

        long[] local_work_size = getLocalWorkSize(m, n);
        long workSize = local_work_size[0] * local_work_size[1];

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{m * n}, new long[]{workSize}, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }


    public void subScalar(cl_mem a, float b, cl_mem c, int m, int n, int rows, int columns) {

        cl_kernel kernel = matrixKernels.get("subScalar");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[]{b}));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{columns}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void rsubColumnVector(cl_mem a, cl_mem b, cl_mem c, int m, int n) {

        cl_kernel kernel = matrixKernels.get("rsubColumnVector");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{m}));

        long[] local_work_size = getLocalWorkSize(m, n);
        long workSize = local_work_size[0] * local_work_size[1];

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{m * n}, new long[]{workSize}, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void rsubRowVector(cl_mem a, cl_mem b, cl_mem c, int m, int n) {

        cl_kernel kernel = matrixKernels.get("rsubRowVector");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{m}));

        long[] local_work_size = getLocalWorkSize(m, n);
        long workSize = local_work_size[0] * local_work_size[1];

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{m * n}, new long[]{workSize}, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }


    public void rsubScalar(cl_mem a, float b, cl_mem c, int m, int n, int rows, int columns) {

        cl_kernel kernel = matrixKernels.get("rsubScalar");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[]{b}));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{columns}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }


    public void div(cl_mem a, cl_mem b, cl_mem c, int m, int n, int rows, int columns) {

        cl_kernel kernel = matrixKernels.get("divMatrix");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{columns}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }


    public void divColumnVector(cl_mem a, cl_mem b, cl_mem c, int m, int n, int rows, int columns) {

        cl_kernel kernel = matrixKernels.get("divColumnVector");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{columns}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{m}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }


    public void divRowVector(cl_mem a, cl_mem b, cl_mem c, int m, int n, int rows, int columns) {

        cl_kernel kernel = matrixKernels.get("divRowVector");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{columns}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{m}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }


    public void divScalar(cl_mem a, float b, cl_mem c, int m, int n, int rows, int columns) {

        cl_kernel kernel = matrixKernels.get("divScalar");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[]{b}));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{columns}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }


    public void rdivColumnVector(cl_mem a, cl_mem b, cl_mem c, int m, int n, int rows, int columns) {

        cl_kernel kernel = matrixKernels.get("rdivColumnVector");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{columns}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{m}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }


    public void rdivRowVector(cl_mem a, cl_mem b, cl_mem c, int m, int n, int rows, int columns) {

        cl_kernel kernel = matrixKernels.get("rdivRowVector");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{columns}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{m}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }


    public void rdivScalar(cl_mem a, float b, cl_mem c, int m, int n, int rows, int columns) {

        cl_kernel kernel = matrixKernels.get("rdivScalar");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[]{b}));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(c));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{rows}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{columns}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void copy(cl_mem input, cl_mem copy, int m, int n) {

        cl_kernel kernel = setgetKernels.get("copy");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(input));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(copy));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void repmat(cl_mem source, cl_mem result, int m, int n, int outputRows, int outputColumns, int inputRows, int inputColumns, int stride) {

        cl_kernel kernel = setgetKernels.get("repmat");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(source));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(result));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, Pointer.to(new int[]{outputRows}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{outputColumns}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{inputRows}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{inputColumns}));
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_uint, Pointer.to(new int[]{stride}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void setSubMatrix(cl_mem source, cl_mem result, int m, int n, int sourceRows, int sourceColumns, int offsetRows, int offsetColumns, int resultStride) {

        cl_kernel kernel = setgetKernels.get("setSubMatrix");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(source));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(result));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, Pointer.to(new int[]{sourceRows}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{sourceColumns}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{offsetRows}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{offsetColumns}));
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_uint, Pointer.to(new int[]{resultStride}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void getSubMatrix(cl_mem source, cl_mem result, int m, int n, int resultRows, int resultColumns, int offsetRows, int offsetColumns, int sourceStride) {

        cl_kernel kernel = setgetKernels.get("getSubMatrix");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(source));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(result));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, Pointer.to(new int[]{resultRows}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{resultColumns}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{offsetRows}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{offsetColumns}));
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_uint, Pointer.to(new int[]{sourceStride}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void setZero(cl_mem values, int m, int n) {

        cl_kernel kernel = setgetKernels.get("setZero");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(values));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void setOne(cl_mem values, int m, int n, int columns, int rows) {

        cl_kernel kernel = setgetKernels.get("setOne");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(values));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_uint, Pointer.to(new int[]{columns}));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, Pointer.to(new int[]{rows}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void xorshift(cl_mem values, int m, int n) {

        cl_kernel kernel = randomKernels.get("xorshift");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(values));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});

    }

    public void uniform(cl_mem random, cl_mem values, int m, int n, int columns, int rows) {

        cl_kernel kernel = randomKernels.get("uniform");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(random));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(values));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, Pointer.to(new int[]{columns}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{rows}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }


    public void gaussian(cl_mem random, cl_mem values, float mean, float variance, int m, int n, int columns, int rows) {

        cl_kernel kernel = randomKernels.get("boxmuller");
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(random));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(values));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_float, Pointer.to(new float[]{mean}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_float, Pointer.to(new float[]{variance}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{m * n}));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, Pointer.to(new int[]{columns}));
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_uint, Pointer.to(new int[]{rows}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void applyCustomKernel(String name, cl_mem a, cl_mem result, int m, int n, int columns, int rows) {

        cl_kernel kernel = customKernels.get(name);
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(result));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, Pointer.to(new int[]{columns}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{rows}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void applyCustomKernel(String name, cl_mem a, cl_mem b, cl_mem result, int m, int n, int columns, int rows) {

        cl_kernel kernel = customKernels.get(name);
        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(result));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, Pointer.to(new int[]{columns}));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, Pointer.to(new int[]{rows}));

        long[] local_work_size = getLocalWorkSize(m, n);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void sgemm(cl_mem a, cl_mem b, cl_mem c, int m, int n, int k) {

        cl_kernel kernel = matrixKernels.get("sgemm");

        cl_event event = new cl_event();

        CL.clSetKernelArg(kernel, 0, Sizeof.cl_int, Pointer.to(new int[]{m}));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[]{n}));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{k}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(a));
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_mem, Pointer.to(b));
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_mem, Pointer.to(c));

        long[] local_work_size = getLocalWorkSize(m, n);
        long workSize = (local_work_size[0] * local_work_size[1]);

        CL.clSetKernelArg(kernel, 6, Sizeof.cl_float * workSize, null);
        CL.clSetKernelArg(kernel, 7, Sizeof.cl_float * workSize, null);

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public void transpose(cl_mem in, cl_mem out, int m, int n) {
        cl_kernel kernel = matrixKernels.get("transpose");

        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(in));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(out));
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_float * blockSizeX * blockSizeY, null);


        long[] local_work_size = {blockSizeX, blockSizeY};
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, new long[]{m, n}, local_work_size, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
    }

    public float sum(cl_mem values, int m, int n) {
        cl_kernel kernel = matrixKernels.get("sum");
        long[] local_work_size = getLocalWorkSize(m, n);
        long workSize = (local_work_size[0] * local_work_size[1]);

        cl_mem result = CL.clCreateBuffer(getContext(),
                CL.CL_MEM_READ_WRITE,
                Sizeof.cl_float * workSize, null, null);
        setZero(result, (int) local_work_size[0], (int) local_work_size[1]);

        cl_event event = new cl_event();
        CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(values));
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_float * workSize, null);
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{m * n}));
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(result));

        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{m * n}, new long[]{workSize}, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});

        float[] sums = getArray(result, (int) workSize);
        float sum = 0.0f;
        for (int i = 0; i < sums.length; i++) {
            sum += sums[i];
        }
        clReleaseMemObject(result);
        return sum;
    }


    public String loadSource(String path) {
        try {
            return new String(Files.readAllBytes(Paths.get(path)));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }


    private static String getString(cl_device_id device, int paramName) {
        // Obtain the length of the string that will be queried
        long size[] = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int) size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length - 1);
    }


    private boolean getFastestDevice(long deviceType) {

        boolean found = false;
        int numPlatformsPointer[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsPointer);
        int numPlatforms = numPlatformsPointer[0];

        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        long maxTheoreticalComputingPower = 0;
        for (cl_platform_id platform : platforms) {

            int numDevicesPointer[] = new int[1];
            try {
                clGetDeviceIDs(platform, deviceType, 0, null, numDevicesPointer);
            } catch (CLException ex) {
                numDevicesPointer[0] = 0;
            }
            int numDevices = numDevicesPointer[0];
            if (numDevices > 0) {
                cl_device_id devices[] = new cl_device_id[numDevices];
                clGetDeviceIDs(platform, deviceType, numDevices, devices, null);

                for (cl_device_id device : devices) {

                    int maxComputeUnits = getInt(device, CL_DEVICE_MAX_COMPUTE_UNITS);
                    long maxClockFrequency = getLong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY);

                    long currentComputingPower = maxComputeUnits * maxClockFrequency;
                    if (maxTheoreticalComputingPower < currentComputingPower) {
                        maxTheoreticalComputingPower = currentComputingPower;
                        found = true;
                        this.platform = platform;
                        this.device = device;
                    }
                }
            }
        }
        return found;
    }

    private long[] getLocalWorkSize(int m, int n) {
        long[] local_work_size = {blockSizeX, blockSizeY};
        if (m == 1) {
            local_work_size[0] = 1;
        }
        if (n == 1) {
            local_work_size[1] = 1;
        }
        return local_work_size;
    }

    private long[] getLongs(cl_device_id device, int paramName, int numValues) {
        long values[] = new long[numValues];
        clGetDeviceInfo(device, paramName, Sizeof.cl_long * numValues, Pointer.to(values), null);
        return values;
    }

    private long getLong(cl_device_id device, int paramName) {
        return getLongs(device, paramName, 1)[0];
    }


    private int[] getInts(cl_device_id device, int paramName, int numValues) {
        int values[] = new int[numValues];
        clGetDeviceInfo(device, paramName, Sizeof.cl_int * numValues, Pointer.to(values), null);
        return values;
    }


    private int getInt(cl_device_id device, int paramName) {
        return getInts(device, paramName, 1)[0];
    }

    private long getSize(cl_device_id device, int paramName) {
        return getSizes(device, paramName, 1)[0];
    }


    private long[] getSizes(cl_device_id device, int paramName, int numValues) {
        // The size of the returned data has to depend on
        // the size of a size_t, which is handled here
        ByteBuffer buffer = ByteBuffer.allocate(
                numValues * Sizeof.size_t).order(ByteOrder.nativeOrder());
        clGetDeviceInfo(device, paramName, Sizeof.size_t * numValues,
                Pointer.to(buffer), null);
        long values[] = new long[numValues];
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

    public cl_context getContext() {
        return context;
    }


    public int[] getDimensions(int rows, int columns) {
        int[] result = new int[2];

        result[0] = (int) Math.ceil(rows / (double) blockSizeY) * blockSizeY;
        result[1] = (int) Math.ceil(columns / (double) blockSizeX) * blockSizeX;

        if (rows == 1) result[0] = 1;
        if (columns == 1) result[1] = 1;

        return result;
    }


    public float[] getArray(cl_mem buffer, int n) {
        float[] result = new float[n];
        Pointer pointer = Pointer.to(result);
        cl_event event = new cl_event();
        CL.clEnqueueReadBuffer(commandQueue, buffer, CL.CL_TRUE, 0, n * Sizeof.cl_float, pointer, 0, null, event);
        CL.clWaitForEvents(1, new cl_event[]{event});
        return result;
    }

    public void defineFloatXKernel(String kernelName, String body) {
        String header = "__kernel void " + kernelName + "(" + "__global const float* input" + ", " + "__global float* output"
                + ", " + "const uint m, const uint n" + ")\n";
        String kernelBody = "{\n"
                + "uint id0 = get_global_id(0);\n"
                + "uint id1 = get_global_id(1);\n"
                + "if(id0 >= m || id1 >= n ) return;\n"
                + "uint id = id1 * get_global_size(0) + id0;\n"
                + "float x = input[id];\n"
                + "output[id] = " + body + ";\n}\n";
        String fullKernel = header + kernelBody;
        customKernelsSource.put(kernelName, fullKernel);
    }

    public void defineFloatXYKernel(String kernelName, String body) {
        String header = "__kernel void " + kernelName + "("
                + "__global const float* input" + ", "
                + "__global const float* input1" + ", "
                + "__global float* output" + ", "
                + "const uint m, const uint n" + ")\n";
        String kernelBody = "{\n"
                + "uint id0 = get_global_id(0);\n"
                + "uint id1 = get_global_id(1);\n"
                + "if(id0 >= m || id1 >= n ) return;\n"
                + "uint id = id1 * get_global_size(0) + id0;\n"
                + "float x = input[id];\n"
                + "float y = input1[id];\n"
                + "output[id] = " + body + ";\n}\n";
        customKernelsSource.put(kernelName, header + kernelBody);
    }

    public void compileCustomKernels() {
        // remove kernels to recompile;
        if (customProgram != null) {
            customKernels.values().forEach(CL::clReleaseKernel);
            customKernels.clear();
            CL.clReleaseProgram(customProgram);
            customProgram = null;
        }
        // get all kernels
        StringBuilder builder = new StringBuilder();
        customKernelsSource.values().forEach(builder::append);

        // create program source from all custom kernels
        String programSource = builder.toString();


        customProgram = clCreateProgramWithSource(context,
                1, new String[]{programSource}, null, null);
        clBuildProgram(customProgram, 0, null, null, null, null);

        for (String kernelName : customKernelsSource.keySet()) {
            customKernels.put(kernelName, clCreateKernel(customProgram, kernelName, null));
        }

    }
}
