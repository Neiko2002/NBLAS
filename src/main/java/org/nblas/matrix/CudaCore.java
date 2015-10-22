package org.nblas.matrix;


import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.jcurand.curandRngType;
import jcuda.jcusolver.JCusolver;
import jcuda.runtime.JCuda;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.jcublas.JCublas2.cublasSetVector;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.jcurand.JCurand.curandCreateGenerator;
import static jcuda.jcurand.JCurand.curandGenerateNormal;
import static jcuda.jcurand.JCurand.curandGenerateUniform;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;

class CudaCore {
    private static CudaCore core = new CudaCore();

    private CUdevice device;
    private CUcontext context;

    private curandGenerator generator;
    private cublasHandle cublasHandle;

    private int threadCount;
    private int threadCount_2DX;
    private int threadCount_2DY;

    private HashMap<String, CUfunction> functions;

    public static CudaCore getCore() {
        return core;
    }

    private CudaCore() {
        functions = new HashMap<>();
        cuInit(0);

        JCuda.setExceptionsEnabled(true);
        JCublas2.setExceptionsEnabled(true);
        JCurand.setExceptionsEnabled(true);
        JCusolver.setExceptionsEnabled(true);

        context = new CUcontext();
        device = new CUdevice();
        int[] version = new int[1];
        JCuda.cudaRuntimeGetVersion(version);
        System.out.println("CUDA Runtime Version: " + version[0]);
        int[] value = new int[1];
        cuDeviceGetAttribute(value, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
        threadCount = value[0];
        int logTHREAD_COUNT = (int) Math.round(Math.log(threadCount) / Math.log(2));
        int logTHREAD_COUNTX = logTHREAD_COUNT / 2;
        int logTHREAD_COUNTY = (logTHREAD_COUNT % 2 == 0) ? logTHREAD_COUNTX : logTHREAD_COUNTX + 1;
        threadCount_2DX = (int) Math.pow(2.0, logTHREAD_COUNTX);
        threadCount_2DY = (int) Math.pow(2.0, logTHREAD_COUNTY);

        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        // curand initialization
        generator = new curandGenerator();
        curandCreateGenerator(generator, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);


        // cublas2 initialization
        cublasHandle = new cublasHandle();
        JCublas2.cublasCreate(cublasHandle);

        JCusolver.initialize();

        loadFromGeneratedFunction("copy1D", CudaPredefined.kernels.get("copy1D"));
        loadFromGeneratedFunction("transpose", CudaPredefined.kernels.get("transpose"));
        loadFunction("getsub", "cuda/getsub.cu");
        loadFunction("setsub", "cuda/setsub.cu");

        for (String name : CudaPredefined.kernels.keySet()) {
            loadFromGeneratedFunction(name, CudaPredefined.kernels.get(name));
        }
    }

    public void execute(String functionName, int rows, int columns, Pointer result, Pointer... args) {
        int blocksY = (int) Math.ceil(columns / (double) threadCount_2DX);
        int blocksX = (int) Math.ceil(rows / (double) threadCount_2DY);
        ArrayList<Pointer> pointers = new ArrayList<>();
        pointers.add(Pointer.to(result));
        pointers.add(Pointer.to(new int[]{columns}));
        pointers.add(Pointer.to(new int[]{rows}));
        for (int i = 0; i < args.length; i++) {
            pointers.add(Pointer.to(args[i]));
        }
        Pointer kernelParameters = Pointer.to(pointers.toArray(new Pointer[0]));

        cuLaunchKernel(functions.get(functionName),
                blocksX, blocksY, 1,
                threadCount_2DX, threadCount_2DY, 1,
                0,
                null,
                kernelParameters, null);
        cuCtxSynchronize();
    }

    public float reduce(String reductionName, Pointer data, int n, float initValue) {
        CUfunction f = functions.get(reductionName);
        int blocks = (int) Math.ceil(n / (double) threadCount);
        Pointer temp = malloc(blocks);
        Pointer kernelParameters = Pointer.to(new Pointer[]{
                Pointer.to(data),
                Pointer.to(temp),
                Pointer.to(new int[]{n}),
                Pointer.to(new float[]{initValue})

        });
        cuLaunchKernel(f,
                blocks, 1, 1,
                threadCount, 1, 1,
                threadCount * Sizeof.FLOAT,
                null,
                kernelParameters, null);
        cuCtxSynchronize();
        while (blocks > 1) {
            int b = blocks;
            blocks = (int) Math.ceil(blocks / (double) threadCount);
            kernelParameters = Pointer.to(new Pointer[]{
                    Pointer.to(temp),
                    Pointer.to(temp),
                    Pointer.to(new int[]{b}),
                    Pointer.to(new float[]{initValue})

            });
            cuLaunchKernel(f,
                    blocks, 1, 1,
                    threadCount, 1, 1,
                    threadCount * Sizeof.FLOAT,
                    null,
                    kernelParameters, null);
            cuCtxSynchronize();
        }
        float[] result = new float[1];
        getData(temp, result);
        return result[0];
    }

    public void reduceRows(String functionName, Pointer data, Pointer result, int rows, int columns, float initValue) {

        int blocksX = (int) Math.ceil(rows / (double) threadCount_2DX);
        int blocksY = (int) Math.ceil(columns / (double) threadCount_2DY);
        Pointer temp = malloc(rows * blocksY);
        CUfunction f = functions.get(functionName);
        reduceCall(f, data, temp, rows, columns, blocksX, blocksY, initValue);
        while (blocksY > 1) {
            int c = blocksY;
            blocksY = (int) Math.ceil(blocksY / (double) threadCount_2DY);
            reduceCall(f, temp, temp, rows, c, blocksX, blocksY, initValue);
        }
        copy1d(temp, result, rows);
        free(temp);
    }

    public void reduceColumns(String functionName, Pointer data, Pointer result, int rows, int columns, float initValue) {

        int blocksX = (int) Math.ceil(rows / (double) threadCount_2DX);
        int blocksY = (int) Math.ceil(columns / (double) threadCount_2DY);
        Pointer temp = malloc(columns * blocksX);
        CUfunction f = functions.get(functionName);
        reduceCall(f, data, temp, rows, columns, blocksX, blocksY, initValue);
        while (blocksX > 1) {
            int r = blocksX;
            blocksX = (int) Math.ceil(blocksX / (double) threadCount_2DY);
            reduceCall(f, temp, temp, r, columns, blocksX, blocksY, initValue);
        }
        copy1d(temp, result, columns);
        free(temp);
    }

    private void reduceCall(CUfunction f, Pointer data, Pointer result, int rows, int columns, int blocksX, int blocksY, float initValue) {

        Pointer kernelParameters = Pointer.to(new Pointer[]{
                Pointer.to(data),
                Pointer.to(result),
                Pointer.to(new int[]{rows}),
                Pointer.to(new int[]{columns}),
                Pointer.to(new float[]{initValue})

        });
        cuLaunchKernel(f,
                blocksX, blocksY, 1,
                threadCount_2DX, threadCount_2DY, 1,
                threadCount * Sizeof.FLOAT,
                null,
                kernelParameters, null);
        cuCtxSynchronize();

    }

    public void getSubMatrix(Pointer data, Pointer result, int resultRows, int resultColumns, int dataRows, int offsetRow, int offsetColumn) {
        int blocksY = (int) Math.ceil(resultColumns / (double) threadCount_2DX);
        int blocksX = (int) Math.ceil(resultRows / (double) threadCount_2DY);

        Pointer kernelParameters = Pointer.to(new Pointer[]{
                Pointer.to(data),
                Pointer.to(result),
                Pointer.to(new int[]{resultRows}),
                Pointer.to(new int[]{resultColumns}),
                Pointer.to(new int[]{dataRows}),
                Pointer.to(new int[]{offsetRow}),
                Pointer.to(new int[]{offsetColumn})

        });

        cuLaunchKernel(functions.get("getsub"),
                blocksX, blocksY, 1,
                threadCount_2DX, threadCount_2DY, 1,
                0,
                null,
                kernelParameters, null);
        cuCtxSynchronize();
    }

    public void setSubMatrix(Pointer result, Pointer data, int rows, int columns, int dataRows, int offsetRow, int offsetColumn) {
        int blocksY = (int) Math.ceil(columns / (double) threadCount_2DX);
        int blocksX = (int) Math.ceil(rows / (double) threadCount_2DY);

        Pointer kernelParameters = Pointer.to(new Pointer[]{
                Pointer.to(data),
                Pointer.to(result),
                Pointer.to(new int[]{rows}),
                Pointer.to(new int[]{columns}),
                Pointer.to(new int[]{dataRows}),
                Pointer.to(new int[]{offsetRow}),
                Pointer.to(new int[]{offsetColumn})

        });

        cuLaunchKernel(functions.get("setsub"),
                blocksX, blocksY, 1,
                threadCount_2DX, threadCount_2DY, 1,
                0,
                null,
                kernelParameters, null);
        cuCtxSynchronize();
    }


    public void mmul(Pointer a, int aRows, int aColumnsbRows,
                     Pointer b, int bColumns,
                     Pointer c) {
        JCublas2.cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                aRows, bColumns, aColumnsbRows,
                Pointer.to(new float[]{1.0f}), a, aRows,
                b, aColumnsbRows,
                Pointer.to(new float[]{0.0f}), c, aRows);
        cuCtxSynchronize();
        cudaDeviceSynchronize();
    }

    public void mmulTransposeA(Pointer a, int aRowsbRows, int aColumns,
                               Pointer b, int bColumns,
                               Pointer c) {
        JCublas2.cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                aColumns, bColumns, aRowsbRows,
                Pointer.to(new float[]{1.0f}), a, aRowsbRows,
                b, aRowsbRows,
                Pointer.to(new float[]{0.0f}), c, aColumns);
        cuCtxSynchronize();
        cudaDeviceSynchronize();
    }

    public void mmulTransposeB(Pointer a, int aRows, int aColumnsbColumns,
                               Pointer b, int bRows,
                               Pointer c) {
        JCublas2.cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                aRows, bRows, aColumnsbColumns,
                Pointer.to(new float[]{1.0f}), a, aRows,
                b, bRows,
                Pointer.to(new float[]{0.0f}), c, aRows);
        cuCtxSynchronize();
        cudaDeviceSynchronize();
    }

    private void copy1d(Pointer data, Pointer copy, int n) {
        int blocks = (int) Math.ceil(n / (double) threadCount);
        CUfunction f = functions.get("copy1D");

        Pointer kernelParameters = Pointer.to(Pointer.to(data), Pointer.to(copy), Pointer.to(new int[]{n}));
        cuLaunchKernel(f,
                blocks, 1, 1,
                threadCount, 1, 1,
                0,
                null,
                kernelParameters, null);
        cuCtxSynchronize();
    }

    public void transpose(Pointer data, Pointer result, int rows, int columns) {
        int blocksX = (int) Math.ceil(rows / (double) threadCount_2DX);
        int blocksY = (int) Math.ceil(columns / (double) threadCount_2DY);
        int sharedSize = (threadCount_2DX + 1) * threadCount_2DY;
        Pointer kernelParameters = Pointer.to(new Pointer[]{
                Pointer.to(data),
                Pointer.to(result),
                Pointer.to(new int[]{rows}),
                Pointer.to(new int[]{columns})});

        CUfunction f = functions.get("transpose");
        cuLaunchKernel(f,
                blocksX, blocksY, 1,
                threadCount_2DX, threadCount_2DY, 1,
                sharedSize * Sizeof.FLOAT,
                null,
                kernelParameters, null);
        cuCtxSynchronize();

    }

    public void randn(Pointer a, int n) {
        curandGenerateNormal(generator, a, n, 0.0f, 1.0f);
        cuCtxSynchronize();
    }

    public void rand(Pointer a, int n) {
        curandGenerateUniform(generator, a, n);
        cuCtxSynchronize();
    }


    public void loadFromGeneratedFunction(String functionName, String function) {
        String path = "c:\\temp\\" + functionName + ".cu";

        try (PrintWriter out = new PrintWriter(path)) {
            out.write(function);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        loadFunction(functionName, path);
    }

    public void loadFunction(String name, String cuFilePath) {

        try {
            String ptxFileName = compilePtxFile(cuFilePath);

            CUmodule module = new CUmodule();
            cuModuleLoad(module, ptxFileName);

            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, module, name);

            functions.put(name, function);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String compilePtxFile(String cuFileName) throws IOException {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1) {
            endIndex = cuFileName.length() - 1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex + 1) + "ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists()) {
            return ptxFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists()) {
            throw new IOException("Input file not found: " + cuFileName);
        }
        String modelString = "-m" + System.getProperty("sun.arch.data.model");
        String command =
                "nvcc " + modelString + " -ptx " +
                        cuFile.getPath() + " -o " + ptxFileName;

        System.out.println("Executing\n" + command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
                new String(toByteArray(process.getErrorStream()));
        String outputMessage =
                new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try {
            exitValue = process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(
                    "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0) {
            System.out.println("nvcc process exitValue " + exitValue);
            System.out.println("errorMessage:\n" + errorMessage);
            System.out.println("outputMessage:\n" + outputMessage);
            throw new IOException(
                    "Could not create .ptx file: " + errorMessage);
        }

        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }

    private byte[] toByteArray(InputStream inputStream)
            throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true) {
            int read = inputStream.read(buffer);
            if (read == -1) {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

    public Pointer malloc(float[] values) {
        Pointer pointer = new Pointer();
        cudaMalloc(pointer, values.length * Sizeof.FLOAT);
        cublasSetVector(values.length, Sizeof.FLOAT, Pointer.to(values), 1, pointer, 1);
        return pointer;
    }

    public Pointer malloc(int length) {
        Pointer pointer = new Pointer();
        cudaMalloc(pointer, length * Sizeof.FLOAT);
        return pointer;
    }

    public void free(Pointer pointer) {
        cudaFree(pointer);
    }

    public void setData(Pointer pointer, float[] values) {
        cublasSetVector(values.length, Sizeof.FLOAT, Pointer.to(values), 1, pointer, 1);
    }

    public void getData(Pointer pointer, float[] values) {
        JCublas2.cublasGetVector(values.length, Sizeof.FLOAT, pointer, 1, Pointer.to(values), 1);
    }


}
