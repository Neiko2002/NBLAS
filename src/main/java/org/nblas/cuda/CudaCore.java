package org.nblas.cuda;



import static jcuda.jcublas.JCublas2.cublasSetVector;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.jcurand.JCurand.curandCreateGenerator;
import static jcuda.jcurand.JCurand.curandGenerateNormal;
import static jcuda.jcurand.JCurand.curandGenerateUniform;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;

import org.nblas.generic.Subprogram;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.jcurand.curandRngType;
import jcuda.jcusolver.JCusolver;
import jcuda.runtime.JCuda;


/**
 * Tuning
 * http://docs.nvidia.com/cuda/maxwell-tuning-guide/
 * 
 * https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDA__TYPES_gd39dec7b9a5c64b8f96d0d09e249ce5d.html#gd39dec7b9a5c64b8f96d0d09e249ce5d
 * https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDA__DEVICE_g52b5ce05cb8c5fb6831b2c0ff2887c74.html#g52b5ce05cb8c5fb6831b2c0ff2887c74
 * 
 * @author Nico
 *
 */
class CudaCore {
    private static CudaCore core = new CudaCore();

    private CudaDevice device;

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
        
    	device = setupDevice();
   
        threadCount = device.getMaxThreadPerBlock();
        int logTHREAD_COUNT = (int) Math.round(Math.log(threadCount) / Math.log(2));
        int logTHREAD_COUNTX = logTHREAD_COUNT / 2;
        int logTHREAD_COUNTY = (logTHREAD_COUNT % 2 == 0) ? logTHREAD_COUNTX : logTHREAD_COUNTX + 1;
        threadCount_2DX = (int) Math.pow(2.0, logTHREAD_COUNTX);
        threadCount_2DY = (int) Math.pow(2.0, logTHREAD_COUNTY);     
        
        // TODO: geht auch und ist x mal schneller (z.b. sub())
//        threadCount_2DX = threadCount;   
//        threadCount_2DY = threadCount;   
        
     
        // curand initialization
        generator = new curandGenerator();
        curandCreateGenerator(generator, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);

        // cublas2 initialization
        cublasHandle = new cublasHandle();
        JCublas2.cublasCreate(cublasHandle);

        // JCusolver initialization
        JCusolver.initialize();

        // load and compile some predefined cuda functions
        functions = new HashMap<>();
        loadFromGeneratedFunction(CudaPredefined.kernels.get("copy1D"));
        loadFromGeneratedFunction(CudaPredefined.kernels.get("transpose"));
        loadFromGeneratedFunction(CudaPredefined.kernels.get("getsub"));
        loadFromGeneratedFunction(CudaPredefined.kernels.get("setsub"));
    }
    
    private CudaDevice setupDevice() {
    	
        JCudaDriver.cuInit(0);

        JCuda.setExceptionsEnabled(true);
        JCublas2.setExceptionsEnabled(true);
        JCurand.setExceptionsEnabled(true);
        JCusolver.setExceptionsEnabled(true);

        CudaDevice device = CudaDevice.getDevices()[0];
        device.use();
        System.out.println("Use CUDA device "+device.toString());

        return device;
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

        JCudaDriver.cuLaunchKernel(functions.get(functionName),
                blocksX, blocksY, 1,
                threadCount_2DX, threadCount_2DY, 1,
                0,
                null,
                kernelParameters, null);
        JCudaDriver.cuCtxSynchronize();
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
        JCudaDriver.cuLaunchKernel(f,
                blocks, 1, 1,
                threadCount, 1, 1,
                threadCount * Sizeof.FLOAT,
                null,
                kernelParameters, null);
        JCudaDriver.cuCtxSynchronize();
        while (blocks > 1) {
            int b = blocks;
            blocks = (int) Math.ceil(blocks / (double) threadCount);
            kernelParameters = Pointer.to(new Pointer[]{
                    Pointer.to(temp),
                    Pointer.to(temp),
                    Pointer.to(new int[]{b}),
                    Pointer.to(new float[]{initValue})

            });
            JCudaDriver.cuLaunchKernel(f,
                    blocks, 1, 1,
                    threadCount, 1, 1,
                    threadCount * Sizeof.FLOAT,
                    null,
                    kernelParameters, null);
            JCudaDriver.cuCtxSynchronize();
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
        JCudaDriver.cuLaunchKernel(f,
                blocksX, blocksY, 1,
                threadCount_2DX, threadCount_2DY, 1,
                threadCount * Sizeof.FLOAT,
                null,
                kernelParameters, null);
        JCudaDriver.cuCtxSynchronize();

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

        JCudaDriver.cuLaunchKernel(functions.get("getsub"),
                blocksX, blocksY, 1,
                threadCount_2DX, threadCount_2DY, 1,
                0,
                null,
                kernelParameters, null);
        JCudaDriver.cuCtxSynchronize();
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

        JCudaDriver.cuLaunchKernel(functions.get("setsub"),
                blocksX, blocksY, 1,
                threadCount_2DX, threadCount_2DY, 1,
                0,
                null,
                kernelParameters, null);
        JCudaDriver.cuCtxSynchronize();
    }


    public void mmul(Pointer a, int aRows, int aColumnsbRows,
                     Pointer b, int bColumns,
                     Pointer c) {
        JCublas2.cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                aRows, bColumns, aColumnsbRows,
                Pointer.to(new float[]{1.0f}), a, aRows,
                b, aColumnsbRows,
                Pointer.to(new float[]{0.0f}), c, aRows);
        JCudaDriver.cuCtxSynchronize();
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
        JCudaDriver.cuCtxSynchronize();
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
        JCudaDriver.cuCtxSynchronize();
        cudaDeviceSynchronize();
    }

    private void copy1d(Pointer data, Pointer copy, int n) {
        int blocks = (int) Math.ceil(n / (double) threadCount);
        CUfunction f = functions.get("copy1D");

        Pointer kernelParameters = Pointer.to(Pointer.to(data), Pointer.to(copy), Pointer.to(new int[]{n}));
        JCudaDriver.cuLaunchKernel(f,
                blocks, 1, 1,
                threadCount, 1, 1,
                0,
                null,
                kernelParameters, null);
        JCudaDriver.cuCtxSynchronize();
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
        JCudaDriver.cuLaunchKernel(f,
                blocksX, blocksY, 1,
                threadCount_2DX, threadCount_2DY, 1,
                sharedSize * Sizeof.FLOAT,
                null,
                kernelParameters, null);
        JCudaDriver.cuCtxSynchronize();

    }

    public void randn(Pointer a, int n) {
        curandGenerateNormal(generator, a, n, 0.0f, 1.0f);
        JCudaDriver.cuCtxSynchronize();
    }

    public void rand(Pointer a, int n) {
        curandGenerateUniform(generator, a, n);
        JCudaDriver.cuCtxSynchronize();
    }


    public void loadFromGeneratedFunction(Subprogram<CUfunction> subprogram) {
        try {
        	String name = subprogram.getProgramName();
        	String sourceCode = subprogram.getSourceCode();
        	
            Path tempDir = Paths.get(System.getProperty("java.io.tmpdir")).resolve("nblas");
            Path ptxFile = tempDir.resolve(name + ".ptx");
            Path cuFile = tempDir.resolve(name + ".cu");
            boolean store = (Files.exists(cuFile) == false);	// muss cu gespeichert werden
            boolean compile = (Files.exists(ptxFile) == false);	// muss ptx compiliert werden
            
        	try {
        		
	            // existiert das Fileverzeichnis
	            if(Files.exists(tempDir) == false)
	            	Files.createDirectories(tempDir);
	
	            // gibt es die cu Datei schon und ist der Inhalt identisch zum aktuellen subprogram
	            if(Files.exists(cuFile)) {
	            	String oldSourceCode = new String(Files.readAllBytes(cuFile), Charset.defaultCharset());
	            	if(oldSourceCode.equalsIgnoreCase(sourceCode) == false) {
	            		Files.delete(cuFile);
	            		store = true;
	            	}
	            }
	            
	            // falls nicht schreibe eine neue Datei
	            if(store) {
	             	Files.write(cuFile, subprogram.getSourceCode().getBytes());
	            	compile = true;
	            }
            		
				// compile die cuda Datei zu einer ptx Datei
				if(compile)
					compilePtxFile(cuFile, ptxFile);

				// lade die ptx Datei
				loadModule(name, ptxFile.toAbsolutePath().toString());
				
			} catch (IOException e) {
				e.printStackTrace();
			}

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private void loadModule(String name, String ptxFileName) {
    	    	
        CUmodule module = new CUmodule();
        JCudaDriver.cuModuleLoad(module, ptxFileName);

        CUfunction function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, name);

        functions.put(name, function);
    }

    /**
     * TODO verwende ProcessBuild Runtime.getRuntime().exec()
     * 
     * http://stackoverflow.com/questions/7696230/nvidia-nvcc-and-cuda-cubin-vs-ptx
     * cubin(native code) Files sind architecture-specific 
     * ptx(intermediate format) Files sind forward-compatible
     * 
     * https://github.com/JuhyunKim-Corelab/cudnn-test/blob/master/nvcc-help.txt
     * nvcc -m64 -code="sm_35" -arch="compute_35" -cubin fargppgidfsargpqgidf.cu -o fargppgidfsargpqgidf.cubin
     *  
     * @param cuFile
     * @return
     * @throws IOException
     */
    private void compilePtxFile(Path cuFile, Path ptxFile) throws IOException {

        String cuFileName = cuFile.toAbsolutePath().toString();
        String ptxFileName = ptxFile.toAbsolutePath().toString();
        
    	if(Files.exists(ptxFile))  {
			Files.delete(ptxFile);
    	}
    	
    	if(Files.exists(cuFile) == false) {
    		throw new IOException("Input file not found: " + cuFileName);
    	}

        String outputFormat = "-ptx"; // ptx oder cubin
        String modelString = "-m"+System.getProperty("sun.arch.data.model"); // 32bit oder 64bit (TODO: performance gewinn bei 32bit?)
        String command = "nvcc " + modelString + " " + outputFormat + " " + cuFile.toString() + " -o " + ptxFileName;

        System.out.println("Executing\n" + command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage = new String(toByteArray(process.getErrorStream()));
        String outputMessage = new String(toByteArray(process.getInputStream()));

        int exitValue = 0;
        try {
            exitValue = process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0) {
            System.out.println("nvcc process exitValue " + exitValue);
            System.out.println("errorMessage:\n" + errorMessage);
            System.out.println("outputMessage:\n" + outputMessage);
            throw new IOException("Could not create .ptx file: " + errorMessage);
        }

        System.out.println("Finished creating PTX file");
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
