package org.nblas.cuda;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.jcurand.JCurand.curandCreateGenerator;
import static jcuda.jcurand.JCurand.curandGenerateNormal;
import static jcuda.jcurand.JCurand.curandGenerateUniform;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaFree;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

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
 * Tuning http://docs.nvidia.com/cuda/maxwell-tuning-guide/
 * 
 * TODO Matrix funktionen wie getSubMatrix gehören eine CudaMatrix Klasse und
 * mmulTransposeA in die CudaFloatMatrix klasse
 * 
 * https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/
 * group__CUDA__TYPES_gd39dec7b9a5c64b8f96d0d09e249ce5d.html#
 * gd39dec7b9a5c64b8f96d0d09e249ce5d
 * https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/
 * group__CUDA__DEVICE_g52b5ce05cb8c5fb6831b2c0ff2887c74.html#
 * g52b5ce05cb8c5fb6831b2c0ff2887c74
 * 
 * @author Nico
 *
 */
class CudaCore {
	
	private static final Map<Integer, CudaCore> activeCores = new HashMap<>();

	/**
	 * Setup the CLCore for a specific device. Device id -1 means the fastest
	 * avaiable device.
	 * 
	 * @param deviceId
	 * @return
	 */
	protected static int setup(int deviceId) {

		JCudaDriver.cuInit(0);

		JCuda.setExceptionsEnabled(true);
		JCublas2.setExceptionsEnabled(true);
		JCurand.setExceptionsEnabled(true);
		JCusolver.setExceptionsEnabled(true);
		
		// get a list of all Cuda devices
		final CudaDevice[] devices = CudaDevice.getDevices();

		// find the fastest device
		if (deviceId == -1 && devices.length > 0) {
			deviceId = IntStream.range(0, devices.length).boxed().max((idx1, idx2) -> {
				return Integer.compare(devices[idx1].getTheoreticalSpeed(), devices[idx2].getTheoreticalSpeed());
			}).get();
		}

		
		// already setup?
		CudaCore core = activeCores.get(deviceId);
		if (core == null) {

			// get device for the new core
			CudaDevice device = devices[deviceId];
			if (device == null) {
				System.out.println("Could not find device with id " + deviceId);
				return -1;
			}

			// create new core
			core = new CudaCore(device);
			System.out.println("Use CUDA device: \n" + device.toString());
			activeCores.put(deviceId, core);
		}

		return deviceId;
	}

	/**
	 * OpenCL Core for a specific device. Creates a new core if there is no core
	 * for the device, otherwise returns the existing one.
	 * 
	 * the device
	 * 
	 * @param index
	 * @return
	 */
	protected static CudaCore getCore(int deviceId) {
		return activeCores.get(deviceId);
	}

	private curandGenerator generator;
	private cublasHandle cublasHandle;

	private int threadCount;
	private int threadCount_2DX;
	private int threadCount_2DY;

	private HashMap<String, Subprogram<CUfunction>> functions;

	private CudaCore(CudaDevice device) {
		device.use();
		
		threadCount = device.getMaxThreadPerBlock();
		int logTHREAD_COUNT = (int) Math.round(Math.log(threadCount) / Math.log(2));
		int logTHREAD_COUNTX = logTHREAD_COUNT / 2;
		int logTHREAD_COUNTY = (logTHREAD_COUNT % 2 == 0) ? logTHREAD_COUNTX : logTHREAD_COUNTX + 1;
		threadCount_2DX = (int) Math.pow(2.0, logTHREAD_COUNTX);
		threadCount_2DY = (int) Math.pow(2.0, logTHREAD_COUNTY);

		// TODO: geht auch und ist x mal schneller (z.b. sub())
		// threadCount_2DX = threadCount;
		// threadCount_2DY = threadCount;

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
	}


	protected void execute(Subprogram<CUfunction> subprogram, int rows, int columns, Pointer result, Pointer... args) {
		int blocksY = (int) Math.ceil(columns / (double) threadCount_2DX);
		int blocksX = (int) Math.ceil(rows / (double) threadCount_2DY);

		// Parameter die an Cuda übergeben werden
		ArrayList<Pointer> pointers = new ArrayList<>();
		pointers.add(Pointer.to(result));
		pointers.add(Pointer.to(new int[] { columns }));
		pointers.add(Pointer.to(new int[] { rows }));
		for (Pointer arg : args)
			pointers.add(Pointer.to(arg));
		Pointer kernelParameters = Pointer.to(pointers.toArray(new Pointer[0]));

		// führe den Cuda Kernel aus
		JCudaDriver.cuLaunchKernel(subprogram.getKernel(), blocksX, blocksY, 1, threadCount_2DX, threadCount_2DY, 1, 0,
				null, kernelParameters, null);
		JCudaDriver.cuCtxSynchronize();
	}

	protected void loadFromGeneratedSubprogram(Subprogram<CUfunction> subprogram) {
		try {

			String name = subprogram.getProgramName();
			String sourceCode = subprogram.getSourceCode();

			Path tempDir = Paths.get(System.getProperty("java.io.tmpdir")).resolve("nblas");
			Path ptxFile = tempDir.resolve(name + ".ptx");
			Path cuFile = tempDir.resolve(name + ".cu");
			boolean store = (Files.exists(cuFile) == false); // muss cu
																// gespeichert
																// werden
			boolean compile = (Files.exists(ptxFile) == false); // muss ptx
																// compiliert
																// werden

			// existiert das Fileverzeichnis
			if (Files.exists(tempDir) == false)
				Files.createDirectories(tempDir);

			// gibt es die cu Datei schon und ist der Inhalt identisch zum
			// aktuellen subprogram
			if (Files.exists(cuFile)) {
				String oldSourceCode = new String(Files.readAllBytes(cuFile), Charset.defaultCharset());
				if (oldSourceCode.equalsIgnoreCase(sourceCode) == false) {
					Files.delete(cuFile);
					store = true;
				}
			}

			// falls nicht schreibe eine neue Datei
			if (store) {
				Files.write(cuFile, subprogram.getSourceCode().getBytes());
				compile = true;
			}

			// compile die cuda Datei zu einer ptx Datei
			if (compile)
				compilePtxFile(cuFile, ptxFile);

			// lade die ptx Datei
			loadModule(subprogram, ptxFile.toAbsolutePath().toString());

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	private void loadModule(Subprogram<CUfunction> subprogram, String ptxFileName) throws Exception {

		String name = subprogram.getProgramName();
		if (functions.get(name) != null)
			throw new Exception("CUDA function " + name + " is already loaded.");

		CUmodule module = new CUmodule();
		JCudaDriver.cuModuleLoad(module, ptxFileName);

		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module, name);
		subprogram.setKernel(function);

		functions.put(name, subprogram);
	}

	/**
	 * TODO verwende ProcessBuild Runtime.getRuntime().exec()
	 * 
	 * http://stackoverflow.com/questions/7696230/nvidia-nvcc-and-cuda-cubin-vs-
	 * ptx cubin(native code) Files sind architecture-specific ptx(intermediate
	 * format) Files sind forward-compatible
	 * 
	 * https://github.com/JuhyunKim-Corelab/cudnn-test/blob/master/nvcc-help.txt
	 * nvcc -m64 -code="sm_35" -arch="compute_35" -cubin fargppgidfsargpqgidf.cu
	 * -o fargppgidfsargpqgidf.cubin
	 * 
	 * @param cuFile
	 * @return
	 * @throws IOException
	 */
	private void compilePtxFile(Path cuFile, Path ptxFile) throws IOException {

		String cuFileName = cuFile.toAbsolutePath().toString();
		String ptxFileName = ptxFile.toAbsolutePath().toString();

		if (Files.exists(ptxFile)) {
			Files.delete(ptxFile);
		}

		if (Files.exists(cuFile) == false) {
			throw new IOException("Input file not found: " + cuFileName);
		}

		String outputFormat = "-ptx"; // ptx oder cubin
		String modelString = "-m" + System.getProperty("sun.arch.data.model"); // 32bit
																				// oder
																				// 64bit
																				// (TODO:
																				// performance
																				// gewinn
																				// bei
																				// 32bit?)
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

	private byte[] toByteArray(InputStream inputStream) throws IOException {
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

	protected void release(Pointer pointer) {
		cudaFree(pointer);
	}

	// --------------------------------------------------------------------------------------------------------
	// -------------------------------------------- manipulation
	// ----------------------------------------------
	// --------------------------------------------------------------------------------------------------------

	/**
	 * TODO copy1D vs FloatMatrix.dup()
	 * 
	 * @param data
	 * @param copy
	 * @param n
	 */
	private void copy1d(Pointer data, Pointer copy, int n) {
		int blocks = (int) Math.ceil(n / (double) threadCount);
		CUfunction f = functions.get("copy1D").getKernel();

		Pointer kernelParameters = Pointer.to(Pointer.to(data), Pointer.to(copy), Pointer.to(new int[] { n }));
		JCudaDriver.cuLaunchKernel(f, blocks, 1, 1, threadCount, 1, 1, 0, null, kernelParameters, null);
		JCudaDriver.cuCtxSynchronize();
	}

	protected void randn(Pointer a, int n) {
		curandGenerateNormal(generator, a, n, 0.0f, 1.0f);
		JCudaDriver.cuCtxSynchronize();
	}

	protected void rand(Pointer a, int n) {
		curandGenerateUniform(generator, a, n);
		JCudaDriver.cuCtxSynchronize();
	}

	// --------------------------------------------------------------------------------------------------------
	// ------------------------------------------ matrix methods
	// ----------------------------------------------
	// --------------------------------------------------------------------------------------------------------

	protected void getSubMatrix(Pointer data, Pointer result, int resultRows, int resultColumns, int dataRows,
			int offsetRow, int offsetColumn) {
		int blocksY = (int) Math.ceil(resultColumns / (double) threadCount_2DX);
		int blocksX = (int) Math.ceil(resultRows / (double) threadCount_2DY);

		Pointer kernelParameters = Pointer
				.to(new Pointer[] { Pointer.to(data), Pointer.to(result), Pointer.to(new int[] { resultRows }),
						Pointer.to(new int[] { resultColumns }), Pointer.to(new int[] { dataRows }),
						Pointer.to(new int[] { offsetRow }), Pointer.to(new int[] { offsetColumn })

		});

		JCudaDriver.cuLaunchKernel(functions.get("getsub").getKernel(), blocksX, blocksY, 1, threadCount_2DX,
				threadCount_2DY, 1, 0, null, kernelParameters, null);
		JCudaDriver.cuCtxSynchronize();

		// JCuda.cudaSetDeviceFlags(JCuda.cudaDeviceScheduleBlockingSync);
		// JCuda.cudaDeviceSynchronize();
	}

	protected void setSubMatrix(Pointer result, Pointer data, int rows, int columns, int dataRows, int offsetRow,
			int offsetColumn) {
		int blocksY = (int) Math.ceil(columns / (double) threadCount_2DX);
		int blocksX = (int) Math.ceil(rows / (double) threadCount_2DY);

		Pointer kernelParameters = Pointer.to(new Pointer[] { Pointer.to(data), Pointer.to(result),
				Pointer.to(new int[] { rows }), Pointer.to(new int[] { columns }), Pointer.to(new int[] { dataRows }),
				Pointer.to(new int[] { offsetRow }), Pointer.to(new int[] { offsetColumn })

		});

		JCudaDriver.cuLaunchKernel(functions.get("setsub").getKernel(), blocksX, blocksY, 1, threadCount_2DX,
				threadCount_2DY, 1, 0, null, kernelParameters, null);
		JCudaDriver.cuCtxSynchronize();

	}

	// --------------------------------------------------------------------------------------------------------
	// ------------------------------------ float matrix methods
	// ----------------------------------------------
	// --------------------------------------------------------------------------------------------------------

	/**
	 * z.b. Sizeof.FLOAT
	 * 
	 * @param values
	 * @param sizeof
	 * @return
	 */
	protected Pointer malloc(float[] values, int sizeof) {
		Pointer pointer = new Pointer();
		JCuda.cudaMalloc(pointer, values.length * sizeof);
		JCublas2.cublasSetVector(values.length, sizeof, Pointer.to(values), 1, pointer, 1);
		return pointer;
	}

	protected Pointer malloc(int length, int sizeof) {
		Pointer pointer = new Pointer();
		JCuda.cudaMalloc(pointer, length * sizeof);
		return pointer;
	}

	protected void setData(Pointer pointer, float[] values) {
		JCublas2.cublasSetVector(values.length, Sizeof.FLOAT, Pointer.to(values), 1, pointer, 1);
	}

	protected void getData(Pointer pointer, float[] values) {
		JCublas2.cublasGetVector(values.length, Sizeof.FLOAT, pointer, 1, Pointer.to(values), 1);
	}

	protected float reduce(String reductionName, Pointer data, int n, float initValue) {
		CUfunction f = functions.get(reductionName).getKernel();
		int blocks = (int) Math.ceil(n / (double) threadCount);
		Pointer temp = malloc(blocks, Sizeof.FLOAT);
		Pointer kernelParameters = Pointer.to(new Pointer[] { Pointer.to(data), Pointer.to(temp),
				Pointer.to(new int[] { n }), Pointer.to(new float[] { initValue })

		});
		JCudaDriver.cuLaunchKernel(f, blocks, 1, 1, threadCount, 1, 1, threadCount * Sizeof.FLOAT, null,
				kernelParameters, null);
		JCudaDriver.cuCtxSynchronize();
		while (blocks > 1) {
			int b = blocks;
			blocks = (int) Math.ceil(blocks / (double) threadCount);
			kernelParameters = Pointer.to(new Pointer[] { Pointer.to(temp), Pointer.to(temp),
					Pointer.to(new int[] { b }), Pointer.to(new float[] { initValue })

			});
			JCudaDriver.cuLaunchKernel(f, blocks, 1, 1, threadCount, 1, 1, threadCount * Sizeof.FLOAT, null,
					kernelParameters, null);
			JCudaDriver.cuCtxSynchronize();
		}
		float[] result = new float[1];
		getData(temp, result);
		return result[0];
	}

	protected void reduceRows(String functionName, Pointer data, Pointer result, int rows, int columns,
			float initValue) {

		int blocksX = (int) Math.ceil(rows / (double) threadCount_2DX);
		int blocksY = (int) Math.ceil(columns / (double) threadCount_2DY);
		Pointer temp = malloc(rows * blocksY, Sizeof.FLOAT);
		CUfunction f = functions.get(functionName).getKernel();
		reduceCall(f, data, temp, rows, columns, blocksX, blocksY, initValue);
		while (blocksY > 1) {
			int c = blocksY;
			blocksY = (int) Math.ceil(blocksY / (double) threadCount_2DY);
			reduceCall(f, temp, temp, rows, c, blocksX, blocksY, initValue);
		}
		copy1d(temp, result, rows);
		release(temp);
	}

	public void reduceColumns(String functionName, Pointer data, Pointer result, int rows, int columns,
			float initValue) {

		int blocksX = (int) Math.ceil(rows / (double) threadCount_2DX);
		int blocksY = (int) Math.ceil(columns / (double) threadCount_2DY);
		Pointer temp = malloc(columns * blocksX, Sizeof.FLOAT);
		CUfunction f = functions.get(functionName).getKernel();
		reduceCall(f, data, temp, rows, columns, blocksX, blocksY, initValue);
		while (blocksX > 1) {
			int r = blocksX;
			blocksX = (int) Math.ceil(blocksX / (double) threadCount_2DY);
			reduceCall(f, temp, temp, r, columns, blocksX, blocksY, initValue);
		}
		copy1d(temp, result, columns);
		release(temp);
	}

	private void reduceCall(CUfunction f, Pointer data, Pointer result, int rows, int columns, int blocksX, int blocksY,
			float initValue) {

		Pointer kernelParameters = Pointer.to(new Pointer[] { Pointer.to(data), Pointer.to(result),
				Pointer.to(new int[] { rows }), Pointer.to(new int[] { columns }), Pointer.to(new float[] { initValue })

		});
		JCudaDriver.cuLaunchKernel(f, blocksX, blocksY, 1, threadCount_2DX, threadCount_2DY, 1,
				threadCount * Sizeof.FLOAT, null, kernelParameters, null);
		JCudaDriver.cuCtxSynchronize();

	}

	protected void transpose(Pointer data, Pointer result, int rows, int columns) {
		int blocksX = (int) Math.ceil(rows / (double) threadCount_2DX);
		int blocksY = (int) Math.ceil(columns / (double) threadCount_2DY);
		int sharedSize = (threadCount_2DX + 1) * threadCount_2DY;
		Pointer kernelParameters = Pointer.to(new Pointer[] { Pointer.to(data), Pointer.to(result),
				Pointer.to(new int[] { rows }), Pointer.to(new int[] { columns }) });

		CUfunction f = functions.get("transpose").getKernel();
		JCudaDriver.cuLaunchKernel(f, blocksX, blocksY, 1, threadCount_2DX, threadCount_2DY, 1,
				sharedSize * Sizeof.FLOAT, null, kernelParameters, null);
		JCudaDriver.cuCtxSynchronize();
	}

	protected void sgemmNN(Pointer a, int aRows, int aColumnsbRows, Pointer b, int bColumns, Pointer c) {
		JCublas2.cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, aRows, bColumns, aColumnsbRows,
				Pointer.to(new float[] { 1.0f }), a, aRows, b, aColumnsbRows, Pointer.to(new float[] { 0.0f }), c,
				aRows);
		JCudaDriver.cuCtxSynchronize();
		cudaDeviceSynchronize();
	}

	protected void sgemmTN(Pointer a, int aRowsbRows, int aColumns, Pointer b, int bColumns, Pointer c) {
		JCublas2.cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, aColumns, bColumns, aRowsbRows,
				Pointer.to(new float[] { 1.0f }), a, aRowsbRows, b, aRowsbRows, Pointer.to(new float[] { 0.0f }), c,
				aColumns);
		JCudaDriver.cuCtxSynchronize();
		cudaDeviceSynchronize();
	}

	protected void sgemmNT(Pointer a, int aRows, int aColumnsbColumns, Pointer b, int bRows, Pointer c) {
		JCublas2.cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, aRows, bRows, aColumnsbColumns,
				Pointer.to(new float[] { 1.0f }), a, aRows, b, bRows, Pointer.to(new float[] { 0.0f }), c, aRows);
		JCudaDriver.cuCtxSynchronize();
		cudaDeviceSynchronize();
	}

}
