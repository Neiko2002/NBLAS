package org.nblas.cuda;

import org.nblas.cuda.blas.CudaLevel1;
import org.nblas.generic.Subprogram;
import org.nblas.impl.ContextDefault;

import jcuda.driver.CUfunction;

public class CudaContext extends ContextDefault {

	protected CudaLevel1 level1;
	
	protected CudaContext(Precision precision, int deviceId) {
		super(precision, deviceId);
	}
	
	public CudaLevel1 getLevel1() {
		return level1;
	}

	public static CudaContext create(Precision precision, int deviceId) {
		deviceId = CudaCore.setup(deviceId);

		//  setup other elements
		CudaContext context = new CudaContext(precision, deviceId);
		
		CudaCore CORE = CudaCore.getCore(context.getDeviceId());
		CudaFloatFunctionBuilder builder = new CudaFloatFunctionBuilder(context);
  
        for (Subprogram<CUfunction> subprogram : CudaPredefined.kernels.values()) {
            CORE.loadFromGeneratedSubprogram(subprogram);
        }
        
        context.level1 = new CudaLevel1(context, builder);
		
		return context;
	}
	

	@Override
	public Backend getBackend() {
		return Backend.CUDA;
	}
}
