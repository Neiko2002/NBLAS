package org.nblas.cl;

import org.jocl.cl_kernel;
import org.nblas.cl.blas.CLLevel1;
import org.nblas.generic.Subprogram;
import org.nblas.impl.ContextDefault;

public class CLContext extends ContextDefault {

	protected CLLevel1 level1;
	
	protected CLContext(Precision precision, int deviceId) {
		super(precision, deviceId);
	}
	
	public CLLevel1 getLevel1() {
		return level1;
	}

	public static CLContext create(Precision precision, int deviceId) {
		deviceId = CLCore.setup(deviceId);

		// setup other elements
		CLContext context = new CLContext(precision, deviceId);
		
		CLCore CORE = CLCore.getCore(context.getDeviceId());
		CLFloatFunctionBuilder builder = new CLFloatFunctionBuilder(context);
		
        // lade alle Predefined Kernels
        for (Subprogram<cl_kernel> subprogram : CLPredefined.getAllSubPrograms())
        	CORE.loadFromGeneratedSubprogram(subprogram);
	
        context.level1 = new CLLevel1(context, builder);
		
		CORE.compileMatrixFunctions();
		
		return context;
	}

	@Override
	public Backend getBackend() {
		return Backend.OpenCL;
	}
}
