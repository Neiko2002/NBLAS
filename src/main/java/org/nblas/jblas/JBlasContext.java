package org.nblas.jblas;

import org.nblas.impl.ContextDefault;

public class JBlasContext extends ContextDefault {

	protected JBlasContext(Precision precision, int deviceId) {
		super(precision, deviceId);
	}
	
	public static JBlasContext create(Precision precision, int deviceId) {
		return new JBlasContext(precision, deviceId);
	}

	@Override
	public Backend getBackend() {
		return Backend.JBLAS;
	}
}
