package org.math.joclblas.testRBM;

import org.math.joclblas.CLFloatMatrix;

@FunctionalInterface
public interface ICLFloatMatrixUnitFunction {
    void apply(CLFloatMatrix units);
}
