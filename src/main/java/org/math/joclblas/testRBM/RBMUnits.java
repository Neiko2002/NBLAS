package org.math.joclblas.testRBM;

/**
 * Created by Moritz on 1/28/2015.
 */
public enum RBMUnits {
    REAL(0), BINARY(1), RECTIFIED(2);

    private final int value;

    private RBMUnits(int i) {
        value = i;
    }

    public int getValue() {
        return value;
    }
}
