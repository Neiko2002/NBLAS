package org.nblas.function;

/**
 * Created by Moritz on 4/19/2015.
 */
public enum ArgumentType {
    SCALAR(0), ROW_VECTOR(1), COLUMN_VECTOR(2), MATRIIX(3);
    private final int value;

    private ArgumentType(int i) {
        value = i;
    }

    public int getValue() {
        return value;
    }
}
