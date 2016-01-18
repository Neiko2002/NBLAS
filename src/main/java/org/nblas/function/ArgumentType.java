package org.nblas.function;

/**
 * Created by Moritz on 4/19/2015.
 */
public enum ArgumentType {
	
    SCALAR(0, "Scalar"), ROW_VECTOR(1, "RowVec"), COLUMN_VECTOR(2, "ColVec"), MATRIX(3, "Mat");
	
    private final int value;
    private final String shortName;

    private ArgumentType(int i, String shortName) {
        this.value = i;
        this.shortName = shortName;
    }

    public int getValue() {
        return value;
    }
    
    public String getShortName() {
		return shortName;
	}
}
