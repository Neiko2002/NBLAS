package org.jcuclblas.function.functionobjects.predefined.binary;

import org.jcuclblas.function.functionobjects.generic.ABinaryInfix;
import org.jcuclblas.function.functionobjects.generic.AFunctionObject;

public class Add extends ABinaryInfix {
    public Add(AFunctionObject... functionObjects) {
        super(functionObjects);
        operation = " + ";
    }
}
