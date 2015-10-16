package org.jcuclblas.function.functionobjects.predefined.binary;

import org.jcuclblas.function.functionobjects.generic.ABinaryInfix;
import org.jcuclblas.function.functionobjects.generic.AFunctionObject;

public class Mul extends ABinaryInfix {
    public Mul(AFunctionObject... functionObjects) {
        super(functionObjects);
        operation = " * ";
    }
}
