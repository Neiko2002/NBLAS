package org.jcuclblas.function.functionobjects.predefined.binary;

import org.jcuclblas.function.functionobjects.generic.ABinaryInfix;
import org.jcuclblas.function.functionobjects.generic.AFunctionObject;

public class Sub extends ABinaryInfix {
    public Sub(AFunctionObject... functionObjects) {
        super(functionObjects);
        operation = " - ";
    }
}
