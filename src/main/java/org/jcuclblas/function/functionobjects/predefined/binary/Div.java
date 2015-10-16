package org.jcuclblas.function.functionobjects.predefined.binary;

import org.jcuclblas.function.functionobjects.generic.ABinaryInfix;
import org.jcuclblas.function.functionobjects.generic.AFunctionObject;

public class Div extends ABinaryInfix {
    public Div(AFunctionObject... functionObjects) {
        super(functionObjects);
        operation = " / ";
    }
}
