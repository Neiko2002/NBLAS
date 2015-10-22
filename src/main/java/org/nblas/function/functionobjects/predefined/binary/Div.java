package org.nblas.function.functionobjects.predefined.binary;

import org.nblas.function.functionobjects.generic.ABinaryInfix;
import org.nblas.function.functionobjects.generic.AFunctionObject;

public class Div extends ABinaryInfix {
    public Div(AFunctionObject... functionObjects) {
        super(functionObjects);
        operation = " / ";
    }
}
