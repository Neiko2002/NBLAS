package org.nblas.function.functionobjects.predefined.binary;

import org.nblas.function.functionobjects.generic.ABinaryInfix;
import org.nblas.function.functionobjects.generic.AFunctionObject;

public class Add extends ABinaryInfix {
    public Add(AFunctionObject... functionObjects) {
        super(functionObjects);
        operation = " + ";
    }
}
