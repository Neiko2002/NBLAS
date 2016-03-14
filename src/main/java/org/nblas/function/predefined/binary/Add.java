package org.nblas.function.predefined.binary;

import org.nblas.function.generic.ABinaryInfix;
import org.nblas.function.generic.AFunctionObject;

public class Add extends ABinaryInfix {
    public Add(AFunctionObject... functionObjects) {
        super(functionObjects);
        operation = " + ";
    }
}
