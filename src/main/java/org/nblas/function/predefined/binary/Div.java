package org.nblas.function.predefined.binary;

import org.nblas.function.generic.ABinaryInfix;
import org.nblas.function.generic.AFunctionObject;

public class Div extends ABinaryInfix {
    public Div(AFunctionObject... functionObjects) {
        super(functionObjects);
        operation = " / ";
    }
}
