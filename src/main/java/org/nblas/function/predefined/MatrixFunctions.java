package org.nblas.function.predefined;

import org.nblas.function.common.Value;
import org.nblas.function.generic.AFunctionObject;
import org.nblas.function.predefined.binary.Add;
import org.nblas.function.predefined.unary.Exp;
import org.nblas.function.predefined.unary.Negate;
import org.nblas.function.predefined.unary.Reciprocal;


/**
 * Created by Nico
 */
public class MatrixFunctions {
	    
    public static AFunctionObject sigmoid(AFunctionObject first) {
		return new Reciprocal(new Add(new Value(1.0), new Exp(new Negate(first))));
    }
}
