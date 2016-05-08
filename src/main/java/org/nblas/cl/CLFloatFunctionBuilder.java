package org.nblas.cl;

import org.jocl.cl_kernel;
import org.nblas.function.AFunctionBuilder;
import org.nblas.function.ArgumentType;
import org.nblas.generic.Subprogram;


class CLFloatFunctionBuilder extends AFunctionBuilder<cl_kernel> {

	protected CLContext context;
	public CLFloatFunctionBuilder(CLContext context) {
		this.context = context;
	}
	
	@Override
    protected Subprogram<cl_kernel> buildFunction(String name, String function, ArgumentType[] args) {
        StringBuilder builder = new StringBuilder();
        StringBuilder parameters = new StringBuilder();
        
        for (int i = 0; i < args.length; i++) {
            String id = "arg" + String.format("%02d", i);
            if (args[i] == ArgumentType.SCALAR) {
            	 parameters.append(", const float arg" + String.format("%02d", i));
            } else {
            	
            	if (args[i] == ArgumentType.MATRIX) {
                    function = function.replaceAll(id, id + "[idx]");
                } else if (args[i] == ArgumentType.COLUMN_VECTOR) {
                    function = function.replaceAll(id, id + "[row]");
                } else if (args[i] == ArgumentType.ROW_VECTOR) {
                    function = function.replaceAll(id, id + "[column]");
                }  
                parameters.append(", __global const float* arg" + String.format("%02d", i));
            }           
        }

        String functionName = generateFunctionName(name, args);
//        String functionName = generateFunctionName(function);
        builder.append("__kernel void " + functionName + "(");
        builder.append("__global float* result");
        builder.append(", const uint rows");
        builder.append(", const uint columns");
        builder.append(parameters.toString());

        builder.append(")\n");
        builder.append("{\n");

        builder.append("uint column = get_global_id(0);\n"); 
        builder.append("uint row = get_global_id(1);\n"); 
        
        // TODO: teuer. sollte ganz raus
        if(args.length > 1 && (args[1] == ArgumentType.COLUMN_VECTOR || args[1] == ArgumentType.ROW_VECTOR))
        	builder.append("if(row >= rows || column >= columns ) return;\n");
        
        builder.append("uint idx = column * get_global_size(1) + row;\n");
        builder.append("result[idx] = ");
        builder.append(function);
        builder.append(";\n");
        builder.append("}\n");

        return new Subprogram<cl_kernel>(functionName, builder.toString(), true);
    }

    @Override
    protected CLContext getContext() {
        return context;
    }
}
