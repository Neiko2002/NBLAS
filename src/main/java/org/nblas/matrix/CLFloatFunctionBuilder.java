package org.nblas.matrix;

import org.nblas.function.ArgumentType;
import org.nblas.function.Context;
import org.nblas.function.functionobjects.generic.AFunctionObject;


class CLFloatFunctionBuilder extends AFunctionBuilder {

    protected String buildFunction(String function, ArgumentType[] args) {
        StringBuilder builder = new StringBuilder();
        StringBuilder parameters = new StringBuilder();
        for (int i = 0; i < args.length; i++) {

            String id = "arg" + String.format("%02d", i);
            if (args[i] == ArgumentType.MATRIIX) {
                function = function.replaceAll(id, id + "[id]");
            } else if (args[i] == ArgumentType.COLUMN_VECTOR) {
                function = function.replaceAll(id, id + "[id0]");
            } else if (args[i] == ArgumentType.ROW_VECTOR) {
                function = function.replaceAll(id, id + "[id1]");
            } else if (args[i] == ArgumentType.SCALAR) {
                function = function.replaceAll(id, id + "[0]");
            }
            parameters.append(", __global const float* arg" + String.format("%02d", i));
        }

        functionName = generateFunctionName(function);

        builder.append("__kernel void " + functionName + "(");
        builder.append("__global float* result");
        builder.append(", const uint columns");
        builder.append(", const uint rows");
        builder.append(parameters.toString());

        builder.append(")\n");
        builder.append("{\n");

        builder.append("uint id0 = get_global_id(0);\n");
        builder.append("uint id1 = get_global_id(1);\n");
        builder.append("if(id0 >= rows || id1 >= columns ) return;\n");
        builder.append("uint id = id1 * get_global_size(0) + id0;\n");
        builder.append("result[id] = ");
        builder.append(function);
        builder.append(";\n");
        builder.append("}\n");

        return builder.toString();
    }

    @Override
    protected Context getContext() {
        return Context.createOpenCLSinglePrecisionContext();
    }
}
