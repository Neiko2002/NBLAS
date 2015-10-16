package org.jcuclblas.matrix;

import org.jcuclblas.function.ArgumentType;
import org.jcuclblas.function.Context;


class CudaFloatFunctionBuilder extends AFunctionBuilder {


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
            parameters.append(", float* arg" + String.format("%02d", i));
        }
        functionName = "f" + generateFunctionName(function);
        builder.append("extern \"C\"\n\n__global__\nvoid " + functionName + "(");
        builder.append("float* result");
        builder.append(", const unsigned int columns");
        builder.append(", const unsigned int rows");
        builder.append(parameters.toString());

        builder.append(")\n");
        builder.append("{\n");

        builder.append("unsigned int id0 = blockIdx.x * blockDim.x + threadIdx.x;\n");
        builder.append("unsigned int id1 = blockIdx.y * blockDim.y + threadIdx.y;\n");
        builder.append("if(id0 >= rows || id1 >= columns ) return;\n");
        builder.append("unsigned int id = id1 * rows + id0;\n");
        builder.append("result[id] = ");
        builder.append(function);
        builder.append(";\n");
        builder.append("}\n");

        return builder.toString();
    }

    @Override
    protected Context getContext() {
        return Context.createCudaSinglePrecisionContext();
    }
}
