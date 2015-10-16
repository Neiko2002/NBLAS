package org.math.joclblas;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

import java.util.Random;

/**
 * Created by Moritz on 4/10/2015.
 */
public class CLRandomField extends ACLMatrix {

    public CLRandomField(int rows, int columns) {
        init(columns, rows);
    }

    public CLRandomField(int rows, int columns, long seed) {
        RANDOM.setSeed(seed);
        init(columns, rows);
    }

    private void init(int columns, int rows) {
        this.rows = rows;
        this.columns = columns;
        this.length = rows * columns;

        int[] clDimensions = Core.getInstance().getDimensions(this.rows, this.columns);
        this.clRows = clDimensions[0];
        this.clColumns = clDimensions[1];
        this.clLength = clColumns * clRows;

        int[] clArray = createCLArray();

        Pointer pointer = Pointer.to(clArray);

        buffer = CL.clCreateBuffer(Core.getInstance().getContext(),
                CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_uint * clLength, pointer, null);
    }

    private int[] createCLArray() {
        int[] result = new int[clLength];
        for (int i = 0; i < clLength; i++) {
            result[i] = RANDOM.nextInt();
        }
        return result;
    }

    public void nextUniform(CLFloatMatrix matrix) {
        Core.getInstance().xorshift(buffer, clColumns, clRows);
        Core.getInstance().uniform(buffer, matrix.buffer, clColumns, clRows, columns, rows);
    }

    public void nextGaussian(CLFloatMatrix matrix, float mean, float variance) {
        Core.getInstance().xorshift(buffer, clColumns, clRows);
        Core.getInstance().gaussian(buffer, matrix.buffer, mean, variance, clColumns, clRows, columns, rows);
    }

    public void nextGaussian(CLFloatMatrix matrix) {
        nextGaussian(matrix, 0.0f, 1.0f);
    }


}
