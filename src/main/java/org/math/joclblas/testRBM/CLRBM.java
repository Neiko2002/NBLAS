package org.math.joclblas.testRBM;

import org.jblas.FloatMatrix;
import org.math.joclblas.CLFloatMatrix;
import org.math.joclblas.CLRandomField;

import java.util.Optional;


public class CLRBM {

    static {
        CLFloatMatrix.defineUnaryElementwiseFunctionX("sigmoid", "1.0f / (exp(-x) + 1.0f)");
        CLFloatMatrix.defineUnaryElementwiseFunctionX("rectify", "sum(0.0f, x)");
        CLFloatMatrix.defineBinaryElementwiseFunctionXY("quantize", "(x > y) ? 1.0f : 0.0f");
        CLFloatMatrix.defineBinaryElementwiseFunctionXY("nrelu", " sum(0.0f, x + y / (exp(-x) + 1.0f))");
    }


    private final CLFloatMatrix weights;
    private final CLFloatMatrix weightsT;


    private Optional<CLRandomField> randomField;
    private Optional<CLFloatMatrix> noise;
    private CLFloatMatrix hidden;
    private CLFloatMatrix visible;
    private CLFloatMatrix visibleT;
    private CLFloatMatrix visibleBias;
    private CLFloatMatrix positiveAssociations;
    private CLFloatMatrix negativeAssociations;
    private CLFloatMatrix batch;
    private CLFloatMatrix batchT;

    private final int batchSize;


    private final RBMUnits hiddenUnits;
    private final RBMUnits visibleUnits;

    private int cd_k = 1;
    private ICLFloatMatrixUnitFunction hiddenFunction1;
    private ICLFloatMatrixUnitFunction gibbsSampleFunction;
    private ICLFloatMatrixUnitFunction visibleFunction;
    private ICLFloatMatrixUnitFunction hiddenFunction2;

    public CLRBM(FloatMatrix weights, int batchSize, RBMUnits hiddenUnits, RBMUnits visibleUnits) {

        this.hiddenUnits = hiddenUnits;
        this.visibleUnits = visibleUnits;

        this.weights = new CLFloatMatrix(weights.getRows(), weights.getColumns(), weights.data);
        this.weightsT = CLFloatMatrix.zeros(weights.getColumns(), weights.getRows());

        this.batchSize = batchSize;

        init();
    }

    private void init() {
        this.hidden = CLFloatMatrix.zeros(this.batchSize, weights.getColumns());

        this.visible = CLFloatMatrix.zeros(this.batchSize, this.weights.getRows());
        this.visibleT = CLFloatMatrix.zeros(this.weights.getRows(), this.batchSize);
        this.visibleBias = CLFloatMatrix.ones(this.batchSize, 1);

        this.positiveAssociations = CLFloatMatrix.zeros(this.weights.getRows(), weights.getColumns());
        this.negativeAssociations = CLFloatMatrix.zeros(this.weights.getRows(), weights.getColumns());

        this.batch = CLFloatMatrix.ones(this.batchSize, this.weights.getRows());
        this.batchT = CLFloatMatrix.zeros(this.weights.getRows(), this.batchSize);


        switch (this.hiddenUnits) {
            case REAL:
                noise = Optional.empty();
                randomField = Optional.empty();
                hiddenFunction1 = hiddenFunction2 = (a) -> {
                    return;
                };
                gibbsSampleFunction = (a) -> {
                    return;
                };

                break;
            case BINARY:
                noise = Optional.of(CLFloatMatrix.zeros(this.batchSize, weights.getColumns()));
                randomField = Optional.of(new CLRandomField(this.batchSize, weights.getColumns()));
                hiddenFunction1 = (units) -> {
                    CLFloatMatrix.applyUnaryFunction("sigmoid", units, units);
                    randomField.get().nextUniform(noise.get());
                    CLFloatMatrix.applyBinaryFunction("quantize", units, noise.get(), units);
                };
                hiddenFunction2 = (units) -> CLFloatMatrix.applyUnaryFunction("sigmoid", units, units);
                gibbsSampleFunction = (gibbs) -> {
                    return;
                };
                break;
            case RECTIFIED:
                noise = Optional.of(CLFloatMatrix.zeros(this.weights.getRows(), weights.getColumns()));
                randomField = Optional.of(new CLRandomField(this.weights.getRows(), weights.getColumns()));
                hiddenFunction1 = hiddenFunction2 = (units) -> CLFloatMatrix.applyUnaryFunction("rectify", units, units);
                gibbsSampleFunction = (gibbs) -> {
                    randomField.get().nextGaussian(noise.get());
                    CLFloatMatrix.applyBinaryFunction("nrelu", gibbs, noise.get(), gibbs);
                };
                break;
            default:
                noise = Optional.empty();
                hiddenFunction1 = hiddenFunction2 = (units) -> {
                    return;
                };
                gibbsSampleFunction = (gibbs) -> {
                    return;
                };
                break;
        }
        switch (this.visibleUnits) {
            case REAL:
                visibleFunction = (units) -> {
                    return;
                };
                break;
            case BINARY:
                visibleFunction = (units) -> CLFloatMatrix.applyUnaryFunction("sigmoid", units, units);
                break;
            case RECTIFIED:
                visibleFunction = (units) -> CLFloatMatrix.applyUnaryFunction("rectify", units, units);
                break;
            default:
                visibleFunction = (units) -> {
                    return;
                };
                break;
        }
    }


    public void train(FloatMatrix batchWithoutBias, float learningRate) {

        CLFloatMatrix clbatch = new CLFloatMatrix(batchWithoutBias.getRows(), batchWithoutBias.getColumns(), batchWithoutBias.data);
        this.batch.setSubMatrix(clbatch, 0, 1);
        clbatch.release();

        float invertedBatchSize = 1.0f / (batch.getRows() * cd_k);
        float v = weights.getColumns() / (float) weights.getRows();
        if (weights.getColumns() > weights.getRows()) {
            v = 1;
        }
        float actualLearningRate = learningRate * invertedBatchSize * v;

        train(actualLearningRate);

    }

    private void train(float learningRate) {

        // positive phase
        CLFloatMatrix.mmul(batch, weights, hidden);
        hiddenFunction1.apply(hidden);

        // gibbs positive
        CLFloatMatrix.transpose(batch, batchT);
        CLFloatMatrix.mmul(batchT, hidden, positiveAssociations);
        gibbsSampleFunction.apply(positiveAssociations);

        // cd-k
        for (int k = 0; k < cd_k; k++) {

            // visible
            CLFloatMatrix.transpose(weights, weightsT);
            CLFloatMatrix.mmul(hidden, weightsT, visible);
            visibleFunction.apply(visible);
            visible.setSubMatrix(visibleBias, 0, 0);

            // negative phase
            CLFloatMatrix.mmul(visible, weights, hidden);
            hiddenFunction2.apply(hidden);
        }

        // gibbs negative
        CLFloatMatrix.transpose(visible, visibleT);
        CLFloatMatrix.mmul(visibleT, hidden, negativeAssociations);
        gibbsSampleFunction.apply(negativeAssociations);

        // cd
        CLFloatMatrix.sub(positiveAssociations, negativeAssociations, positiveAssociations);
        CLFloatMatrix.mul(positiveAssociations, learningRate, positiveAssociations);
        CLFloatMatrix.add(weights, positiveAssociations, weights);
    }

    public FloatMatrix getHidden(FloatMatrix batchWithoutBias) {

        CLFloatMatrix clbatchWithoutBias = new CLFloatMatrix(batchWithoutBias.getRows(), batchWithoutBias.getColumns(), batchWithoutBias.data);
        CLFloatMatrix clbatch = CLFloatMatrix.ones(batchWithoutBias.getRows(), batchWithoutBias.getColumns() + 1);
        clbatch.setSubMatrix(clbatchWithoutBias, 0, 1);
        clbatchWithoutBias.release();

        CLFloatMatrix clHidden = CLFloatMatrix.zeros(clbatch.getRows(), weights.getColumns());

        CLFloatMatrix.mmul(clbatch, weights, clHidden);
        hiddenFunction2.apply(clHidden);

        FloatMatrix hidden = new FloatMatrix(clbatch.getRows(), weights.getColumns(), clHidden.toArray());
        clHidden.release();
        clbatch.release();
        return hidden.getRange(0, clHidden.getRows(), 1, clHidden.getColumns());
    }

    public FloatMatrix getVisible(FloatMatrix hiddenWithoutBias) {

        CLFloatMatrix clHiddenWithoutBias = new CLFloatMatrix(hiddenWithoutBias.getRows(), hiddenWithoutBias.getColumns(), hiddenWithoutBias.data);
        CLFloatMatrix clHidden = CLFloatMatrix.ones(hiddenWithoutBias.getRows(), hiddenWithoutBias.getColumns() + 1);
        clHidden.setSubMatrix(clHiddenWithoutBias, 0, 1);
        clHiddenWithoutBias.release();

        CLFloatMatrix clVisible = CLFloatMatrix.zeros(clHidden.getRows(), weights.getRows());

        CLFloatMatrix.transpose(weights, weightsT);
        CLFloatMatrix.mmul(clHidden, weightsT, clVisible);
        visibleFunction.apply(clVisible);

        FloatMatrix visible = new FloatMatrix(clVisible.getRows(), weights.getRows(), clVisible.toArray());

        clHidden.release();
        clVisible.release();

        return visible.getRange(0, visible.getRows(), 1, visible.getColumns());
    }

    public int getColumns() {
        return weights.getColumns() - 1;
    }

    public float getError() {
        return 0;
    }

    public FloatMatrix getWeights() {
        return new FloatMatrix(this.weights.getRows(), this.weights.getColumns(), weights.toArray());
    }

    public int getBatchSize() {
        return batchSize;
    }

    public RBMUnits getHiddenUnitsType() {
        return hiddenUnits;
    }

    public RBMUnits getVisibleUnitsType() {
        return visibleUnits;
    }

    protected void finalize() throws Throwable {
        releaseCLMatrices();

        super.finalize();
    }

    private void releaseCLMatrices() {
        hidden.release();
        visible.release();
        positiveAssociations.release();
        negativeAssociations.release();
        batch.release();
        batchT.release();
        visibleT.release();
        visibleBias.release();

        if (randomField.isPresent()) {
            randomField.get().release();
            noise.get().release();
        }
    }
}
