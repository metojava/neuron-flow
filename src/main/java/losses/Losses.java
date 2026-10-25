package losses;

public class Losses {
    private Losses() {
    }

    public static LossFunction mse() {
        return LossFunction.meanSquaredError();
    }

    public static LossFunction meanSquaredError() {
        return LossFunction.meanSquaredError();
    }

    public static LossFunction bce() {
        return LossFunction.binaryCrossEntropy();
    }

    public static LossFunction binaryCrossEntropy() {
        return LossFunction.binaryCrossEntropy();
    }

    public static LossFunction softmaxCrossEntropy() {
        return LossFunction.softmaxCrossEntropy();
    }

    public static LossFunction crossEntropy() {
        return LossFunction.crossEntropy();
    }

    public static LossFunction categoricalCrossEntropy() {
        return LossFunction.categoricalCrossEntropy();
    }

    public static LossFunction sparseCategoricalCrossEntropy() {
        return LossFunction.sparseCategoricalCrossEntropy();
    }

    public static LossFunction mae() {
        return LossFunction.mae();
    }

    public static LossFunction l1() {
        return LossFunction.l1();
    }

    public static LossFunction huber(float delta) {
        return new HuberLoss(delta);
    }

    public static LossFunction hinge() {
        return HingeLoss.of();
    }
}
