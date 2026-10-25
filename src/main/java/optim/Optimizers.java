package optim;

public class Optimizers {
    private Optimizers() {
    }

    public static SGD sgd(float lr) {
        return new SGD(lr);
    }

    public static Adam adam(float lr) {
        return new Adam(lr);
    }

    public static AdamW adamw(float lr) {
        return new AdamW(lr);
    }

    public static RMSProp rmsprop(float lr) {
        return new RMSProp(lr);
    }

    public static Adagrad adagrad(float lr) {
        return new Adagrad(lr);
    }

    public static Nadam nadam(float lr) {
        return new Nadam(lr);
    }

    public static NAG nag(float lr) {
        return new NAG(lr);
    }
}
