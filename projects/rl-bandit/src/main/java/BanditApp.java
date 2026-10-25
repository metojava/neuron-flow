import activations.ActivationFunction;
import evaluation.ClassificationMetrics;
import initialization.WeightInitializer;
import layers.NeuralNetwork;
import losses.LossFunction;
import optim.SGD;
import training.Trainer;

import java.util.Random;

/**
 * Project 11 - Reinforcement learning on a multi-armed bandit.
 *
 * A small network estimates the value of each of four arms. The
 * agent acts epsilon-greedily and nudges the chosen arm's output
 * toward the observed reward (a one-step temporal-difference
 * update expressed as a supervised fit).
 */
public class BanditApp {

    private static final int ARMS = 4;
    private static final float[] TRUE_PAYOFF = {
        0.20f, 0.70f, 0.30f, 0.45f
    };
    private static final int STEPS = 4000;

    public static void main(String[] args) {
        NeuralNetwork q = new NeuralNetwork(
            new int[] {ARMS, 24, ARMS},
            new ActivationFunction[] {
                ActivationFunction.linear(),
                ActivationFunction.relu(),
                ActivationFunction.linear()
            },
            LossFunction.meanSquaredError(),
            new SGD(0.05f),
            42L,
            WeightInitializer.heNormal());

        Random rand = new Random(42L);
        float[] state = {1.0f, 1.0f, 1.0f, 1.0f};
        float epsilon = 0.20f;
        float totalReward = 0.0f;

        System.out.println("step   avg reward   eps");
        for (int step = 1; step <= STEPS; step++) {
            float[] values = q.predict(state);
            int arm = rand.nextFloat() < epsilon
                ? rand.nextInt(ARMS)
                : ClassificationMetrics.argmax(values);

            float reward = TRUE_PAYOFF[arm]
                + 0.05f * (float) rand.nextGaussian();
            totalReward += reward;

            float[] target = values.clone();
            target[arm] = reward;
            q.fit(new Trainer(1, 1, false, step),
                new float[][] {state}, new float[][] {target});

            if (step % 400 == 0) {
                System.out.printf("%4d   %.3f        %.2f%n",
                    step, totalReward / step, epsilon);
            }
            epsilon = Math.max(0.01f, epsilon * 0.999f);
        }

        float[] estimated = q.predict(state);
        System.out.println();
        System.out.print("estimated Q: ");
        print(estimated);
        System.out.print("true payoffs: ");
        print(TRUE_PAYOFF);
    }

    private static void print(float[] values) {
        System.out.print("[");
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                System.out.print(" ");
            }
            System.out.printf("%.2f", values[i]);
        }
        System.out.println("]");
    }
}
