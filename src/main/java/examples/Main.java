package examples;

/**
 * Entry point that dispatches to a specific demo based on the
 * first command-line argument. With no argument, runs the
 * Fashion-MNIST demo (auto-downloads data).
 */
public class Main {

    public static void main(String[] args) throws Exception {
        String task = args.length > 0 ? args[0] : "fashion";

        switch (task) {
            case "fashion":
                FashionMNISTRunner.main(args);
                break;
            case "mnist":
                MNISTRunner.main(args);
                break;
            case "ml":
                ClassicalMlDemo.main(args);
                break;
            default:
                System.out.println("Unknown task: " + task);
                System.out.println("Usage: Main [fashion|mnist|ml]");
        }
    }
}
