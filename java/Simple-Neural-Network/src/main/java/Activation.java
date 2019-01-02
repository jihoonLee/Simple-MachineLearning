public class Activation {
    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    public static double sigmoidDerivative(double x) {
        double y = sigmoid(x);
        return y * (1.0 - y);
    }
}
