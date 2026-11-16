package fastlowess;

/**
 * Shared argument checks for public fit / stream / predict entry points.
 */
final class Inputs {

    private Inputs() {
    }

    static void requireXy(double[] x, double[] y) {
        if (x == null || y == null) {
            throw new IllegalArgumentException("x and y must not be null");
        }
        if (x.length != y.length) {
            throw new IllegalArgumentException("x and y must have the same length");
        }
        if (x.length == 0) {
            throw new IllegalArgumentException("x and y must not be empty");
        }
    }

    static void requireQueryX(double[] newX) {
        if (newX == null || newX.length == 0) {
            throw new IllegalArgumentException("newX must not be null or empty");
        }
    }

    static void requireCustomWeights(double[] customWeights, int yLength) {
        if (customWeights == null) {
            return;
        }
        if (customWeights.length == 0) {
            throw new IllegalArgumentException("customWeights must not be empty");
        }
        if (customWeights.length != yLength) {
            throw new IllegalArgumentException("customWeights must match y length");
        }
    }
}
