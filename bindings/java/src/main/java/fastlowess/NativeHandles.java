package fastlowess;

import java.lang.ref.Cleaner;
import java.util.function.LongConsumer;

/**
 * Owns a native handle with idempotent close and a {@link Cleaner} safety net.
 */
final class NativeHandles {

    private static final Cleaner CLEANER = Cleaner.create();

    private NativeHandles() {
    }

    static final class Owned implements AutoCloseable {
        private final State state;
        private final Cleaner.Cleanable cleanable;

        Owned(long handle, LongConsumer freer) {
            if (handle == 0L) {
                throw new IllegalArgumentException("native handle must be non-zero");
            }
            this.state = new State(handle, freer);
            this.cleanable = CLEANER.register(this, state);
        }

        long get() {
            return state.handle();
        }

        boolean isOpen() {
            return state.handle() != 0L;
        }

        @Override
        public void close() {
            cleanable.clean();
        }

        private static final class State implements Runnable {
            private final LongConsumer freer;
            private long handle;

            State(long handle, LongConsumer freer) {
                this.handle = handle;
                this.freer = freer;
            }

            synchronized long handle() {
                return handle;
            }

            @Override
            public synchronized void run() {
                if (handle != 0L) {
                    freer.accept(handle);
                    handle = 0L;
                }
            }
        }
    }
}
