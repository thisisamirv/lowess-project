const test = require('node:test');
const assert = require('node:assert');

// Import WASM bindings using require (works in Node with generated pkg)
const fastlowess = require('../../bindings/wasm/pkg/fastlowess_wasm.js');

test('WASM batch smoothing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const result = fastlowess.smooth(x, y, {
        fraction: 0.3,
        return_diagnostics: true
    });

    assert.strictEqual(result.x.length, 5);
    assert.strictEqual(result.y.length, 5);
    // Check diagnostics using getters
    assert.ok(result.diagnostics.rmse < 0.1);
});

test('WASM streaming smoothing', () => {
    const streamer = new fastlowess.StreamingLowessWasm({
        fraction: 0.3
    }, {
        chunk_size: 10,
        overlap: 2
    });

    const x = new Float64Array(Array.from({ length: 20 }, (_, i) => i));
    const y = new Float64Array(Array.from({ length: 20 }, (_, i) => i * 2));

    const result = streamer.process_chunk(x, y);
    // WASM processChunk returns a struct, safe to check .y existence/length if populated
    if (result) {
        assert.ok(result.y.length >= 0);
    }

    const finalResult = streamer.finalize();
    if (finalResult) {
        assert.ok(finalResult.y.length > 0);
    }
});

test('WASM online smoothing', () => {
    const online = new fastlowess.OnlineLowessWasm({
        fraction: 0.5
    }, {
        window_capacity: 10,
        min_points: 2
    });

    let lastVal;
    for (let i = 0; i < 10; i++) {
        lastVal = online.update(i, i * 2);
    }

    // lastVal should not be undefined/null after enough points
    assert.ok(lastVal !== undefined && lastVal !== null);
    assert.ok(Math.abs(lastVal - 18) < 1.0);
});

test('WASM options parsing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const result = fastlowess.smooth(x, y, {
        weight_function: 'tricube',
        robustness_method: 'bisquare',
        boundary_policy: 'extend',
        scaling_method: 'mad'
    });

    assert.strictEqual(result.y.length, 5);
});

test('WASM custom_weights: uniform weights match no weights', () => {
    const n = 20;
    const x = new Float64Array(Array.from({ length: n }, (_, i) => i * 0.5));
    const y = new Float64Array(x.map(v => Math.sin(v)));
    const weights = Array.from({ length: n }, () => 1.0);

    const result_no_w = fastlowess.smooth(x, y, { fraction: 0.4, iterations: 2 });
    const result_unit_w = fastlowess.smooth(x, y, {
        fraction: 0.4, iterations: 2, custom_weights: weights
    });

    for (let i = 0; i < n; i++) {
        assert.ok(
            Math.abs(result_no_w.y[i] - result_unit_w.y[i]) < 1e-10,
            `y[${i}] diverges: ${result_no_w.y[i]} vs ${result_unit_w.y[i]}`
        );
    }
});

test('WASM custom_weights: zero weight reduces outlier influence', () => {
    const n = 10;
    const x = new Float64Array(Array.from({ length: n }, (_, i) => i));
    const y = new Float64Array(x.map(v => v * 2.0));
    y[5] = 100.0;  // outlier

    const weights = Array.from({ length: n }, () => 1.0);
    weights[5] = 0.0;

    const result_no_w = fastlowess.smooth(x, y, { fraction: 0.5, iterations: 0 });
    const result_zero_w = fastlowess.smooth(x, y, {
        fraction: 0.5, iterations: 0, custom_weights: weights
    });

    const true_val = 5.0 * 2.0;
    const err_no_w = Math.abs(result_no_w.y[5] - true_val);
    const err_zero_w = Math.abs(result_zero_w.y[5] - true_val);

    assert.ok(
        err_zero_w < err_no_w,
        `zero weight should reduce error (no_w=${err_no_w.toFixed(2)}, zero_w=${err_zero_w.toFixed(2)})`
    );
});

test('WASM custom_weights: wrong length throws error', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    assert.throws(() => {
        fastlowess.smooth(x, y, { fraction: 0.5, custom_weights: [1, 1, 1] });
    });
});
