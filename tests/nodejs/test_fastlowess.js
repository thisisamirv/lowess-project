const test = require('node:test');
const assert = require('node:assert');

const fastlowess = require('../../bindings/nodejs');

test('batch smoothing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const model = new fastlowess.Lowess({
        fraction: 0.3,
        return_diagnostics: true
    });

    const result = model.fit(x, y);

    assert.strictEqual(result.x.length, 5);
    assert.strictEqual(result.y.length, 5);
    assert.ok(result.diagnostics.rmse < 0.1);
});

test('streaming smoothing', () => {
    const streamer = new fastlowess.StreamingLowess({
        fraction: 0.3
    }, {
        chunk_size: 10,
        overlap: 2
    });

    const x = new Float64Array(Array.from({ length: 20 }, (_, i) => i));
    const y = new Float64Array(Array.from({ length: 20 }, (_, i) => i * 2));

    const result = streamer.process_chunk(x, y);
    assert.ok(result.y.length >= 0);

    const finalResult = streamer.finalize();
    assert.ok(finalResult.y.length > 0);
});

test('online smoothing', () => {
    const online = new fastlowess.OnlineLowess({
        fraction: 0.5
    }, {
        window_capacity: 10,
        min_points: 2
    });

    let lastVal = null;
    for (let i = 0; i < 10; i++) {
        const xArr = new Float64Array([i]);
        const yArr = new Float64Array([i * 2]);
        const res = online.add_points(xArr, yArr);

        if (res.y.length > 0) {
            lastVal = res.y[0];
        }
    }

    assert.ok(lastVal !== null);
    assert.ok(Math.abs(lastVal - 18) < 1.0);
});

test('options parsing', () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const model = new fastlowess.Lowess({
        weight_function: 'tricube',
        robustness_method: 'bisquare',
        boundary_policy: 'extend',
        scaling_method: 'mad'
    });

    const result = model.fit(x, y);

    assert.strictEqual(result.y.length, 5);
});

test('async batch smoothing', async () => {
    const x = new Float64Array([1, 2, 3, 4, 5]);
    const y = new Float64Array([2, 4, 6, 8, 10]);

    const model = new fastlowess.Lowess({
        fraction: 0.3
    });

    if (typeof model.fit_async !== 'function') {
        console.error('Available properties on model:', Object.getOwnPropertyNames(Object.getPrototypeOf(model)));
        throw new Error('model.fit_async is not a function');
    }
    const result = await model.fit_async(x, y);

    assert.strictEqual(result.x.length, 5);
    assert.strictEqual(result.y.length, 5);
    assert.ok(result.y[0] > 0);
});
