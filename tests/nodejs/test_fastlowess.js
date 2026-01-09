const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fastlowess = require('../../bindings/nodejs');

test('batch smoothing', (t) => {
  const x = new Float64Array([1, 2, 3, 4, 5]);
  const y = new Float64Array([2, 4, 6, 8, 10]);
  
  const result = fastlowess.smooth(x, y, {
    fraction: 0.3,
    returnDiagnostics: true
  });
  
  assert.strictEqual(result.x.length, 5);
  assert.strictEqual(result.y.length, 5);
  assert.ok(result.diagnostics.rmse < 0.1);
});

test('streaming smoothing', (t) => {
  const streamer = new fastlowess.StreamingLowess({
    fraction: 0.3
  }, {
    chunkSize: 10,
    overlap: 2
  });
  
  const x = new Float64Array(Array.from({length: 20}, (_, i) => i));
  const y = new Float64Array(Array.from({length: 20}, (_, i) => i * 2));
  
  const result = streamer.processChunk(x, y);
  assert.ok(result.y.length >= 0);
  
  const finalResult = streamer.finalize();
  assert.ok(finalResult.y.length > 0);
});

test('online smoothing', (t) => {
  const online = new fastlowess.OnlineLowess({
    fraction: 0.5
  }, {
    windowCapacity: 10,
    minPoints: 2
  });
  
  let lastVal;
  for (let i = 0; i < 10; i++) {
    lastVal = online.update(i, i * 2);
  }
  
  assert.ok(lastVal !== null);
  assert.ok(Math.abs(lastVal - 18) < 1.0);
});

test('options parsing', (t) => {
  const x = new Float64Array([1, 2, 3, 4, 5]);
  const y = new Float64Array([2, 4, 6, 8, 10]);
  
  const result = fastlowess.smooth(x, y, {
    weightFunction: 'tricube',
    robustnessMethod: 'bisquare',
    boundaryPolicy: 'extend',
    scalingMethod: 'mad'
  });
  
  assert.strictEqual(result.y.length, 5);
});
