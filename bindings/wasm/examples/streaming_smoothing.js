const { smooth, StreamingLowessWasm } = require('../pkg/fastlowess_wasm.js');

/**
 * fastLowess WASM Streaming Smoothing Example
 * 
 * This example demonstrates streaming LOWESS smoothing for large datasets using WebAssembly:
 * - Basic chunked processing
 * - Handling datasets that don't fit in memory
 */

function main() {
    console.log("=== fastLowess WASM Streaming Mode Example ===");

    // 1. Generate Very Large Dataset
    // 100,000 points
    const nPoints = 100000;
    console.log(`Generating large dataset: ${nPoints} points...`);
    
    // Pre-allocate arrays
    const x = new Float64Array(nPoints);
    const y = new Float64Array(nPoints);
    
    for (let i = 0; i < nPoints; i++) {
        x[i] = (i / (nPoints - 1)) * 100; // range 0 to 100
        // Gaussian noise
        let u1 = Math.random();
        let u2 = Math.random();
        let z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        y[i] = Math.cos(x[i] * 0.1) + z * 0.5;
    }

    // 2. Regular Batch Smoothing (for comparison)
    // Note: WASM batch might be slower than native, but still fast
    console.log("Running Batch LOWESS (WASM)...");
    const batchStart = process.hrtime.bigint();
    const resBatch = smooth(x, y, { fraction: 0.01 });
    const batchEnd = process.hrtime.bigint();
    const batchTime = Number(batchEnd - batchStart) / 1e9;
    console.log(`Batch took: ${batchTime.toFixed(4)} seconds`);

    // 3. Streaming Mode
    // Divide the data into chunks of 2,000 for low memory usage
    console.log("Running Streaming LOWESS (WASM Chunked)...");
    const streamStart = process.hrtime.bigint();
    
    const streamer = new StreamingLowessWasm(
        { fraction: 0.01 },
        { chunkSize: 2000, overlap: 200 }
    );

    const chunkSize = 2000;
    const resChunks = [];
    
    for (let i = 0; i < nPoints; i += chunkSize) {
        const chunkX = x.subarray(i, Math.min(i + chunkSize, nPoints));
        const chunkY = y.subarray(i, Math.min(i + chunkSize, nPoints));
        
        const chunkRes = streamer.processChunk(chunkX, chunkY);
        // WASM call returns struct with .y getter
        if (chunkRes && chunkRes.y) resChunks.push(chunkRes.y);
    }
    const finalChunk = streamer.finalize();
    if (finalChunk && finalChunk.y) resChunks.push(finalChunk.y);
    
    // Combine chunks
    const totalLen = resChunks.reduce((acc, c) => acc + c.length, 0);
    const streamY = new Float64Array(totalLen);
    let offset = 0;
    for (const c of resChunks) {
        streamY.set(c, offset);
        offset += c.length;
    }

    const streamEnd = process.hrtime.bigint();
    const streamTime = Number(streamEnd - streamStart) / 1e9;
    console.log(`Streaming took: ${streamTime.toFixed(4)} seconds`);

    // 4. Verify Accuracy
    let mse = 0;
    const cmpLen = Math.min(resBatch.y.length, streamY.length);
    
    for (let i = 0; i < cmpLen; i++) {
        mse += Math.pow(resBatch.y[i] - streamY[i], 2);
    }
    mse /= cmpLen;
    
    console.log(`Mean Squared Difference (Batch vs Stream): ${mse.toExponential(2)}`);

    // Show sample of results
    console.log("\nSample comparison (indices 1000-1005):");
    console.log("Index\tBatch\t\tStreaming\tDiff");
    for (let i = 1000; i <= 1005; i++) {
        if (i < cmpLen) {
            const diff = Math.abs(resBatch.y[i] - streamY[i]);
            console.log(`${i}\t${resBatch.y[i].toFixed(6)}\t${streamY[i].toFixed(6)}\t${diff.toFixed(6)}`);
        }
    }

    console.log("\n=== WASM Streaming Smoothing Example Complete ===");
}

main();
