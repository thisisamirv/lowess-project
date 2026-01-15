const fastlowess = require('../../bindings/nodejs');

/**
 * fastlowess Online Smoothing Example
 * 
 * This example demonstrates online LOWESS smoothing for real-time data:
 * - Basic incremental processing with streaming data
 * - Real-time sensor data smoothing
 * - Different update modes (Full vs Incremental)
 * - Memory-bounded processing with sliding window
 */

function main() {
    console.log("=== fastlowess Online Smoothing Example ===");

    // 1. Simulate a real-time signal
    // A sine wave with changing frequency and random noise
    const nPoints = 1000;
    const x = new Float64Array(nPoints);
    const yTrue = new Float64Array(nPoints);
    const y = new Float64Array(nPoints);

    for (let i = 0; i < nPoints; i++) {
        x[i] = i;
        yTrue[i] = 20.0 + 5.0 * Math.sin(x[i] * 0.1) + 2.0 * Math.sin(x[i] * 0.02);
        // Gaussian noise
        let u1 = Math.random();
        let u2 = Math.random();
        let z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        y[i] = yTrue[i] + z * 1.2;
    }

    // Add some sudden spikes (sensor glitches)
    for (let i = 200; i < 205; i++) y[i] += 15.0;
    for (let i = 600; i < 610; i++) y[i] -= 10.0;

    console.log(`Simulating ${nPoints} real-time data points...`);

    // 2. Sequential Online Processing
    
    // Full Update Mode (higher accuracy)
    console.log("Processing with 'full' update mode...");
    const onlineFull = new fastlowess.OnlineLowess(
        { fraction: 0.3, iterations: 3 },
        { windowCapacity: 50, updateMode: "full" }
    );
    const resFull = new Float64Array(nPoints);
    for (let i = 0; i < nPoints; i++) {
        const chunkX = new Float64Array([x[i]]);
        const chunkY = new Float64Array([y[i]]);
        const res = onlineFull.addPoints(chunkX, chunkY);
        resFull[i] = res.y[0];
    }

    // Incremental Update Mode (faster for large windows)
    console.log("Processing with 'incremental' update mode...");
    const onlineInc = new fastlowess.OnlineLowess(
        { fraction: 0.3, iterations: 3 },
        { windowCapacity: 50, updateMode: "incremental" }
    );
    const resInc = new Float64Array(nPoints);
    for (let i = 0; i < nPoints; i++) {
        const chunkX = new Float64Array([x[i]]);
        const chunkY = new Float64Array([y[i]]);
        const res = onlineInc.addPoints(chunkX, chunkY);
        resInc[i] = res.y[0];
    }

    // Compare results
    console.log("\nResults Comparison:");

    // Show sample around spike area
    console.log("\nSample around spike (indices 198-208):");
    console.log("Index\tRaw\t\tTrue\t\tFull\t\tIncremental");
    for (let i = 198; i <= 208; i++) {
        // Handle initial points where output might be null (though by 198 it should be fine)
        const fullVal = isNaN(resFull[i]) ? "NaN" : resFull[i].toFixed(2);
        const incVal = isNaN(resInc[i]) ? "NaN" : resInc[i].toFixed(2);
        console.log(`${i}\t${y[i].toFixed(2)}\t\t${yTrue[i].toFixed(2)}\t\t${fullVal}\t\t${incVal}`);
    }

    // Calculate overall statistics
    // Filter out initial NaNs
    let mseFull = 0;
    let mseInc = 0;
    let count = 0;

    for (let i = 0; i < nPoints; i++) {
        if (!isNaN(resFull[i]) && !isNaN(resInc[i])) {
            mseFull += Math.pow(resFull[i] - yTrue[i], 2);
            mseInc += Math.pow(resInc[i] - yTrue[i], 2);
            count++;
        }
    }
    mseFull /= count;
    mseInc /= count;

    console.log("\nMean Squared Error vs True Signal:");
    console.log(` - Full Update:        ${mseFull.toFixed(4)}`);
    console.log(` - Incremental Update: ${mseInc.toFixed(4)}`);

    console.log("\n=== Online Smoothing Example Complete ===");
}

main();
