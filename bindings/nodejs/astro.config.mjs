import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
    base: process.env.VITE_BASE ?? '/',
    outDir: './dist',
    integrations: [
        starlight({
            title: 'fastlowess',
            description: 'High-performance LOWESS smoothing for Node.js',
            sidebar: [
                {
                    label: 'Getting Started',
                    items: [
                        { label: 'Installation', slug: 'installation' },
                        { label: 'Quick Start', slug: 'quickstart' },
                        { label: 'Concepts', slug: 'concepts' },
                        { label: 'Parameters', slug: 'parameters' },
                    ],
                },
                {
                    label: 'Adapters',
                    items: [
                        { label: 'Adapter Choice', slug: 'adapter-choice' },
                        { label: 'GPU Backend', slug: 'gpu-backend' },
                    ],
                },
                {
                    label: 'Usage',
                    items: [
                        { label: 'Batch', slug: 'batch' },
                        { label: 'Streaming', slug: 'streaming' },
                        { label: 'Online', slug: 'online' },
                    ],
                },
                {
                    label: 'Analysis',
                    items: [
                        { label: 'Intervals', slug: 'intervals' },
                        { label: 'Cross-Validation', slug: 'cross-validation' },
                    ],
                },
                {
                    label: 'Customization',
                    items: [
                        { label: 'Kernels', slug: 'kernels' },
                        { label: 'Robustness', slug: 'robustness' },
                        { label: 'Scaling', slug: 'scaling' },
                        { label: 'Custom Weights', slug: 'custom-weights' },
                        { label: 'Boundary', slug: 'boundary' },
                        { label: 'Merge', slug: 'merge' },
                    ],
                },
                {
                    label: 'Use Cases',
                    items: [
                        { label: 'Genomics', slug: 'use-case-genomics' },
                        { label: 'Time Series', slug: 'use-case-time-series' },
                        { label: 'Real-Time', slug: 'use-case-real-time' },
                    ],
                },
                {
                    label: 'API Guide',
                    items: [
                        { label: 'Batch API', slug: 'api' },
                        { label: 'Streaming API', slug: 'api-streaming' },
                        { label: 'Online API', slug: 'api-online' },
                    ],
                },
                { label: 'API Reference', slug: 'reference/' },
            ],
        }),
    ],
});
