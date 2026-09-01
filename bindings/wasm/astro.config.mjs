import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
    site: process.env.SITE ?? 'https://thisisamirv.github.io',
    base: process.env.VITE_BASE ?? '/',
    outDir: './dist',
    markdown: {
        remarkPlugins: [remarkMath],
        rehypePlugins: [rehypeKatex],
    },
    integrations: [
        starlight({
            title: 'fastlowess-wasm',
            description: 'High-performance LOWESS smoothing for WebAssembly',
            customCss: ['./src/styles/katex.css'],
            expressiveCode: {
                shiki: { langAlias: { output: 'text' } },
            },
            sidebar: [
                {
                    label: 'Getting Started',
                    items: [
                        { label: 'Installation', slug: 'introduction/installation' },
                        { label: 'Quick Start', slug: 'introduction/quickstart' },
                        { label: 'Concepts', slug: 'introduction/concepts' },
                        { label: 'Benchmarks', slug: 'benchmarks' },
                    ],
                },
                {
                    label: 'User Guide',
                    items: [
                        { label: 'Adapter Choice', slug: 'guide/adapter-choice' },
                        { label: 'Intervals', slug: 'guide/intervals' },
                        { label: 'Cross-Validation', slug: 'guide/cross-validation' },
                    ],
                },
                {
                    label: 'Weight & Robustness',
                    items: [
                        { label: 'Kernels', slug: 'weighting/kernels' },
                        { label: 'Robustness', slug: 'weighting/robustness' },
                        { label: 'Scaling', slug: 'weighting/scaling' },
                        { label: 'Custom Weights', slug: 'weighting/custom-weights' },
                    ],
                },
                {
                    label: 'Advanced',
                    items: [
                        { label: 'Boundary', slug: 'advanced/boundary' },
                        { label: 'Merge', slug: 'advanced/merge' },
                    ],
                },
                {
                    label: 'Use Cases',
                    items: [
                        { label: 'Genomics', slug: 'use-case/use-case-genomics' },
                        { label: 'Time Series', slug: 'use-case/use-case-time-series' },
                        { label: 'Real-Time', slug: 'use-case/use-case-real-time' },
                    ],
                },
                {
                    label: 'API Guide',
                    items: [
                        { label: 'Batch API', slug: 'api/api' },
                        { label: 'Streaming API', slug: 'api/api-streaming' },
                        { label: 'Online API', slug: 'api/api-online' },
                    ],
                },
                { label: 'API Reference', slug: 'reference' },
                { label: 'News', slug: 'news' },
            ],
        }),
    ],
});
