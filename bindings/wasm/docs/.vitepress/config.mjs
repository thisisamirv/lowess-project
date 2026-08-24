export default {
    title: 'fastlowess-wasm',
    description: 'High-performance LOWESS smoothing for WebAssembly',
    base: process.env.VITE_BASE || '/',
    themeConfig: {
        nav: [
            { text: 'Guide', link: '/installation' },
            { text: 'API Reference', link: '/reference/' },
        ],
        sidebar: [
            {
                text: 'Getting Started',
                items: [
                    { text: 'Installation', link: '/installation' },
                    { text: 'Quick Start', link: '/quickstart' },
                    { text: 'Concepts', link: '/concepts' },
                    { text: 'Parameters', link: '/parameters' },
                ],
            },
            {
                text: 'Adapters',
                items: [
                    { text: 'Adapter Choice', link: '/adapter-choice' },
                ],
            },
            {
                text: 'Usage',
                items: [
                    { text: 'Batch', link: '/batch' },
                    { text: 'Streaming', link: '/streaming' },
                    { text: 'Online', link: '/online' },
                ],
            },
            {
                text: 'Analysis',
                items: [
                    { text: 'Intervals', link: '/intervals' },
                    { text: 'Cross-Validation', link: '/cross-validation' },
                ],
            },
            {
                text: 'Customization',
                items: [
                    { text: 'Kernels', link: '/kernels' },
                    { text: 'Robustness', link: '/robustness' },
                    { text: 'Scaling', link: '/scaling' },
                    { text: 'Custom Weights', link: '/custom-weights' },
                    { text: 'Boundary', link: '/boundary' },
                    { text: 'Merge', link: '/merge' },
                ],
            },
            {
                text: 'Use Cases',
                items: [
                    { text: 'Genomics', link: '/use-case-genomics' },
                    { text: 'Time Series', link: '/use-case-time-series' },
                    { text: 'Real-Time', link: '/use-case-real-time' },
                ],
            },
            {
                text: 'API Guide',
                items: [
                    { text: 'Batch API', link: '/api' },
                    { text: 'Streaming API', link: '/api-streaming' },
                    { text: 'Online API', link: '/api-online' },
                ],
            },
            { text: 'API Reference', link: '/reference/' },
        ],
        socialLinks: [],
    },
}
