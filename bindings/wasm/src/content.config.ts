import { defineCollection, z } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

export const collections = {
    docs: defineCollection({
        loader: docsLoader(),
        // title default covers TypeDoc's root index page, which has no frontmatter title
        schema: docsSchema({ extend: z.object({ title: z.string().default('API Reference') }) }),
    }),
};
