import { defineCollection, z } from 'astro:content';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';

export const collections = {
    docs: defineCollection({
        loader: docsLoader(),
        // title default covers TypeDoc's root index page, which has no frontmatter title
        schema: docsSchema({ extend: z.object({ title: z.string().default('API Reference') }) }),
    }),
    i18n: defineCollection({
        loader: i18nLoader(),
        schema: i18nSchema(),
    }),
};
