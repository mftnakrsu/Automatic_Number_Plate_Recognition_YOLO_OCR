import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';
import { defineCollection } from 'astro:content';

export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
  // Overrides of Starlight's built-in UI strings, one JSON file per locale.
  i18n: defineCollection({ loader: i18nLoader(), schema: i18nSchema() }),
};
