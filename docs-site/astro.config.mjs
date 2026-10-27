// @ts-check
import starlight from '@astrojs/starlight';
import { defineConfig } from 'astro/config';

// The documentation at https://anpr.mefby.com, published to GitHub Pages with a custom domain (set in
// the repository's Pages settings, not with a CNAME file, because the site deploys through GitHub
// Actions). The site lives at the root, so internal links start with `/`.
export default defineConfig({
  site: 'https://anpr.mefby.com',
  integrations: [
    starlight({
      title: 'anpr-pipeline',
      description:
        'License-plate recognition from detection to privacy-aware storage: a Python package, a CLI and a FastAPI HTTP and WebSocket service.',
      logo: { src: './src/assets/logo.svg' },
      favicon: '/favicon.svg',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR',
        },
      ],
      editLink: {
        baseUrl:
          'https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/edit/main/docs-site/',
      },
      locales: {
        root: { label: 'English', lang: 'en' },
        tr: { label: 'Türkçe', lang: 'tr' },
      },
      customCss: ['./src/styles/custom.css'],
      // src/pages/404.astro renders the 404 page instead.
      disable404Route: true,
      sidebar: [
        {
          label: 'Get started',
          translations: { tr: 'Başlangıç' },
          items: ['start/install', 'start/first-plate'],
        },
        {
          label: 'Guide',
          translations: { tr: 'Kılavuz' },
          items: [
            'guide/cli',
            'guide/http-api',
            'guide/websocket',
            'guide/watchlist',
            'guide/configuration',
          ],
        },
        {
          label: 'Operations',
          translations: { tr: 'İşletim' },
          items: ['operations/privacy', 'operations/deployment', 'operations/monitoring'],
        },
        {
          label: 'Development',
          translations: { tr: 'Geliştirme' },
          items: ['development/architecture', 'development/contributing'],
        },
        {
          label: 'Project',
          translations: { tr: 'Proje' },
          items: ['roadmap'],
        },
      ],
    }),
  ],
});
