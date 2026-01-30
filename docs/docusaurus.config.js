// @ts-check
import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'HiveFrame',
  tagline: 'Bee-inspired distributed data processing framework',
  favicon: 'img/favicon.ico',

  url: 'https://hiveframe.readthedocs.io',
  baseUrl: '/',

  organizationName: 'hiveframe',
  projectName: 'hiveframe',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  markdown: {
    mermaid: true,
  },

  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/hiveframe/hiveframe/tree/main/docs/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/hiveframe-social-card.jpg',
      navbar: {
        title: 'HiveFrame',
        logo: {
          alt: 'HiveFrame Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialsSidebar',
            position: 'left',
            label: 'Tutorials',
          },
          {
            type: 'docSidebar',
            sidebarId: 'howtoSidebar',
            position: 'left',
            label: 'How-To Guides',
          },
          {
            type: 'docSidebar',
            sidebarId: 'explanationSidebar',
            position: 'left',
            label: 'Explanation',
          },
          {
            type: 'docSidebar',
            sidebarId: 'referenceSidebar',
            position: 'left',
            label: 'Reference',
          },
          {
            href: 'https://github.com/hiveframe/hiveframe',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Learn',
            items: [
              {
                label: 'Tutorials',
                to: '/docs/tutorials/getting-started',
              },
              {
                label: 'How-To Guides',
                to: '/docs/how-to/read-write-parquet',
              },
            ],
          },
          {
            title: 'Understand',
            items: [
              {
                label: 'Core Concepts',
                to: '/docs/explanation/waggle-dance-protocol',
              },
              {
                label: 'Architecture',
                to: '/docs/explanation/architecture-overview',
              },
            ],
          },
          {
            title: 'Reference',
            items: [
              {
                label: 'API Reference',
                to: '/docs/reference/core',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/hiveframe/hiveframe',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} HiveFrame. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'yaml', 'json'],
      },
      mermaid: {
        theme: { light: 'neutral', dark: 'dark' },
      },
    }),
};

export default config;
