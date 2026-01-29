import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * HiveFrame documentation sidebar following the Diataxis framework:
 * - Tutorials: Learning-oriented
 * - How-to Guides: Problem-oriented
 * - Reference: Information-oriented
 * - Explanation: Understanding-oriented
 */
const sidebars: SidebarsConfig = {
  docs: [
    'intro',
    {
      type: 'category',
      label: '📚 Tutorials',
      description: 'Learning-oriented lessons to get you started',
      collapsed: false,
      items: [
        'tutorials/getting-started',
        'tutorials/dataframe-operations',
        'tutorials/streaming-basics',
      ],
    },
    {
      type: 'category',
      label: '🔧 How-to Guides',
      description: 'Problem-oriented recipes for common tasks',
      collapsed: false,
      items: [
        'how-to-guides/overview',
        'how-to-guides/deploy-production',
      ],
    },
    {
      type: 'category',
      label: '📖 Reference',
      description: 'Information-oriented technical descriptions',
      collapsed: false,
      items: [
        'reference/api-overview',
      ],
    },
    {
      type: 'category',
      label: '💡 Explanation',
      description: 'Understanding-oriented discussions',
      collapsed: false,
      items: [
        'explanation/bee-colony-metaphor',
        'explanation/waggle-dance',
      ],
    },
  ],
};

export default sidebars;
