/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // Tutorials - Learning-oriented (for newcomers)
  tutorialsSidebar: [
    {
      type: 'category',
      label: 'Tutorials',
      link: {
        type: 'doc',
        id: 'tutorials/index',
      },
      items: [
        'tutorials/getting-started',
        'tutorials/first-pipeline',
        'tutorials/dataframe-basics',
        'tutorials/streaming-application',
        'tutorials/sql-analytics',
        'tutorials/kubernetes-deployment',
      ],
    },
  ],

  // How-To Guides - Task-oriented (for practitioners)
  howtoSidebar: [
    {
      type: 'category',
      label: 'How-To Guides',
      link: {
        type: 'doc',
        id: 'how-to/index',
      },
      items: [
        {
          type: 'category',
          label: 'Data Storage',
          items: [
            'how-to/read-write-parquet',
            'how-to/use-delta-lake',
            'how-to/delta-time-travel',
          ],
        },
        {
          type: 'category',
          label: 'Resilience',
          items: [
            'how-to/configure-retry',
            'how-to/use-circuit-breaker',
            'how-to/handle-errors-dlq',
          ],
        },
        {
          type: 'category',
          label: 'Monitoring',
          items: [
            'how-to/setup-monitoring',
            'how-to/configure-logging',
            'how-to/enable-tracing',
          ],
        },
        {
          type: 'category',
          label: 'Streaming',
          items: [
            'how-to/configure-windows',
            'how-to/manage-watermarks',
            'how-to/delivery-guarantees',
          ],
        },
        {
          type: 'category',
          label: 'Connectors',
          items: [
            'how-to/connect-kafka',
            'how-to/connect-postgres',
            'how-to/connect-http',
          ],
        },
      ],
    },
  ],

  // Explanation - Understanding-oriented (for deeper knowledge)
  explanationSidebar: [
    {
      type: 'category',
      label: 'Explanation',
      link: {
        type: 'doc',
        id: 'explanation/index',
      },
      items: [
        'explanation/architecture-overview',
        'explanation/waggle-dance-protocol',
        'explanation/three-tier-workers',
        'explanation/abc-optimization',
        'explanation/streaming-windows-watermarks',
        'explanation/pheromone-signaling',
        'explanation/colony-temperature',
        'explanation/comparison-spark',
      ],
    },
  ],

  // Reference - Information-oriented (for looking things up)
  referenceSidebar: [
    {
      type: 'category',
      label: 'API Reference',
      link: {
        type: 'doc',
        id: 'reference/index',
      },
      items: [
        'reference/core',
        'reference/dataframe',
        'reference/sql',
        'reference/streaming',
        'reference/storage',
        'reference/resilience',
        'reference/connectors',
        'reference/monitoring',
        'reference/kubernetes',
        'reference/dashboard',
        'reference/exceptions',
      ],
    },
  ],
};

export default sidebars;
