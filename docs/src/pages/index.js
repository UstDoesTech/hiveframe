import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          🐝 {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/tutorials/getting-started">
            Get Started in 5 Minutes ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

const FeatureList = [
  {
    title: '🎯 Tutorials',
    description: 'Step-by-step guides to learn HiveFrame from scratch. Build your first pipeline and master the bee colony paradigm.',
    link: '/docs/tutorials/getting-started',
    linkText: 'Start Learning',
  },
  {
    title: '📋 How-To Guides',
    description: 'Task-oriented recipes for common operations. Storage, streaming, resilience, monitoring, and more.',
    link: '/docs/how-to',
    linkText: 'Find Solutions',
  },
  {
    title: '💡 Explanation',
    description: 'Understand how HiveFrame works. Deep dives into the waggle dance protocol, ABC algorithm, and architecture.',
    link: '/docs/explanation',
    linkText: 'Learn Concepts',
  },
  {
    title: '📚 Reference',
    description: 'Complete API documentation for all modules. Core, DataFrame, SQL, Streaming, Storage, and more.',
    link: '/docs/reference',
    linkText: 'Browse API',
  },
];

function Feature({title, description, link, linkText}) {
  return (
    <div className={clsx('col col--3')}>
      <div className="feature-card text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
        <Link className="button button--primary button--sm" to={link}>
          {linkText}
        </Link>
      </div>
    </div>
  );
}

function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

function BiomimicrySection() {
  return (
    <section className={styles.biomimicry}>
      <div className="container">
        <Heading as="h2" className="text--center margin-bottom--lg">
          🐝 Biomimetic Data Processing
        </Heading>
        <div className="row">
          <div className="col col--6">
            <table>
              <thead>
                <tr>
                  <th>Bee Behavior</th>
                  <th>Software Pattern</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>🕺 Waggle Dance</td>
                  <td>Quality-weighted task distribution</td>
                </tr>
                <tr>
                  <td>👷 Three-Tier Workers</td>
                  <td>Employed (exploit), Onlooker (reinforce), Scout (explore)</td>
                </tr>
                <tr>
                  <td>🧪 Pheromone Signaling</td>
                  <td>Backpressure and rate limiting</td>
                </tr>
                <tr>
                  <td>🌡️ Colony Temperature</td>
                  <td>System load regulation (homeostasis)</td>
                </tr>
                <tr>
                  <td>🔄 Abandonment</td>
                  <td>Self-healing through ABC algorithm</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="col col--6">
            <div className="padding--lg">
              <Heading as="h3">Why Bee Colony Optimization?</Heading>
              <ul>
                <li>✅ <strong>No single point of failure</strong> - Unlike Spark's driver model</li>
                <li>✅ <strong>Emergent load balancing</strong> - Workers self-organize</li>
                <li>✅ <strong>Adaptive backpressure</strong> - Pheromone-based flow control</li>
                <li>✅ <strong>Self-healing</strong> - Automatic recovery from failures</li>
                <li>✅ <strong>Quality-aware scheduling</strong> - Best tasks get more workers</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function QuickInstall() {
  return (
    <section className={styles.quickInstall}>
      <div className="container">
        <Heading as="h2" className="text--center margin-bottom--lg">
          ⚡ Quick Install
        </Heading>
        <div className="row">
          <div className="col col--6 col--offset-3">
            <pre className={styles.codeBlock}>
              <code>pip install hiveframe</code>
            </pre>
            <p className="text--center margin-top--md">
              Optional dependencies: <code>pip install hiveframe[kafka,postgres,monitoring]</code>
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} - Bee-Inspired Data Processing`}
      description="A biomimetic distributed data processing framework. Alternative to Apache Spark using bee colony optimization.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <BiomimicrySection />
        <QuickInstall />
      </main>
    </Layout>
  );
}
