# HiveFrame Documentation

This directory contains the HiveFrame documentation built with [Docusaurus](https://docusaurus.io/).

## Documentation Structure

The documentation follows the [Diataxis](https://diataxis.fr/) framework:

- **Tutorials** (`docs/tutorials/`) - Learning-oriented guides to get you started
- **How-to Guides** (`docs/how-to-guides/`) - Problem-oriented recipes for common tasks
- **Reference** (`docs/reference/`) - Information-oriented technical descriptions
- **Explanation** (`docs/explanation/`) - Understanding-oriented discussions of key topics

## Development

### Installation

```bash
npm install
```

### Local Development

```bash
npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to the main branch.

## Contributing

When adding new documentation:

1. Place content in the appropriate Diataxis category:
   - **Tutorials**: Step-by-step learning experiences
   - **How-to Guides**: Goal-oriented task instructions
   - **Reference**: Technical descriptions and API docs
   - **Explanation**: Conceptual discussions and architecture

2. Update `sidebars.ts` to include new pages in the navigation

3. Test locally with `npm start` before committing

4. Ensure all internal links work with `npm run build`

## More Information

- [Docusaurus Documentation](https://docusaurus.io/)
- [Diataxis Framework](https://diataxis.fr/)
