# HiveFrame Product Showcase Website

This is the product showcase website for HiveFrame, hosted at `apiaryio.com`. It's separate from the documentation site and focuses on marketing the product's key features and benefits.

## Overview

The website is a static HTML/CSS/JavaScript site that showcases:

- **Hero Section**: Eye-catching introduction with animated bee swarm visualization
- **Architecture Comparison**: Visual comparison between traditional (Spark) and HiveFrame architectures
- **Nature-Inspired Features**: Detailed feature cards explaining bee-inspired concepts
- **Benefits Section**: Why choose HiveFrame over alternatives
- **Comparison Table**: Side-by-side feature comparison with Apache Spark
- **Getting Started**: Quick installation and usage guide
- **Use Cases**: Real-world applications and scenarios
- **Call-to-Actions**: Multiple CTAs throughout the page to encourage adoption

## File Structure

```
website/
├── index.html      # Main HTML file with all content
├── styles.css      # Complete styling with responsive design
├── script.js       # Interactive features and animations
└── README.md       # This file
```

## Features

### Design
- Modern, clean design with bee-inspired color palette (yellow/amber accents)
- Fully responsive layout that works on desktop, tablet, and mobile
- Smooth scrolling and subtle animations
- Professional typography using Inter font

### Interactivity
- Animated bee swarm visualization
- Smooth scroll navigation
- Copy-to-clipboard for code snippets
- Scroll-based fade-in animations
- Interactive hover effects on cards

### Performance
- Lightweight vanilla JavaScript (no frameworks)
- Optimized CSS with CSS variables
- Fast loading with minimal dependencies
- Only external dependency: Google Fonts (Inter)

## Local Development

To view the website locally, simply open `index.html` in a web browser:

```bash
# Using Python's built-in HTTP server
cd website
python -m http.server 8000

# Or using Node.js http-server
npx http-server

# Then open http://localhost:8000 in your browser
```

## Deployment Options

### Option 1: Static Hosting (Recommended)

Deploy to any static hosting service:

- **Netlify**: Drop the folder or connect to GitHub
- **Vercel**: `vercel deploy`
- **GitHub Pages**: Push to gh-pages branch
- **AWS S3 + CloudFront**: Upload to S3 bucket with static hosting
- **Azure Static Web Apps**: Deploy via GitHub Actions
- **Cloudflare Pages**: Connect to repository

### Option 2: Traditional Web Server

Upload files to any web server (Apache, Nginx, etc.):

```bash
# Copy files to web server
scp -r website/* user@apiaryio.com:/var/www/html/
```

### Option 3: Docker

Create a simple Dockerfile:

```dockerfile
FROM nginx:alpine
COPY website/ /usr/share/nginx/html/
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

Build and run:

```bash
docker build -t hiveframe-website .
docker run -p 80:80 hiveframe-website
```

## Domain Configuration

To point `apiaryio.com` to this website:

1. **DNS Configuration**: Update A/CNAME records to point to your hosting service
2. **SSL Certificate**: Use Let's Encrypt or your hosting provider's SSL
3. **CDN** (Optional): Configure CloudFlare or similar for better performance

### Example DNS Records

```
Type    Name    Value                   TTL
A       @       192.0.2.1              3600
CNAME   www     apiaryio.com           3600
```

## Customization

### Update Links

The website currently links to GitHub repositories and placeholder documentation. Update these in `index.html`:

- Documentation links (currently `href="#"`)
- GitHub organization (currently `UstDoesTech/hiveframe`)
- Examples repository path

### Color Scheme

To change colors, update CSS variables in `styles.css`:

```css
:root {
    --primary-color: #FFB400;      /* Main brand color */
    --primary-dark: #E5A000;       /* Darker variant */
    --secondary-color: #2C3E50;    /* Dark backgrounds */
    --accent-color: #3498DB;       /* Accent elements */
    /* ... other colors ... */
}
```

### Content Updates

All content is in `index.html`. Key sections:

- **Line 14-27**: Navigation
- **Line 30-110**: Hero section
- **Line 113-171**: Architecture comparison
- **Line 174-223**: Features grid
- **Line 226-276**: Benefits section
- **Line 279-327**: Comparison table
- **Line 330-400**: Getting started
- **Line 403-431**: Use cases
- **Line 434-468**: Final CTA
- **Line 471-523**: Footer

## Browser Compatibility

- Chrome/Edge: ✓ Full support
- Firefox: ✓ Full support
- Safari: ✓ Full support
- Mobile browsers: ✓ Full support
- IE11: ⚠️ Partial support (some CSS features may not work)

## Performance

The website is optimized for fast loading:

- Minimal external dependencies (only Google Fonts)
- ~40KB HTML (uncompressed)
- ~15KB CSS (uncompressed)
- ~3KB JavaScript (uncompressed)
- **Total size: ~60KB** (before compression)

With gzip compression (typically enabled by default on web servers):
- **Compressed size: ~15-20KB**

## Analytics (Optional)

To add analytics, insert tracking code before closing `</head>` tag in `index.html`:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

## SEO Optimization

The website includes:

- ✓ Semantic HTML structure
- ✓ Meta description and keywords
- ✓ Proper heading hierarchy (h1, h2, h3)
- ✓ Alt text for images (when added)
- ✓ Mobile-responsive design
- ✓ Fast loading times

To further improve SEO:

1. Add `robots.txt` file
2. Add `sitemap.xml` file
3. Implement Open Graph tags for social sharing
4. Add structured data (JSON-LD)

## License

This website follows the same MIT license as the HiveFrame project.

## Support

For issues or questions about the website, please open an issue in the main HiveFrame repository.
