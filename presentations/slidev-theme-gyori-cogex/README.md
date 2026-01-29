# slidev-theme-gyori-cogex

A Slidev theme for INDRA CoGEx MCP analysis presentations, designed with the Gyori Lab aesthetic for computational biology.

## Features

### Color Palette

The theme uses the Gyori Lab color system:

- **Primary (Cyan/Teal)** `#17a2b8` - Scientific credibility, data visualization
- **Secondary (Purple)** `#BF40BF` - Mechanistic pathways, causal relationships
- **Accent (Green)** `#5bb900` - Positive validation, success states

### Layouts

| Layout | Description |
|--------|-------------|
| `cover` | Dramatic title slide with animated network background |
| `section` | Section dividers with gradient backgrounds |
| `default` | Standard content slides |
| `two-cols` | Side-by-side content layout |
| `side-title` | Colored sidebar with main content |
| `end` | Closing slide with contact info |

### Components

| Component | Description |
|-----------|-------------|
| `<StatementCard>` | INDRA statement visualization |
| `<CliqueCard>` | Protein clique display |
| `<StatBox>` | Statistics highlight |
| `<Gene>` | Inline gene symbol (italic) |
| `<CausalPath>` | Causal pathway diagram |
| `<EvidenceList>` | Literature evidence list |
| `<ResultsHighlight>` | Key findings emphasis |

### Biology-Specific Styles

CSS classes for biological notation:

```html
<span class="gene">SOD1</span>           <!-- Italic gene symbol -->
<span class="protein">P00441</span>       <!-- Monospace protein ID -->
<span class="pathway">Autophagy</span>    <!-- Bold pathway name -->

<span class="stmt-activation">activates</span>
<span class="stmt-inhibition">inhibits</span>
```

## Installation

```bash
# In your Slidev project
npm install slidev-theme-gyori-cogex
```

Or use locally:

```yaml
---
theme: ./path/to/slidev-theme-gyori-cogex
---
```

## Usage

```markdown
---
theme: gyori-cogex
title: My INDRA Analysis
---

# Title Slide

Content here...

---
layout: section
color: cogex
---

# Section Title

---
layout: two-cols
---

::left::
Left content

::right::
Right content
```

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build theme
npm run build
```

## Credits

- **Gyori Lab** - Laboratory of Systems Pharmacology, Harvard Medical School
- **INDRA** - Integrated Network and Dynamical Reasoning Assembler
- **Slidev** - Presentation framework for developers

## License

MIT
