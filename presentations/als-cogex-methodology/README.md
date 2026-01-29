# AI-Augmented Biomedical Discovery Presentation

A 40-minute presentation on using INDRA CoGEx MCP and LLM agents to investigate sporadic ALS.

## Overview

This presentation serves two purposes:
1. **Methodology introduction**: Teaching lab members how to use LLM agents with knowledge graphs
2. **ALS findings**: Presenting the results of our sALS proteomics investigation

## Target Audience

Lab members who:
- Have not used LLM agents before
- Don't know what MCP (Model Context Protocol) is
- Want to understand how to apply this methodology to their own research

## Structure (40 minutes)

| Part | Duration | Slides | Content |
|------|----------|--------|---------|
| 1. The Problem | 5 min | 1-5 | Why biomedical knowledge integration is broken |
| 2. The Approach | 10 min | 6-14 | LLMs, agents, MCP, and INDRA CoGEx |
| 3. ALS Case Study | 15 min | 15-26 | Walking through the actual investigation |
| 4. Pitfalls | 7 min | 27-33 | What can go wrong and how to handle it |
| 5. Results | 3 min | 34-37 | Actionable findings and future directions |

## Running the Presentation

```bash
# Install dependencies
npm install

# Start development server (opens browser)
npm run dev

# Build for production
npm run build

# Export to PDF
npm run export-pdf
```

## Key Messages

### For the Methodology

1. LLM agents are NOT chatbots - they can use tools to access real databases
2. MCP standardizes how LLMs connect to knowledge sources
3. Answers are grounded in real databases, not LLM memory
4. This amplifies expertise, it doesn't replace it

### For the Biology

1. sALS shows massive sex × disease interaction
2. Mitochondrial apoptosis emerges as the convergent death pathway
3. Drug targets exist (ROCK1) but critical gaps remain (TDP-43, SOD1)
4. β-catenin-estrogen connection may explain sex dimorphism

## Pitfalls Covered

1. **Hallucination** - MCP grounds answers in databases
2. **Entity ambiguity** - GILDA grounding with semantic context
3. **Confirmation bias** - Query broadly, ask for counter-examples
4. **Knowledge gaps** - Cross-reference, note sources explicitly

## Speaker Notes

Every slide includes detailed speaker notes with:
- Talking points
- Timing guidance
- Key phrases to emphasize
- Transitions to next topics

Access speaker notes in Slidev by pressing `S` to enter presenter mode.

## Customization

### To update with your own data:

1. Replace sample statistics in StatBox components
2. Update the terminal output examples with actual queries
3. Add your own institution/contact info on cover and end slides
4. Adjust the findings table based on your specific results

### Theme customization:

The presentation uses the `slidev-theme-gyori-cogex` theme located in:
```
../slidev-theme-gyori-cogex/
```

Available colors: `gyori` (teal), `cogex` (purple), `bio` (green)

## Assets Needed

For the best presentation:
- [ ] Screenshots of actual terminal sessions
- [ ] Diagram of MCP architecture (can use mermaid in slides)
- [ ] Example of grounded vs. hallucinated response
- [ ] Photos/diagrams of ALS pathology (optional)

## License

Internal use - Gyori Lab, Laboratory of Systems Pharmacology
