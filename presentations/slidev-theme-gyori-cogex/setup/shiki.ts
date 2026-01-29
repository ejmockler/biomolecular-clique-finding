import { defineShikiSetup } from '@slidev/types'

/**
 * Shiki syntax highlighting configuration
 *
 * Optimized for computational biology code examples:
 * - Python for data analysis
 * - R for statistics
 * - Bash for pipelines
 */
export default defineShikiSetup(() => {
  return {
    themes: {
      dark: 'github-dark',
      light: 'github-light',
    },
    langs: [
      'python',
      'r',
      'bash',
      'shell',
      'json',
      'yaml',
      'markdown',
      'javascript',
      'typescript',
    ],
  }
})
