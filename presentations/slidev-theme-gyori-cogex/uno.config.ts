import { defineConfig } from 'unocss'

/**
 * Gyori Lab / INDRA CoGEx Color Palette
 *
 * Primary: Cyan/Teal - scientific credibility, data visualization
 * Secondary: Purple/Magenta - mechanistic pathways, causal relationships
 * Accent: Green - positive validation, success states
 *
 * Design Philosophy:
 * - Clean, academic aesthetic with computational biology focus
 * - High contrast for readability in conference settings
 * - Color-coded semantic meanings for biology concepts
 */
export default defineConfig({
  theme: {
    colors: {
      // Primary palette - Gyori Lab cyan/teal
      gyori: {
        50: '#e6f7fa',
        100: '#cceff5',
        200: '#99dfeb',
        300: '#66cfe1',
        400: '#33bfd7',
        500: '#17a2b8',  // Primary brand color
        600: '#128293',
        700: '#0e626e',
        800: '#094149',
        900: '#052125',
      },

      // Secondary - Purple for pathways/mechanisms
      cogex: {
        50: '#f9e6f9',
        100: '#f2ccf2',
        200: '#e699e6',
        300: '#d966d9',
        400: '#cc33cc',
        500: '#BF40BF',  // INDRA purple
        600: '#993399',
        700: '#732673',
        800: '#4d1a4d',
        900: '#260d26',
      },

      // Accent - Green for validation/success
      bio: {
        50: '#eef8e6',
        100: '#dcf1cc',
        200: '#b9e399',
        300: '#96d566',
        400: '#73c733',
        500: '#5bb900',  // Gyori green
        600: '#499400',
        700: '#376f00',
        800: '#244a00',
        900: '#122500',
      },

      // Semantic colors for biology
      activation: '#5bb900',      // Green - activation/increase
      inhibition: '#dc3545',      // Red - inhibition/decrease
      complex: '#17a2b8',         // Cyan - complex formation
      modification: '#BF40BF',    // Purple - PTM/modification
      transport: '#ffc107',       // Yellow - translocation

      // Clique analysis colors (matching viz module)
      clique: {
        up: '#0d9488',
        down: '#f97316',
        bidirectional: '#7c3aed',
        male: '#0ea5e9',
        female: '#ec4899',
        both: '#8b5cf6',
      },
    },
  },

  shortcuts: {
    // Background shortcuts
    'bg-main': 'bg-white text-slate-800 dark:(bg-slate-900 text-slate-100)',
    'bg-gyori': 'bg-gyori-500 text-white',
    'bg-cogex': 'bg-cogex-500 text-white',
    'bg-bio': 'bg-bio-500 text-white',
    'bg-code': 'bg-slate-100 dark:bg-slate-800 rounded-lg',

    // Card/panel styles
    'panel': 'bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 border border-slate-200 dark:border-slate-700',
    'panel-gyori': 'bg-gyori-50 dark:bg-gyori-900 rounded-xl p-6 border-2 border-gyori-500',
    'panel-cogex': 'bg-cogex-50 dark:bg-cogex-900 rounded-xl p-6 border-2 border-cogex-500',

    // Text utilities
    'text-gradient-gyori': 'bg-gradient-to-r from-gyori-500 to-cogex-500 text-transparent bg-clip-text',
    'text-mono': 'font-mono text-sm',
    'text-gene': 'font-italic text-gyori-600 dark:text-gyori-400',
    'text-protein': 'font-mono text-cogex-600 dark:text-cogex-400',

    // Biology-specific badges
    'badge': 'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
    'badge-activation': 'badge bg-bio-100 text-bio-800 dark:(bg-bio-900 text-bio-200)',
    'badge-inhibition': 'badge bg-red-100 text-red-800 dark:(bg-red-900 text-red-200)',
    'badge-complex': 'badge bg-gyori-100 text-gyori-800 dark:(bg-gyori-900 text-gyori-200)',
    'badge-modification': 'badge bg-cogex-100 text-cogex-800 dark:(bg-cogex-900 text-cogex-200)',

    // Slide layout utilities
    'slide-content': 'px-14 py-10',
    'slide-title': 'text-4xl font-bold text-slate-900 dark:text-white mb-4',
    'slide-subtitle': 'text-xl text-slate-600 dark:text-slate-400',

    // Grid layouts for data visualization
    'grid-stats': 'grid grid-cols-2 md:grid-cols-4 gap-4',
    'stat-card': 'panel text-center',
    'stat-value': 'text-3xl font-bold text-gyori-500',
    'stat-label': 'text-sm text-slate-500 mt-1',

    // Network visualization frame
    'network-frame': 'bg-slate-50 dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700',

    // Clique visualization styles
    'clique-up': 'text-clique-up',
    'clique-down': 'text-clique-down',
    'clique-bi': 'text-clique-bidirectional',
  },

  // Safelist commonly used dynamic classes
  safelist: [
    ...['gyori', 'cogex', 'bio'].flatMap(c => [
      `bg-${c}-50`, `bg-${c}-100`, `bg-${c}-500`, `bg-${c}-900`,
      `text-${c}-500`, `text-${c}-600`, `border-${c}-500`,
    ]),
    ...['activation', 'inhibition', 'complex', 'modification', 'transport'].map(c => [
      `bg-${c}`, `text-${c}`, `border-${c}`,
    ]).flat(),
  ],
})
