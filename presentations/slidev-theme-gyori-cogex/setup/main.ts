/**
 * Slidev theme setup - Enforce dark mode
 *
 * This theme is designed exclusively for dark mode.
 * Light mode is not supported.
 */

import { defineAppSetup } from '@slidev/types'

export default defineAppSetup(({ app }) => {
  // Force dark mode on the document
  if (typeof document !== 'undefined') {
    document.documentElement.classList.add('dark')
    document.documentElement.classList.remove('light')

    // Prevent any toggles from switching to light mode
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.attributeName === 'class') {
          const html = document.documentElement
          if (!html.classList.contains('dark')) {
            html.classList.add('dark')
          }
          if (html.classList.contains('light')) {
            html.classList.remove('light')
          }
        }
      })
    })

    observer.observe(document.documentElement, { attributes: true })
  }
})
