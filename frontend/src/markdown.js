import { marked } from 'marked'
import DOMPurify from 'dompurify'

marked.setOptions({ gfm: true, breaks: true })

export function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(String(text || '')))
}
