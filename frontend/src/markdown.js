import { marked } from 'marked'
import DOMPurify from 'dompurify'

marked.setOptions({ gfm: true, breaks: true })

const CITE_RE = /《([^《》]{2,30})》\s*第\s*([一二三四五六七八九十百千零〇两0-9]{1,12})\s*条/g

function shortLaw(name) {
  return String(name || '').replace('中华人民共和国', '')
}

// 把回答正文里的「《法律》第X条」包成可点击定位的 span。
// 只包装文本节点、span 内容用 textContent 注入，DOM 已过 DOMPurify，不引入新风险。
export function linkCitations(html, citations = []) {
  if (!citations.length || !html) return html
  const container = document.createElement('div')
  container.innerHTML = html

  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT)
  const textNodes = []
  while (walker.nextNode()) textNodes.push(walker.currentNode)

  for (const node of textNodes) {
    const text = node.nodeValue || ''
    CITE_RE.lastIndex = 0
    let match
    let last = 0
    const parts = []
    while ((match = CITE_RE.exec(text))) {
      const law = match[1]
      const label = `第${match[2]}条`
      const index = citations.findIndex((citation) => {
        const sameLaw =
          citation.law_name === law ||
          shortLaw(citation.law_name) === shortLaw(law)
        return (
          sameLaw &&
          (citation.article_label === label ||
            shortLaw(citation.article_label) === shortLaw(label))
        )
      })
      if (match.index > last) {
        parts.push(document.createTextNode(text.slice(last, match.index)))
      }
      const span = document.createElement('span')
      span.className = 'cite-inline'
      span.dataset.cite = String(index)
      span.textContent = match[0]
      parts.push(span)
      last = match.index + match[0].length
    }
    if (!parts.length) continue
    if (last < text.length) parts.push(document.createTextNode(text.slice(last)))
    const fragment = document.createDocumentFragment()
    for (const part of parts) fragment.appendChild(part)
    node.parentNode.replaceChild(fragment, node)
  }
  return container.innerHTML
}

export function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(String(text || '')))
}
