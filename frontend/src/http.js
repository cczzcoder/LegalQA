// 统一请求入口：同源 Cookie + 不安全方法自动带 X-CSRF-Token（双提交）。
// 其余组件不要直接 fetch，统一走这里，避免漏带 CSRF 头。

export function csrfToken() {
  const match = document.cookie.match(/(?:^|;\s*)lm_csrf=([^;]+)/)
  return match ? decodeURIComponent(match[1]) : ''
}

export function apiFetchRaw(url, options = {}) {
  const method = (options.method || 'GET').toUpperCase()
  const headers = { ...(options.headers || {}) }
  if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) {
    headers['X-CSRF-Token'] = csrfToken()
  }
  if (
    options.body &&
    !(options.body instanceof FormData) &&
    headers['Content-Type'] === undefined
  ) {
    headers['Content-Type'] = 'application/json'
  }
  return fetch(url, { ...options, credentials: 'include', headers })
}

export async function apiFetch(url, options = {}) {
  const response = await apiFetchRaw(url, options)
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = data?.detail
    const message = Array.isArray(detail) && detail[0]?.msg
      ? detail[0].msg
      : typeof detail === 'string'
        ? detail
        : `HTTP ${response.status}`
    throw new Error(message)
  }
  return data
}
