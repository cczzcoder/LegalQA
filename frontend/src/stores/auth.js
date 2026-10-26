import { defineStore } from 'pinia'

function errorText(data, fallback) {
  const detail = data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg
  return fallback
}

export const useAuthStore = defineStore('auth', {
  state: () => ({
    user: null,
    ready: false,
    loading: false,
    error: '',
  }),

  actions: {
    async me() {
      this.loading = true
      try {
        const response = await fetch('/api/auth/me', { credentials: 'include' })
        this.user = response.ok ? await response.json() : null
      } catch {
        this.user = null
      } finally {
        this.loading = false
        this.ready = true
      }
    },

    async login(username, password) {
      return this.submit('/api/auth/login', { username, password })
    },

    async register(username, password) {
      return this.submit('/api/auth/register', { username, password })
    },

    async submit(url, payload) {
      this.loading = true
      this.error = ''
      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify(payload),
        })
        const data = await response.json().catch(() => ({}))
        if (!response.ok) throw new Error(errorText(data, '请求失败'))
        this.user = data
        return true
      } catch (error) {
        this.error = error.message || '请求失败'
        return false
      } finally {
        this.loading = false
      }
    },

    async logout() {
      try {
        await fetch('/api/auth/logout', { method: 'POST', credentials: 'include' })
      } catch {
        /* 用户仍可本地退出 */
      }
      this.user = null
      this.error = ''
    },
  },
})
