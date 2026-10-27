import { defineStore } from 'pinia'
import { apiFetch, apiFetchRaw } from '../http'

export const useAuthStore = defineStore('auth', {
  state: () => ({
    user: null,
    ready: false,
    loading: false,
    error: '',
    consoleCode: '',
  }),

  actions: {
    async me() {
      this.loading = true
      try {
        // GET 无需 CSRF 头；后端会给未登录页面补发 CSRF cookie。
        const response = await apiFetchRaw('/api/auth/me')
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

    async register(username, password, email = '') {
      return this.submit('/api/auth/register', { username, password, email })
    },

    async submit(url, payload) {
      this.loading = true
      this.error = ''
      try {
        const data = await apiFetch(url, {
          method: 'POST',
          body: JSON.stringify(payload),
        })
        this.user = data
        this.consoleCode = data.console_code || ''
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
        await apiFetch('/api/auth/logout', { method: 'POST' })
      } catch {
        /* 用户仍可本地退出 */
      }
      this.user = null
      this.error = ''
    },
  },
})
