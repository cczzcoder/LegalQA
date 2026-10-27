import { defineStore } from 'pinia'
import { apiFetch, apiFetchRaw } from '../http'

function uid() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export const useChatStore = defineStore('chat', {
  state: () => ({
    sessions: [],
    currentId: null,
    messages: [],
    citations: [],
    activeCitation: -1,
    status: '',
    streaming: false,
    error: '',
    controller: null,
    ready: false,
  }),

  getters: {
    hasMessages: (state) => state.messages.length > 0,
  },

  actions: {
    async loadSessions() {
      this.sessions = await apiFetch('/api/sessions')
    },

    async createSession() {
      const session = await apiFetch('/api/sessions', {
        method: 'POST',
        body: JSON.stringify({}),
      })
      this.currentId = session.id
      this.messages = []
      this.citations = []
      this.activeCitation = -1
      await this.loadSessions()
      return session.id
    },

    async init() {
      if (this.ready) return
      try {
        await this.loadSessions()
        if (this.sessions.length) await this.openSession(this.sessions[0].id)
        else await this.createSession()
        this.ready = true
      } catch (error) {
        this.error = error.message || '加载会话失败'
      }
    },

    reset() {
      this.stop()
      this.sessions = []
      this.currentId = null
      this.messages = []
      this.citations = []
      this.activeCitation = -1
      this.status = ''
      this.error = ''
      this.ready = false
    },

    async newChat() {
      this.stop()
      this.error = ''
      try {
        await this.createSession()
      } catch (error) {
        this.error = error.message || '创建会话失败'
      }
    },

    async openSession(id) {
      this.stop()
      this.error = ''
      try {
        const data = await apiFetch(`/api/sessions/${id}`)
        this.currentId = data.id
        this.messages = (data.messages || []).map((message) => ({
          ...message,
          id: message.id || uid(),
          citations: message.citations || [],
          meta: message.meta || null,
        }))
        const last = [...this.messages].reverse().find((m) => m.role === 'assistant' && m.citations?.length)
        this.citations = last?.citations || []
        this.activeCitation = -1
        this.status = ''
      } catch (error) {
        this.error = error.message || '打开会话失败'
      }
    },

    showCitations(citations, index = -1) {
      this.citations = Array.isArray(citations) ? citations : []
      this.activeCitation = index
    },

    selectCitation(index) {
      this.activeCitation = index
    },

    stop() {
      if (this.controller) {
        this.controller.abort()
        this.controller = null
      }
      this.streaming = false
      this.status = ''
    },

    applyEvent(event, assistant) {
      if (event.type === 'status') {
        this.status = event.meta?.stage || 'thinking'
      } else if (event.type === 'delta') {
        assistant.content += event.text || ''
        this.status = 'answering'
      } else if (event.type === 'reset') {
        assistant.content = ''
        this.status = 'retrieving'
      } else if (event.type === 'replace') {
        assistant.content = event.text || ''
        this.status = 'answering'
      } else if (event.type === 'citations') {
        assistant.citations = event.citations || []
        this.citations = event.citations || []
        this.activeCitation = -1
      } else if (event.type === 'meta') {
        assistant.meta = event.meta || {}
      } else if (event.type === 'error') {
        this.error = event.error || '服务异常'
        this.status = 'error'
        if (!assistant.content) assistant.content = '服务暂时不可用，请稍后重试。'
      } else if (event.type === 'done') {
        this.status = ''
      }
    },

    async send(text) {
      const query = String(text || '').trim()
      if (!query || this.streaming) return

      this.error = ''
      if (!this.currentId) {
        try {
          await this.createSession()
        } catch (error) {
          this.error = error.message || '创建会话失败'
          return
        }
      }

      this.messages.push({ role: 'user', content: query, id: uid() })
      this.messages.push({
        role: 'assistant',
        content: '',
        id: uid(),
        citations: [],
        meta: null,
      })
      // push 之后取 Pinia 的响应式代理，后续增量更新才能触发重渲染
      const assistant = this.messages[this.messages.length - 1]
      this.citations = []
      this.activeCitation = -1
      this.status = 'thinking'
      this.streaming = true
      this.controller = new AbortController()

      try {
        const response = await apiFetchRaw(`/api/sessions/${this.currentId}/chat`, {
          method: 'POST',
          body: JSON.stringify({ query }),
          signal: this.controller.signal,
        })
        if (!response.ok) {
          const info = await response.json().catch(() => ({}))
          throw new Error(info.detail || `HTTP ${response.status}`)
        }

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          let boundary
          while ((boundary = buffer.indexOf('\n\n')) >= 0) {
            const block = buffer.slice(0, boundary)
            buffer = buffer.slice(boundary + 2)
            for (const line of block.split('\n')) {
              if (!line.startsWith('data: ')) continue
              try {
                this.applyEvent(JSON.parse(line.slice(6)), assistant)
              } catch {
                /* 忽略单个坏事件 */
              }
            }
          }
        }
      } catch (error) {
        if (error.name !== 'AbortError') {
          this.error = error.message || '请求失败'
          this.status = 'error'
          if (!assistant.content) assistant.content = '请求失败，请确认后端已启动后重试。'
        }
      } finally {
        this.streaming = false
        this.controller = null
        if (this.status !== 'error') this.status = ''
        try {
          await this.loadSessions()
        } catch {
          /* 列表刷新失败不影响当前对话 */
        }
      }
    },
  },
})
