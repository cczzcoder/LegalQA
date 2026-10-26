import { defineStore } from 'pinia'

const STORAGE_KEY = 'legalmind.sessions.v1'

function uid() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function loadSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    const data = raw ? JSON.parse(raw) : []
    return Array.isArray(data) ? data : []
  } catch {
    return []
  }
}

export const useChatStore = defineStore('chat', {
  state: () => ({
    sessions: loadSessions(),
    currentId: null,
    messages: [],
    citations: [],
    activeCitation: -1,
    status: '',
    streaming: false,
    error: '',
    controller: null,
  }),

  getters: {
    hasMessages: (state) => state.messages.length > 0,
  },

  actions: {
    persist() {
      if (!this.currentId) return
      const firstUser = this.messages.find((message) => message.role === 'user')
      const session = {
        id: this.currentId,
        title: (firstUser?.content || '新对话').replace(/\s+/g, ' ').slice(0, 26),
        updatedAt: Date.now(),
        messages: this.messages,
        citations: this.citations,
      }
      const index = this.sessions.findIndex((item) => item.id === this.currentId)
      if (index >= 0) this.sessions[index] = session
      else this.sessions.unshift(session)
      this.sessions = this.sessions.slice(0, 20)
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(this.sessions))
      } catch {
        /* 隐私模式/配额满时忽略 */
      }
    },

    newChat() {
      this.stop()
      this.currentId = uid()
      this.messages = []
      this.citations = []
      this.activeCitation = -1
      this.status = ''
      this.error = ''
    },

    openSession(id) {
      const session = this.sessions.find((item) => item.id === id)
      if (!session) return
      this.stop()
      this.currentId = id
      this.messages = JSON.parse(JSON.stringify(session.messages || []))
      this.citations = session.citations || []
      this.activeCitation = -1
      this.status = ''
      this.error = ''
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

      if (!this.currentId) this.currentId = uid()
      this.error = ''
      const history = this.messages.map((message) => ({
        role: message.role,
        content: message.content,
      }))

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

      const turnId = Math.max(this.messages.filter((m) => m.role === 'user').length - 1, 0)
      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, history, turn_id: turnId }),
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
        this.persist()
      }
    },
  },
})
