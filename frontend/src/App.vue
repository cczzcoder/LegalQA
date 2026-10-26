<template>
  <div v-if="!auth.ready" class="boot-screen">
    <div class="boot-mark">法</div>
    <p>正在校验登录状态…</p>
  </div>
  <LoginView v-else-if="!auth.user" />
  <div v-else class="app-shell">
    <Sidebar
      class="sidebar"
      :class="{ open: sidebarOpen }"
      :username="auth.user?.username || ''"
      :sessions="store.sessions"
      :active-id="store.currentId"
      @new="onSidebarNew"
      @example="onSidebarExample"
      @open="onSidebarOpen"
      @logout="onLogout"
      @close="sidebarOpen = false"
    />

    <main class="main">
      <header class="topbar">
        <div class="topbar-title">
          <span class="eyebrow">LEGALMIND</span>
          <h1>把问题说清楚，我们把法条找出来</h1>
        </div>
        <div class="topbar-actions">
          <el-tag class="model-tag" effect="plain" type="info">qwen-plus · 法条 RAG</el-tag>
          <el-button class="mobile-only" text @click="openSidebar">菜单</el-button>
          <el-button class="mobile-only" text @click="toggleCitations">法条依据</el-button>
        </div>
      </header>

      <section ref="scrollEl" class="chat-scroll">
        <div v-if="!store.hasMessages" class="empty-state">
          <div class="empty-mark">§</div>
          <h2>用大白话描述你的处境</h2>
          <p>系统检索中国现行法条，给出带引用、带版本、可溯源的参考回答。</p>
          <div class="empty-examples">
            <button
              v-for="item in examples.slice(0, 4)"
              :key="item.title"
              @click="onExample(item.question)"
            >
              <span>{{ item.title }}</span>
              <small>{{ item.question }}</small>
            </button>
          </div>
        </div>

        <div v-else class="messages">
          <ChatMessage
            v-for="message in store.messages"
            :key="message.id"
            :message="message"
            @citation-click="onCitationClick"
          />
          <div v-if="store.streaming" class="thinking">
            <span class="thinking-dot"></span>
            {{ statusText }}
          </div>
        </div>
      </section>

      <Composer :streaming="store.streaming" @send="store.send" @stop="store.stop" />
      <div v-if="store.error" class="error-bar">{{ store.error }}</div>
    </main>

    <div
      v-if="isMobile && sidebarOpen"
      class="sidebar-backdrop"
      @click="sidebarOpen = false"
    ></div>
    <div
      v-if="isMobile && showCitations"
      class="panel-backdrop"
      @click="showCitations = false"
    ></div>
    <CitationPanel
      :citations="store.citations"
      :active-index="store.activeCitation"
      :open="showCitations"
      @select="store.selectCitation"
      @close="showCitations = false"
    />
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import Sidebar from './components/Sidebar.vue'
import ChatMessage from './components/ChatMessage.vue'
import CitationPanel from './components/CitationPanel.vue'
import Composer from './components/Composer.vue'
import LoginView from './components/LoginView.vue'
import examples from './legalExamples'
import { useAuthStore } from './stores/auth'
import { useChatStore } from './stores/chat'

const auth = useAuthStore()
const store = useChatStore()
const scrollEl = ref(null)
const isMobile = ref(window.innerWidth < 1024)
const showCitations = ref(!isMobile.value)
const sidebarOpen = ref(false)

const statusText = computed(() => {
  if (store.status === 'retrieving') return '正在检索法条…'
  if (store.status === 'answering') return '正在整理回答…'
  if (store.status === 'error') return '请求异常'
  return '正在分析问题…'
})

function onExample(question) {
  store.showCitations([], -1)
  store.send(question)
}

function openSidebar() {
  sidebarOpen.value = true
  showCitations.value = false
}

function toggleCitations() {
  showCitations.value = !showCitations.value
  if (showCitations.value) sidebarOpen.value = false
}

function onCitationClick(citations, index) {
  store.showCitations(citations, index)
  sidebarOpen.value = false
  showCitations.value = true
}

function onSidebarNew() {
  sidebarOpen.value = false
  store.newChat()
}

function onSidebarExample(question) {
  sidebarOpen.value = false
  onExample(question)
}

function onSidebarOpen(id) {
  sidebarOpen.value = false
  store.openSession(id)
}

async function onLogout() {
  sidebarOpen.value = false
  await auth.logout()
  store.reset()
}

function scrollBottom() {
  nextTick(() => {
    if (scrollEl.value) scrollEl.value.scrollTop = scrollEl.value.scrollHeight
  })
}

watch(
  () => [store.messages.length, store.messages.at(-1)?.content, store.streaming],
  scrollBottom,
)

watch(
  () => auth.user,
  async (user) => {
    if (user) await store.init()
    else store.reset()
  },
)

onMounted(async () => {
  await auth.me()
  window.addEventListener('resize', () => {
    isMobile.value = window.innerWidth < 1024
    if (!isMobile.value) showCitations.value = true
  })
})
</script>
