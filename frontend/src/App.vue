<template>
  <div v-if="!auth.ready" class="boot-screen">
    <div class="boot-mark">法</div>
    <p>正在校验登录状态…</p>
  </div>
  <LoginView v-else-if="!auth.user" />
  <div v-else class="app-shell" :class="{ 'app-shell--wide': view !== 'chat' }">
    <Sidebar
      v-if="view === 'chat'"
      class="sidebar"
      :class="{ open: sidebarOpen }"
      :username="auth.user?.username || ''"
      :sessions="store.sessions"
      :active-id="store.currentId"
      @new="onSidebarNew"
      @example="onSidebarExample"
      @open="onSidebarOpen"
      @delete="onSidebarDelete"
      @logout="onLogout"
      @close="sidebarOpen = false"
    />

    <main class="main">
      <header ref="topbarEl" class="topbar">
        <div class="topbar-title">
          <span class="eyebrow">LEGALQA</span>
          <h1>{{ viewTitle }}</h1>
        </div>
        <div class="topbar-actions">
          <nav class="view-tabs" aria-label="视图切换">
            <button
              v-for="tab in tabs"
              :key="tab.id"
              :class="{ active: view === tab.id }"
              @click="switchView(tab.id)"
            >
              {{ tab.label }}
            </button>
          </nav>
          <el-tag v-if="view === 'chat'" class="model-tag" effect="plain" type="info">
            qwen-plus · 法条 RAG
          </el-tag>
          <el-button
            v-if="view === 'chat'"
            class="mobile-only"
            text
            @click="openSidebar"
          >菜单</el-button>
          <el-button
            v-if="view === 'chat'"
            class="mobile-only"
            text
            @click="toggleCitations"
          >法条依据</el-button>
          <div v-if="view !== 'chat'" class="topbar-user">
            <span class="topbar-username">{{ auth.user?.username }}</span>
            <button class="topbar-logout" @click="onLogout">退出</button>
          </div>
        </div>
      </header>

      <template v-if="view === 'chat'">
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
      </template>

      <LawLibraryView v-else-if="view === 'laws'" />
      <EvalDashboardView v-else-if="view === 'eval'" />
      <AdminView v-else />
    </main>

    <template v-if="view === 'chat'">
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
    </template>

    <el-dialog
      v-model="verifyOpen"
      title="验证邮箱"
      width="min(92vw, 400px)"
      :close-on-click-modal="false"
      :close-on-press-escape="false"
      :show-close="false"
    >
      <p class="feedback-hint">
        验证码已发送至 {{ auth.user?.email }}（未配置邮件服务时请查看服务端控制台输出）。
      </p>
      <el-input
        v-model="verifyCode"
        maxlength="6"
        placeholder="6 位数字验证码"
        @keyup.enter="submitVerify"
      />
      <p v-if="verifyConsoleCode" class="login-hint">开发模式验证码：{{ verifyConsoleCode }}</p>
      <p v-if="verifyError" class="feedback-error">{{ verifyError }}</p>
      <div class="feedback-actions">
        <el-button size="small" :loading="verifySending" @click="resendVerify">重新发送</el-button>
        <el-button size="small" type="primary" :loading="verifySending" @click="submitVerify">提交</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import Sidebar from './components/Sidebar.vue'
import ChatMessage from './components/ChatMessage.vue'
import CitationPanel from './components/CitationPanel.vue'
import Composer from './components/Composer.vue'
import LoginView from './components/LoginView.vue'
import LawLibraryView from './components/LawLibraryView.vue'
import EvalDashboardView from './components/EvalDashboardView.vue'
import AdminView from './components/AdminView.vue'
import examples from './legalExamples'
import { apiFetch } from './http'
import { useAuthStore } from './stores/auth'
import { useChatStore } from './stores/chat'

const auth = useAuthStore()
const store = useChatStore()
const scrollEl = ref(null)
const topbarEl = ref(null)
const isMobile = ref(window.innerWidth < 1024)
const showCitations = ref(!isMobile.value)
const sidebarOpen = ref(false)
const view = ref('chat')

const verifyOpen = ref(false)
const verifyCode = ref('')
const verifyError = ref('')
const verifySending = ref(false)
const verifyConsoleCode = ref('')

async function submitVerify() {
  verifySending.value = true
  verifyError.value = ''
  try {
    await apiFetch('/api/auth/verify-email', {
      method: 'POST',
      body: JSON.stringify({ code: verifyCode.value.trim() }),
    })
    auth.user = { ...auth.user, email_verified: 1 }
    verifyOpen.value = false
    verifyCode.value = ''
  } catch (error) {
    verifyError.value = error.message || '验证失败'
  } finally {
    verifySending.value = false
  }
}

async function resendVerify() {
  verifySending.value = true
  verifyError.value = ''
  verifyConsoleCode.value = ''
  try {
    const data = await apiFetch('/api/auth/verify-email/resend', { method: 'POST' })
    verifyConsoleCode.value = data.console_code || ''
  } catch (error) {
    verifyError.value = error.message || '重新发送失败'
  } finally {
    verifySending.value = false
  }
}

const baseTabs = [
  { id: 'chat', label: '问答' },
  { id: 'laws', label: '法条库' },
  { id: 'eval', label: '评测' },
]

const tabs = computed(() =>
  auth.user?.role === 'admin'
    ? [...baseTabs, { id: 'admin', label: '管理' }]
    : baseTabs,
)

const titles = {
  chat: '把问题说清楚，我们把法条找出来',
  laws: '法条库：先看原文，再谈适用',
  eval: '评测看板：每一次改动都有据可查',
  admin: '管理端：反馈与法条库维护',
}

const viewTitle = computed(() => titles[view.value] || titles.chat)

const statusText = computed(() => {
  if (store.status === 'retrieving') return '正在检索法条…'
  if (store.status === 'answering') return '正在整理回答…'
  if (store.status === 'error') return '请求异常'
  return '正在分析问题…'
})

function switchView(next) {
  if (view.value === next) return
  view.value = next
  sidebarOpen.value = false
  showCitations.value = next === 'chat' && !isMobile.value
  syncTopbarHeight()
}

// 顶栏在窄屏会换行，高度不固定（64~72px）。法条面板/遮罩必须贴住它的
// 真实底部，否则会压住顶栏第二行。
function syncTopbarHeight() {
  nextTick(() => {
    const height = topbarEl.value?.getBoundingClientRect().height
    if (height) {
      document.documentElement.style.setProperty('--topbar-h', `${height}px`)
    }
  })
}

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

function onSidebarDelete(id) {
  sidebarOpen.value = false
  store.deleteSession(id)
}

async function onLogout() {
  sidebarOpen.value = false
  view.value = 'chat'
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
    if (user && user.role !== 'admin' && view.value === 'admin') view.value = 'chat'
    if (user) await store.init()
    else store.reset()
    // 有邮箱但未验证时弹出验证框（可跳过，不影响使用）
    if (user?.email && !user.email_verified) {
      verifyConsoleCode.value = auth.consoleCode
      verifyOpen.value = true
    }
  },
)

onMounted(async () => {
  await auth.me()
  syncTopbarHeight()
  window.addEventListener('resize', () => {
    isMobile.value = window.innerWidth < 1024
    if (isMobile.value) {
      showCitations.value = false
      sidebarOpen.value = false
    } else if (view.value === 'chat') {
      showCitations.value = true
    }
    syncTopbarHeight()
  })
})
</script>
