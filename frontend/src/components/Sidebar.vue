<template>
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark">法</div>
      <div class="brand-text">
        <strong>LegalQA</strong>
        <span>中文法律问答</span>
      </div>
      <button class="sidebar-close" aria-label="关闭菜单" @click="$emit('close')">×</button>
    </div>

    <el-button class="new-chat" type="primary" @click="$emit('new')">
      <el-icon><Plus /></el-icon>
      新对话
    </el-button>

    <section class="side-section">
      <h4>示例问题</h4>
      <button
        v-for="item in examples"
        :key="item.title"
        class="example-item"
        @click="$emit('example', item.question)"
      >
        <span class="example-title">{{ item.title }}</span>
        <span class="example-question">{{ item.question }}</span>
      </button>
    </section>

    <section v-if="sessions.length" class="side-section sessions">
      <h4>最近对话</h4>
      <button
        v-for="session in sessions"
        :key="session.id"
        class="session-item"
        :class="{ active: session.id === activeId }"
        @click="$emit('open', session.id)"
      >
        <span class="session-title">{{ session.title }}</span>
        <small>{{ formatDate(session.updated_at) }}</small>
      </button>
    </section>

    <div v-if="username" class="side-user">
      <span class="user-name">{{ username }}</span>
      <button class="logout-button" @click="$emit('logout')">退出</button>
    </div>

    <div class="side-foot">
      回答仅供参考，不构成正式法律意见。<br />
      法条版本以官方公布为准。
    </div>
  </aside>
</template>

<script setup>
import { Plus } from '@element-plus/icons-vue'
import examples from '../legalExamples'

defineProps({
  username: { type: String, default: '' },
  sessions: { type: Array, default: () => [] },
  activeId: { type: String, default: '' },
})
defineEmits(['new', 'example', 'open', 'close', 'logout'])

function formatDate(timestamp) {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  return `${date.getMonth() + 1}/${date.getDate()}`
}
</script>
