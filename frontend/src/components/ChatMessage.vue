<template>
  <div class="message" :class="message.role">
    <div class="avatar" aria-hidden="true">{{ message.role === 'user' ? '我' : '法' }}</div>
    <div class="bubble">
      <div v-if="message.role === 'user'" class="user-text">{{ message.content }}</div>
      <div v-else class="markdown" v-html="html"></div>

      <div v-if="message.role === 'assistant' && message.citations?.length" class="chips">
        <button
          v-for="(citation, index) in message.citations.slice(0, 6)"
          :key="`${citation.law_name}-${citation.article_no}-${index}`"
          class="chip"
          @click="$emit('citation-click', message.citations, index)"
        >
          {{ shortLaw(citation.law_name) }} · {{ citation.article_label }}
        </button>
      </div>

      <div v-if="message.meta?.degraded || message.meta?.output_blocked" class="message-flags">
        <span v-if="message.meta?.degraded" class="flag flag-amber">系统降级</span>
        <span v-if="message.meta?.output_blocked" class="flag flag-gold">引用已核验</span>
      </div>

      <div v-if="message.role === 'assistant' && message.content" class="message-actions">
        <button class="text-button" @click="copy">{{ copied ? '已复制' : '复制回答' }}</button>
        <span class="message-meta" v-if="message.meta?.latency_ms">
          {{ (message.meta.latency_ms / 1000).toFixed(1) }}s
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { renderMarkdown } from '../markdown'

const props = defineProps({ message: { type: Object, required: true } })
defineEmits(['citation-click'])

const html = computed(() => renderMarkdown(props.message.content))
const copied = ref(false)

function shortLaw(name) {
  return String(name || '').replace('中华人民共和国', '')
}

async function copy() {
  try {
    await navigator.clipboard.writeText(props.message.content || '')
    copied.value = true
    setTimeout(() => (copied.value = false), 1200)
  } catch {
    /* 浏览器不允许时忽略 */
  }
}
</script>
