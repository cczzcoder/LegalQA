<template>
  <div class="message" :class="message.role">
    <div class="avatar" aria-hidden="true">{{ message.role === 'user' ? '我' : '法' }}</div>
    <div class="bubble">
      <div v-if="message.role === 'user'" class="user-text">{{ message.content }}</div>
      <div v-else class="markdown" v-html="html" @click="onMarkdownClick"></div>

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
        <button class="text-button" @click="feedbackOpen = true">反馈</button>
        <span class="message-meta" v-if="message.meta?.latency_ms">
          {{ (message.meta.latency_ms / 1000).toFixed(1) }}s
        </span>
      </div>

      <el-dialog
        v-model="feedbackOpen"
        title="反馈这个问题"
        width="min(92vw, 400px)"
        :close-on-click-modal="false"
      >
        <div v-if="feedbackState === 'done'" class="feedback-done">
          已收到你的反馈，谢谢！
        </div>
        <template v-else>
          <p class="feedback-hint">反馈会连同当前回答一起提交，用于改进回答质量。</p>
          <el-radio-group v-model="feedbackCategory" class="feedback-cats">
            <el-radio value="引用错误">引用错误</el-radio>
            <el-radio value="答案错误">答案错误</el-radio>
            <el-radio value="条号不存在">条号不存在</el-radio>
            <el-radio value="其他">其他</el-radio>
          </el-radio-group>
          <el-input
            v-model="feedbackComment"
            class="feedback-comment"
            type="textarea"
            :rows="3"
            maxlength="1000"
            placeholder="补充说明（选填）"
          />
          <p v-if="feedbackError" class="feedback-error">{{ feedbackError }}</p>
          <div class="feedback-actions">
            <el-button size="small" @click="feedbackOpen = false">取消</el-button>
            <el-button
              size="small"
              type="primary"
              :loading="feedbackState === 'sending'"
              @click="submitFeedback"
            >提交</el-button>
          </div>
        </template>
      </el-dialog>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { linkCitations, renderMarkdown } from '../markdown'
import { useChatStore } from '../stores/chat'
import { apiFetch } from '../http'

const props = defineProps({ message: { type: Object, required: true } })
const emit = defineEmits(['citation-click'])

const store = useChatStore()
const html = computed(() =>
  linkCitations(renderMarkdown(props.message.content), props.message.citations || []),
)
const copied = ref(false)

function onMarkdownClick(event) {
  const link = event.target.closest('.cite-inline')
  if (!link) return
  const index = Number(link.dataset.cite)
  if (Number.isInteger(index) && index >= 0) {
    emit('citation-click', props.message.citations || [], index)
  }
}

const feedbackOpen = ref(false)
const feedbackCategory = ref('引用错误')
const feedbackComment = ref('')
const feedbackState = ref('idle')
const feedbackError = ref('')

async function submitFeedback() {
  feedbackState.value = 'sending'
  feedbackError.value = ''
  try {
    await apiFetch('/api/feedback', {
      method: 'POST',
      body: JSON.stringify({
        session_id: store.currentId,
        message_id: props.message.id,
        category: feedbackCategory.value,
        comment: feedbackComment.value,
      }),
    })
    feedbackState.value = 'done'
  } catch (error) {
    feedbackState.value = 'idle'
    feedbackError.value = error.message || '提交失败，请稍后重试'
  }
}

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
