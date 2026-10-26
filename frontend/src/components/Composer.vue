<template>
  <div class="composer">
    <div class="composer-box">
      <textarea
        v-model="text"
        class="composer-input"
        maxlength="2000"
        rows="2"
        placeholder="描述你遇到的法律问题，例如：公司拖欠工资三个月了，我该怎么办？"
        @keydown.enter.exact.prevent="submit"
        @keydown.ctrl.enter.prevent="submit"
      ></textarea>
      <div class="composer-actions">
        <span class="composer-hint">
          Enter 发送 · Shift+Enter 换行
          <span class="char-count">{{ text.length }}/2000</span>
        </span>
        <el-button v-if="streaming" type="warning" plain @click="$emit('stop')">停止生成</el-button>
        <el-button v-else type="primary" :disabled="!text.trim()" @click="submit">发送</el-button>
      </div>
    </div>
    <p class="composer-note">回答仅供参考，不构成正式法律意见。涉及重大权益请咨询执业律师。</p>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({ streaming: { type: Boolean, default: false } })
const emit = defineEmits(['send', 'stop'])
const text = ref('')

function submit() {
  const value = text.value.trim()
  if (!value || props.streaming) return
  emit('send', value)
  text.value = ''
}
</script>
