<template>
  <aside class="citation-panel" :class="{ open: open }">
    <div class="panel-head">
      <div>
        <h3>法条依据</h3>
        <p>本次检索到的条文与版本</p>
      </div>
      <div class="panel-head-actions">
        <el-tag size="small" effect="plain">{{ citations.length }} 条</el-tag>
        <button class="panel-close" aria-label="关闭法条依据" @click="$emit('close')">×</button>
      </div>
    </div>

    <div v-if="!citations.length" class="panel-empty">
      <div class="empty-icon">§</div>
      <p>提问后，这里会显示检索到的法条原文、版本与效力。</p>
    </div>

    <div v-else class="citation-list" ref="listEl">
      <article
        v-for="(citation, index) in citations"
        :key="`${citation.law_name}-${citation.article_no}-${index}`"
        class="citation-card"
        :class="{ active: index === activeIndex, flash: index === flashIndex }"
        @click="$emit('select', index)"
      >
        <div class="cite-top">
          <span class="cite-law">{{ shortLaw(citation.law_name) }}</span>
          <span class="cite-no">{{ citation.article_label }}</span>
          <el-tag :type="citation.valid ? 'success' : 'danger'" size="small" effect="light">
            {{ citation.valid ? '现行有效' : '已废止' }}
          </el-tag>
          <el-tag v-if="citation.source === 'backfill'" size="small" type="warning" effect="plain">
            回捞核实
          </el-tag>
        </div>
        <div class="cite-meta">
          {{ citation.version || citation.effective_date || '版本未标注' }}
          <span v-if="citation.doc_type"> · {{ citation.doc_type }}</span>
        </div>
        <p class="cite-text">{{ citation.text }}</p>
      </article>
    </div>

    <div class="panel-foot">
      法条由系统检索/回查，引用请以官方公布文本为准。
    </div>
  </aside>
</template>

<script setup>
import { nextTick, ref, watch } from 'vue'

const props = defineProps({
  citations: { type: Array, default: () => [] },
  activeIndex: { type: Number, default: -1 },
  open: { type: Boolean, default: true },
})
defineEmits(['select', 'close'])

const listEl = ref(null)
const flashIndex = ref(-1)

watch(
  () => props.activeIndex,
  async (index) => {
    if (index < 0) return
    await nextTick()
    const card = listEl.value?.querySelectorAll('.citation-card')[index]
    card?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    flashIndex.value = index
    setTimeout(() => {
      if (flashIndex.value === index) flashIndex.value = -1
    }, 1600)
  },
)

function shortLaw(name) {
  return String(name || '').replace('中华人民共和国', '')
}
</script>
