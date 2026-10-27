<template>
  <section class="eval-view">
    <div class="eval-head">
      <div>
        <h2>评测看板</h2>
        <p>读取 data/eval 下已保存的结果，不重新调用模型。</p>
      </div>
      <el-select
        v-model="runId"
        class="run-select"
        filterable
        placeholder="选择评测批次"
        :disabled="loadingRuns || !runs.length"
        @change="loadRun"
      >
        <el-option-group v-if="keyRuns.length" label="关键节点">
          <el-option
            v-for="run in keyRuns"
            :key="run.id"
            :label="run.label"
            :value="run.id"
          >
            <div class="run-option">
              <span>{{ run.label }}</span>
              <em>{{ run.case_count }} 条 · {{ fmtPct(run.pass_rate) }}</em>
            </div>
          </el-option>
        </el-option-group>
        <el-option-group v-if="otherRuns.length" label="历史实验">
          <el-option
            v-for="run in otherRuns"
            :key="run.id"
            :label="run.id"
            :value="run.id"
          >
            <div class="run-option">
              <span>{{ run.id }}</span>
              <em>{{ run.case_count }} 条 · {{ fmtPct(run.pass_rate) }}</em>
            </div>
          </el-option>
        </el-option-group>
      </el-select>
    </div>

    <div v-if="error" class="pane-state error">{{ error }}</div>
    <div v-else-if="loadingRuns || loadingDetail" class="pane-state">正在读取评测结果…</div>
    <div v-else-if="!runs.length" class="pane-state">data/eval 下还没有可读的评测结果。</div>

    <template v-else-if="detail">
      <div class="eval-run-meta">
        <strong>{{ current?.label || detail.id }}</strong>
        <span v-if="current?.note">{{ current.note }}</span>
        <span>{{ detail.case_count }} 条用例</span>
        <span v-if="current?.updated_at">更新于 {{ fmtTime(current.updated_at) }}</span>
      </div>

      <div class="eval-metrics">
        <div class="metric-card accent">
          <span>通过率</span>
          <strong>{{ fmtPct(detail.pass_rate) }}</strong>
          <small>{{ detail.passed }} / {{ detail.case_count }}</small>
        </div>
        <div class="metric-card">
          <span>平均耗时</span>
          <strong>{{ fmtSeconds(detail.avg_latency_ms) }}</strong>
          <small>端到端，含模型与工具</small>
        </div>
        <div class="metric-card">
          <span>P90 耗时</span>
          <strong>{{ fmtSeconds(detail.p90_latency_ms) }}</strong>
          <small>长尾用例</small>
        </div>
        <div class="metric-card">
          <span>失败用例</span>
          <strong>{{ detail.failures.length }}</strong>
          <small>规则检查未全通过</small>
        </div>
      </div>

      <div class="eval-grid">
        <section class="check-panel">
          <h3>检查项</h3>
          <div v-for="item in detail.checks" :key="item.name" class="check-row">
            <div class="check-row-head">
              <span>{{ item.name }}</span>
              <em>{{ item.failed ? `${item.failed}/${item.total} 失败` : '全部通过' }}</em>
            </div>
            <div class="check-track">
              <div
                class="check-fill ok"
                :style="{ width: okWidth(item) }"
              ></div>
              <div
                v-if="item.failed"
                class="check-fill bad"
                :style="{ width: badWidth(item) }"
              ></div>
            </div>
          </div>
          <p class="check-note">「检索覆盖」为诊断项，不参与通过判定。</p>
        </section>

        <section class="failure-panel">
          <div class="failure-panel-head">
            <h3>失败明细</h3>
            <span>{{ detail.failures.length }} 条</span>
          </div>
          <div v-if="!detail.failures.length" class="pane-state">
            本批用例全部通过。
          </div>
          <div v-else class="failure-list">
            <article
              v-for="item in detail.failures"
              :key="item.id"
              class="failure-card"
              :class="{ open: Boolean(expanded[item.id]) }"
            >
              <button class="failure-head" @click="toggleCase(item.id)">
                <span class="case-id">{{ item.id }}</span>
                <span class="failure-query">{{ item.query }}</span>
                <span class="failure-tags">
                  <span
                    v-for="check in item.failed_checks"
                    :key="check.name"
                    class="flag flag-danger"
                  >{{ check.name }}</span>
                </span>
                <small>{{ fmtSeconds(item.latency_ms) }}</small>
              </button>

              <div v-if="expanded[item.id]" class="failure-detail">
                <p v-if="expanded[item.id].loading" class="detail-hint">正在读取用例…</p>
                <p v-else-if="expanded[item.id].error" class="detail-hint error">
                  {{ expanded[item.id].error }}
                </p>
                <template v-else-if="expanded[item.id].data">
                  <div class="failure-checks">
                    <div v-for="check in item.failed_checks" :key="check.name">
                      <strong>{{ check.name }}</strong>
                      <span>{{ check.detail || '未通过' }}</span>
                    </div>
                  </div>
                  <div class="answer-block markdown" v-html="answerHtml(item.id)"></div>
                  <div v-if="expanded[item.id].data.retrieved?.length" class="retrieved-list">
                    <span
                      v-for="(doc, index) in expanded[item.id].data.retrieved"
                      :key="`${doc.law}-${doc.no}-${index}`"
                      :title="doc.law"
                    >{{ shortLaw(doc.law) }} · 第{{ doc.no }}条</span>
                  </div>
                  <p v-if="expanded[item.id].data.backfilled?.length" class="backfill-note">
                    回捞核验：
                    <span
                      v-for="(doc, index) in expanded[item.id].data.backfilled"
                      :key="`${doc.law}-${doc.no}-${index}`"
                    >{{ shortLaw(doc.law) }}第{{ doc.no }}条{{ index < expanded[item.id].data.backfilled.length - 1 ? '、' : '' }}</span>
                  </p>
                </template>
              </div>
            </article>
          </div>
        </section>
      </div>
    </template>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { renderMarkdown } from '../markdown'

const runs = ref([])
const runId = ref('')
const detail = ref(null)
const loadingRuns = ref(true)
const loadingDetail = ref(false)
const error = ref('')
const expanded = ref({})

const keyRuns = computed(() => runs.value.filter((run) => run.pinned))
const otherRuns = computed(() => runs.value.filter((run) => !run.pinned))
const current = computed(() => runs.value.find((run) => run.id === runId.value) || null)

async function getJSON(url) {
  const response = await fetch(url, { credentials: 'include' })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`)
  return data
}

async function bootstrap() {
  try {
    const data = await getJSON('/api/eval/runs')
    runs.value = data.runs
    if (runs.value.length) {
      runId.value = runs.value[0].id
      await loadRun()
    }
  } catch (err) {
    error.value = err.message || '评测结果加载失败'
  } finally {
    loadingRuns.value = false
  }
}

async function loadRun() {
  if (!runId.value) return
  loadingDetail.value = true
  error.value = ''
  detail.value = null
  expanded.value = {}
  try {
    detail.value = await getJSON(`/api/eval/runs/${encodeURIComponent(runId.value)}`)
  } catch (err) {
    error.value = err.message || '评测结果加载失败'
  } finally {
    loadingDetail.value = false
  }
}

async function toggleCase(caseId) {
  if (expanded.value[caseId]) {
    const next = { ...expanded.value }
    delete next[caseId]
    expanded.value = next
    return
  }
  expanded.value = { ...expanded.value, [caseId]: { loading: true } }
  try {
    const data = await getJSON(
      `/api/eval/runs/${encodeURIComponent(runId.value)}/cases/${encodeURIComponent(caseId)}`,
    )
    expanded.value = { ...expanded.value, [caseId]: { loading: false, data } }
  } catch (err) {
    expanded.value = {
      ...expanded.value,
      [caseId]: { loading: false, error: err.message || '用例加载失败' },
    }
  }
}

function answerHtml(caseId) {
  const data = expanded.value[caseId]?.data
  return data ? renderMarkdown(data.answer) : ''
}

function okWidth(item) {
  if (!item.failed) return '100%'
  return `${100 - badWidthPct(item)}%`
}

function badWidth(item) {
  return `${badWidthPct(item)}%`
}

function badWidthPct(item) {
  return Math.max(item.fail_rate * 100, 2)
}

function fmtPct(value) {
  return `${((Number(value) || 0) * 100).toFixed(1)}%`
}

function fmtSeconds(ms) {
  const value = Number(ms) || 0
  return value ? `${(value / 1000).toFixed(1)}s` : '—'
}

function fmtTime(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  const pad = (number) => String(number).padStart(2, '0')
  return `${date.getMonth() + 1}/${date.getDate()} ${pad(date.getHours())}:${pad(date.getMinutes())}`
}

function shortLaw(name) {
  return String(name || '').replace('中华人民共和国', '')
}

onMounted(bootstrap)
</script>
