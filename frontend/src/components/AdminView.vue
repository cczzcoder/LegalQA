<template>
  <section class="admin-view">
    <div class="admin-tabs">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        :class="{ active: active === tab.id }"
        @click="switchTab(tab.id)"
      >
        {{ tab.label }}
        <em v-if="tab.id === 'feedback' && overview.feedback_open">
          {{ overview.feedback_open }}
        </em>
      </button>
    </div>

    <div class="admin-body">
      <!-- 概览 -->
      <div v-if="active === 'overview'" class="admin-overview">
        <div class="admin-cards">
          <div class="admin-card">
            <span>未读反馈</span>
            <strong>{{ overview.feedback_open }}</strong>
          </div>
          <div class="admin-card">
            <span>待处理导入</span>
            <strong>{{ overview.imports_pending }}</strong>
          </div>
          <div class="admin-card" :class="{ warn: overview.index.lagged }">
            <span>索引状态</span>
            <strong>{{ overview.index.lagged ? '落后' : '最新' }}</strong>
            <small>{{ overview.index.corpus_count }} 条语料 / {{ overview.index.index_count }} 条索引</small>
          </div>
          <div class="admin-card">
            <span>语料规模</span>
            <strong>{{ overview.corpus.total_laws }}</strong>
            <small>部法律 · {{ overview.corpus.total_articles }} 条条文</small>
          </div>
        </div>

        <h3>最近操作</h3>
        <div v-if="!overview.recent_audit?.length" class="admin-empty">暂无审计记录。</div>
        <div v-else class="audit-list">
          <div v-for="item in overview.recent_audit" :key="item.id" class="audit-item">
            <span class="audit-action">{{ actionLabel(item.action) }}</span>
            <span class="audit-detail">{{ item.username || '-' }} {{ item.detail }}</span>
            <em class="audit-time">{{ fmtTime(item.created_at) }}</em>
          </div>
        </div>

        <h3>版本提醒</h3>
        <div v-if="!overview.version_warnings.length" class="admin-empty">
          没有需要关注的版本问题。
        </div>
        <div v-else class="warn-list">
          <div
            v-for="item in overview.version_warnings"
            :key="item.name"
            class="warn-item"
          >
            <span>{{ item.name }}</span>
            <em>
              {{ item.reason }}
              <template v-if="item.version">（{{ item.version }}）</template>
            </em>
          </div>
        </div>
        <p v-if="overview.index.reason" class="admin-note">
          索引说明：{{ overview.index.reason }}
        </p>
      </div>

      <!-- 反馈 -->
      <div v-else-if="active === 'feedback'" class="admin-feedback">
        <div class="admin-toolbar">
          <el-select
            v-model="feedbackStatus"
            class="fb-status"
            @change="loadFeedback"
          >
            <el-option label="待处理" value="open" />
            <el-option label="已处理" value="done" />
            <el-option label="已忽略" value="ignored" />
            <el-option label="全部" value="" />
          </el-select>
          <span class="admin-count">共 {{ feedback.total }} 条</span>
        </div>

        <div v-if="!feedback.items.length" class="admin-empty">暂无反馈。</div>
        <div v-else class="fb-list">
          <article v-for="item in feedback.items" :key="item.id" class="fb-card">
            <header class="fb-head">
              <span class="fb-cat">{{ item.category }}</span>
              <span class="fb-meta">{{ fmtTime(item.created_at) }}</span>
              <span class="fb-status" :class="item.status">{{ statusLabel(item.status) }}</span>
            </header>
            <p class="fb-q"><strong>问：</strong>{{ item.question }}</p>
            <p class="fb-a"><strong>答：</strong>{{ item.answer }}</p>
            <p v-if="item.comment" class="fb-comment"><strong>反馈：</strong>{{ item.comment }}</p>
            <footer class="fb-citations">
              <span
                v-for="(citation, index) in item.citations"
                :key="index"
                class="fb-chip"
              >{{ shortLaw(citation.law_name) }} · {{ citation.article_label || `第${citation.article_no}条` }}</span>
            </footer>
            <div class="fb-actions">
              <el-button
                v-if="item.status === 'open'"
                size="small"
                type="primary"
                @click="patchFeedback(item.id, 'done')"
              >标记已处理</el-button>
              <el-button
                v-if="item.status === 'open'"
                size="small"
                @click="patchFeedback(item.id, 'ignored')"
              >忽略</el-button>
              <el-button
                v-if="item.status !== 'open'"
                size="small"
                @click="patchFeedback(item.id, 'open')"
              >重新打开</el-button>
            </div>
          </article>
        </div>
      </div>

      <!-- 导入 -->
      <div v-else-if="active === 'imports'" class="admin-imports">
        <div class="admin-upload">
          <el-upload
            drag
            :auto-upload="false"
            :limit="1"
            accept=".docx,.docm,.pdf"
            :on-change="onPick"
            :on-remove="onRemove"
          >
            <div class="upload-inner">把 .docx / .pdf 拖到这里，或点击选择</div>
            <template #tip>
              <div class="el-upload__tip">
                上限 20MB；分析会比对重复与版本时效，确认后才入库。
              </div>
            </template>
          </el-upload>
          <el-button
            type="primary"
            :disabled="!pickedFile || uploading"
            :loading="uploading"
            @click="uploadFile"
          >上传</el-button>
        </div>

        <div v-if="!imports.length" class="admin-empty">还没有导入记录。</div>
        <div v-else class="imp-list">
          <article v-for="item in imports" :key="item.id" class="imp-card">
            <header class="imp-head">
              <span class="imp-name">{{ item.filename }}</span>
              <span class="imp-status" :class="item.status">{{ statusLabel(item.status) }}</span>
              <span class="imp-meta">{{ fmtSize(item.size_bytes) }} · {{ fmtTime(item.created_at) }}</span>
            </header>

            <div v-if="item.report && (item.report.ok || item.report.suggestion || item.error)" class="imp-report">
              <div
                class="imp-suggestion"
                :class="{ warn: (item.report.suggestion || '').includes('警告') }"
              >{{ item.report.suggestion || item.error }}</div>
              <p v-if="item.report.law_name" class="imp-detail">
                识别为《{{ item.report.law_name }}》
                <template v-if="item.report.version_date">（{{ item.report.version_date }} 版本）</template>
                <template v-if="item.report.article_count">，共 {{ item.report.article_count }} 条条文</template>
              </p>
              <p v-if="item.report.missing_numbers?.length" class="imp-detail warn">
                缺号：{{ item.report.missing_numbers.join('、') }}
              </p>
              <p v-if="item.report.duplicated_numbers?.length" class="imp-detail warn">
                重号：{{ item.report.duplicated_numbers.join('、') }}
              </p>
              <p v-if="item.report.existing?.found" class="imp-detail">
                库内同名 {{ item.report.existing.article_count }} 条，条文重合
                {{ (item.report.existing.overlap_rate * 100).toFixed(1) }}%
              </p>
              <p v-if="item.report.timeliness?.version_missing" class="imp-detail warn">
                文件名缺少版本日期
              </p>
              <p v-if="item.report.timeliness?.years_since_version != null" class="imp-detail">
                版本距今 {{ item.report.timeliness.years_since_version }} 年
              </p>
              <p v-if="item.report.apply_result" class="imp-detail">
                {{ item.report.apply_result.index_status }}
              </p>
            </div>

            <footer class="imp-actions">
              <el-button
                v-if="['uploaded', 'failed'].includes(item.status)"
                size="small"
                :loading="busy === item.id + 'analyze'"
                @click="analyzeImport(item.id)"
              >分析</el-button>
              <el-button
                v-if="item.status === 'analyzed'"
                size="small"
                type="primary"
                :loading="busy === item.id + 'apply'"
                @click="applyImport(item)"
              >确认入库（替换旧版）</el-button>
              <span v-if="busy === item.id + 'apply'" class="admin-note">
                入库会增量更新索引，可能需要几十秒
              </span>
            </footer>
          </article>
        </div>
      </div>

      <!-- 语料 -->
      <div v-else class="admin-corpus">
        <div class="admin-toolbar">
          <el-input
            v-model="lawFilter"
            class="corpus-filter"
            clearable
            placeholder="过滤法律名称"
          />
          <span class="admin-count">共 {{ laws.length }} 部法律</span>
        </div>
        <div v-if="!filteredLaws.length" class="admin-empty">没有匹配的法律。</div>
        <div v-else class="corpus-list">
          <article v-for="law in filteredLaws" :key="law.name" class="corpus-row">
            <span class="corpus-name">{{ law.name }}</span>
            <span class="corpus-version" :class="{ warn: lawWarned(law) }">
              {{ law.latest_version || '无版本日期' }}
            </span>
            <span class="corpus-count">{{ law.article_count }} 条 · 有效 {{ law.valid_count }}</span>
          </article>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { apiFetch } from '../http'

const tabs = [
  { id: 'overview', label: '概览' },
  { id: 'feedback', label: '反馈' },
  { id: 'imports', label: '导入' },
  { id: 'corpus', label: '语料' },
]

const active = ref('overview')
const overview = ref({
  feedback_open: 0,
  imports_pending: 0,
  index: { lagged: false, corpus_count: 0, index_count: 0, reason: '' },
  corpus: { total_laws: 0, total_articles: 0 },
  version_warnings: [],
})
const feedback = ref({ total: 0, items: [] })
const feedbackStatus = ref('open')
const imports = ref([])
const laws = ref([])
const lawFilter = ref('')
const pickedFile = ref(null)
const uploading = ref(false)
const busy = ref('')
const error = ref('')

const filteredLaws = computed(() => {
  const keyword = lawFilter.value.trim()
  if (!keyword) return laws.value
  return laws.value.filter((law) => law.name.includes(keyword))
})

async function loadOverview() {
  try {
    overview.value = await apiFetch('/api/admin/overview')
  } catch (err) {
    error.value = err.message || '概览加载失败'
  }
}

async function loadFeedback() {
  const params = new URLSearchParams({ limit: '100' })
  if (feedbackStatus.value) params.set('status', feedbackStatus.value)
  try {
    feedback.value = await apiFetch(`/api/admin/feedback?${params}`)
  } catch (err) {
    error.value = err.message || '反馈加载失败'
  }
}

async function loadImports() {
  try {
    imports.value = (await apiFetch('/api/admin/imports')).items
  } catch (err) {
    error.value = err.message || '导入记录加载失败'
  }
}

async function loadLaws() {
  try {
    laws.value = (await apiFetch('/api/laws')).laws
  } catch (err) {
    error.value = err.message || '语料加载失败'
  }
}

async function patchFeedback(id, status) {
  try {
    await apiFetch(`/api/admin/feedback/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ status }),
    })
    await Promise.all([loadFeedback(), loadOverview()])
  } catch (err) {
    error.value = err.message || '更新反馈失败'
  }
}

function onPick(file) {
  pickedFile.value = file.raw || null
}

function onRemove() {
  pickedFile.value = null
}

async function uploadFile() {
  uploading.value = true
  error.value = ''
  const form = new FormData()
  form.append('file', pickedFile.value)
  try {
    await apiFetch('/api/admin/imports', {
      method: 'POST',
      body: form,
    })
    pickedFile.value = null
    await Promise.all([loadImports(), loadOverview()])
  } catch (err) {
    error.value = err.message || '上传失败'
  } finally {
    uploading.value = false
  }
}

async function analyzeImport(id) {
  busy.value = `${id}analyze`
  error.value = ''
  try {
    await apiFetch(`/api/admin/imports/${id}/analyze`, { method: 'POST' })
    await loadImports()
  } catch (err) {
    error.value = err.message || '分析失败'
  } finally {
    busy.value = ''
  }
}

async function applyImport(item) {
  busy.value = `${item.id}apply`
  error.value = ''
  try {
    await apiFetch(`/api/admin/imports/${item.id}/apply`, {
      method: 'POST',
      body: JSON.stringify({ replace: true }),
    })
    await Promise.all([loadImports(), loadLaws(), loadOverview()])
  } catch (err) {
    error.value = err.message || '入库失败'
  } finally {
    busy.value = ''
  }
}

function switchTab(id) {
  active.value = id
  error.value = ''
  if (id === 'feedback') loadFeedback()
  if (id === 'imports') loadImports()
  if (id === 'corpus') loadLaws()
  if (id === 'overview') loadOverview()
}

function statusLabel(status) {
  return (
    { uploaded: '待分析', analyzed: '待确认', failed: '解析失败', applied: '已入库',
      open: '待处理', done: '已处理', ignored: '已忽略' }[status] || status
  )
}

function actionLabel(action) {
  return (
    { login: '登录', login_failed: '登录失败', register: '注册', logout: '退出',
      admin_promote: '提升管理员', feedback_status: '反馈处理', import_upload: '上传文献',
      import_analyze: '文献分析', import_apply: '文献入库', import_delete: '删除导入' }[action] || action
  )
}

function lawWarned(law) {
  if (!law.latest_version) return true
  const year = parseInt(law.latest_version.slice(0, 4), 10)
  return Number.isNaN(year) || new Date().getFullYear() - year >= 5
}

function fmtSize(bytes) {
  const value = Number(bytes) || 0
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / 1024 / 1024).toFixed(1)} MB`
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

onMounted(() => {
  loadOverview()
  loadImports()
  loadLaws()
})
</script>
