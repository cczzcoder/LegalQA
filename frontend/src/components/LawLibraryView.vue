<template>
  <section class="library-view">
    <div class="library-bar">
      <div class="library-stats">
        <span><strong>{{ stats.total_laws }}</strong>部法律</span>
        <span><strong>{{ stats.total_articles }}</strong>条条文</span>
        <span><strong>{{ stats.valid_articles }}</strong>条现行有效</span>
      </div>
      <el-input
        v-model="query"
        class="library-search"
        clearable
        placeholder="搜关键词或条号：拖欠工资 / 第八十七条"
        @input="onQueryInput"
        @clear="reload"
      />
    </div>

    <div class="library-body">
      <aside class="law-index">
        <button
          class="law-index-item"
          :class="{ active: !activeLaw }"
          @click="selectLaw('')"
        >
          <span>全部法律</span>
          <small>{{ stats.total_articles }}</small>
        </button>
        <button
          v-for="law in laws"
          :key="law.name"
          class="law-index-item"
          :class="{ active: activeLaw === law.name }"
          :title="law.name"
          @click="selectLaw(law.name)"
        >
          <span>{{ law.name }}</span>
          <small>{{ law.article_count }}</small>
        </button>
      </aside>

      <div class="article-pane">
        <div v-if="error" class="pane-state error">{{ error }}</div>
        <div v-else-if="loading && !items.length" class="pane-state">正在读取法条库…</div>
        <template v-else>
          <div class="article-summary">
            <span>{{ activeLaw || '全部法律' }}</span>
            <em>{{ total }} 条{{ query.trim() ? '匹配' : '' }}</em>
          </div>

          <div v-if="!items.length" class="pane-state">
            没有匹配的法条，换个关键词试试。
          </div>
          <div v-else class="article-list" :class="{ dim: loading }">
            <article v-for="item in items" :key="item.id" class="law-article">
              <header class="law-article-head">
                <span v-if="!activeLaw" class="law-article-name">{{ item.law_name }}</span>
                <strong>{{ item.article_label }}</strong>
                <el-tag v-if="item.version_date" size="small" effect="plain">
                  {{ item.version_date }} 版本
                </el-tag>
                <el-tag v-if="!item.is_valid" size="small" type="info">已失效</el-tag>
              </header>
              <p>{{ item.text }}</p>
              <footer>
                <span>
                  {{ item.chapter }}
                  <template v-if="item.section"> · {{ item.section }}</template>
                </span>
                <span>{{ item.doc_type }}</span>
              </footer>
            </article>
          </div>

          <el-pagination
            v-if="total > pageSize"
            class="pager"
            layout="prev, pager, next"
            :total="total"
            :page-size="pageSize"
            :current-page="page"
            @current-change="load"
          />
        </template>
      </div>
    </div>
  </section>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'

const stats = ref({ total_laws: 0, total_articles: 0, valid_articles: 0 })
const laws = ref([])
const items = ref([])
const total = ref(0)
const page = ref(1)
const pageSize = 20
const query = ref('')
const activeLaw = ref('')
const loading = ref(true)
const error = ref('')

let timer = null
let requestSeq = 0

async function getJSON(url) {
  const response = await fetch(url, { credentials: 'include' })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`)
  return data
}

async function load(target = 1) {
  const seq = ++requestSeq
  loading.value = true
  error.value = ''
  const params = new URLSearchParams({
    offset: String((target - 1) * pageSize),
    limit: String(pageSize),
  })
  if (activeLaw.value) params.set('law', activeLaw.value)
  if (query.value.trim()) params.set('q', query.value.trim())
  try {
    const data = await getJSON(`/api/laws/articles?${params}`)
    if (seq !== requestSeq) return
    items.value = data.items
    total.value = data.total
    page.value = target
  } catch (err) {
    if (seq === requestSeq) error.value = err.message || '法条加载失败'
  } finally {
    if (seq === requestSeq) loading.value = false
  }
}

async function bootstrap() {
  try {
    const data = await getJSON('/api/laws')
    stats.value = data.stats
    laws.value = data.laws
  } catch (err) {
    error.value = err.message || '法条库加载失败'
    loading.value = false
    return
  }
  await load(1)
}

function selectLaw(name) {
  if (activeLaw.value === name) return
  activeLaw.value = name
  load(1)
}

function onQueryInput() {
  clearTimeout(timer)
  timer = setTimeout(() => load(1), 280)
}

function reload() {
  clearTimeout(timer)
  load(1)
}

onMounted(bootstrap)
onBeforeUnmount(() => clearTimeout(timer))
</script>
