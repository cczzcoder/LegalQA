<template>
  <div class="login-page">
    <section class="login-hero">
      <div class="login-brand">
        <div class="brand-mark">法</div>
        <div class="brand-text">
          <strong>LegalMind</strong>
          <span>中文法律问答</span>
        </div>
      </div>
      <h1>把问题说清楚，<br />我们把法条找出来。</h1>
      <p class="hero-lead">
        检索 23 部现行法律、4264 条法条；回答带引用、版本与出处。
      </p>
      <ul class="hero-points">
        <li>引用的每一条都能回查原文</li>
        <li>标出版本与效力，避免旧法误导</li>
        <li>没找到依据时明确说不确定</li>
      </ul>
      <div class="hero-foot">本回答仅供参考，不构成正式法律意见。</div>
    </section>

    <section class="login-panel">
      <div class="login-card">
        <div class="login-tabs" role="tablist">
          <button :class="{ active: mode === 'login' }" @click="switchMode('login')">登录</button>
          <button :class="{ active: mode === 'register' }" @click="switchMode('register')">注册</button>
        </div>
        <h2>{{ mode === 'login' ? '欢迎回来' : '创建账号' }}</h2>
        <p class="login-sub">
          {{ mode === 'login' ? '登录后继续之前的咨询记录。' : '注册后会话会保存在本地数据库。' }}
        </p>

        <form @submit.prevent="submit">
          <label for="username">用户名</label>
          <el-input
            id="username"
            v-model="username"
            size="large"
            maxlength="32"
            placeholder="3-32 位中文、字母、数字或下划线"
            autocomplete="username"
          />

          <label for="password">密码</label>
          <el-input
            id="password"
            v-model="password"
            type="password"
            show-password
            size="large"
            maxlength="128"
            :placeholder="mode === 'register' ? '至少 8 位' : '请输入密码'"
            autocomplete="current-password"
          />

          <template v-if="mode === 'register'">
            <label for="confirm">确认密码</label>
            <el-input
              id="confirm"
              v-model="confirm"
              type="password"
              show-password
              size="large"
              maxlength="128"
              placeholder="再次输入密码"
              autocomplete="new-password"
            />
          </template>

          <div v-if="error" class="login-error">{{ error }}</div>

          <el-button
            type="primary"
            size="large"
            native-type="submit"
            :loading="auth.loading"
            class="login-submit"
          >
            {{ mode === 'login' ? '登录' : '注册并进入' }}
          </el-button>
        </form>

        <p class="login-note">账号只用于保存会话；密码经 PBKDF2 加盐存储。</p>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useAuthStore } from '../stores/auth'

const auth = useAuthStore()
const mode = ref('login')
const username = ref('')
const password = ref('')
const confirm = ref('')
const error = ref('')

function switchMode(next) {
  mode.value = next
  error.value = ''
  auth.error = ''
}

async function submit() {
  error.value = ''
  if (!username.value.trim() || !password.value) {
    error.value = '请输入用户名和密码'
    return
  }
  if (mode.value === 'register') {
    if (password.value.length < 8) {
      error.value = '密码至少 8 位'
      return
    }
    if (password.value !== confirm.value) {
      error.value = '两次输入的密码不一致'
      return
    }
  }
  const ok =
    mode.value === 'login'
      ? await auth.login(username.value, password.value)
      : await auth.register(username.value, password.value)
  if (!ok) error.value = auth.error || '请求失败'
}
</script>
