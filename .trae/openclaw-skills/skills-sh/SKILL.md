---
summary: "使用 Vercel Labs 的 skills.sh CLI 工具来搜索、安装和管理 AI 技能"
read_when:
  - 需要搜索、安装或管理 AI 技能时
  - 需要使用 skills.sh CLI 工具时
---

# skills.sh Skill - Vercel Labs AI 技能管理工具

## 什么是 skills.sh？

skills.sh 是 Vercel Labs 推出的开源 AI 技能管理平台，提供标准化的技能库，可以通过一行命令安装和管理各种 AI 技能。

## 核心功能

- **技能搜索**: 使用 `skills find` 搜索技能
- **技能安装**: 使用 `skills add` 安装技能
- **技能列表**: 使用 `skills list` 查看已安装的技能
- **技能更新**: 使用 `skills update` 更新技能到最新版本

## 使用方法

### 搜索技能

```bash
npx skills find [query]
```

### 安装技能

```bash
npx skills add <package>
```

示例：
```bash
npx skills add vercel-labs/agent-skills
```

### 查看已安装的技能

```bash
npx skills list
```

### 更新技能

```bash
npx skills update
```

## 常用技能包

- `vercel-labs/agent-skills` - Vercel Labs 官方技能包
- `openclaw-expert` - OpenClaw 学习专家

## 注意事项

- 使用 `npx skills --help` 查看完整帮助信息
- 技能会安装到当前项目的 `node_modules` 目录
- 可以使用 `--global` 选项安装全局技能
