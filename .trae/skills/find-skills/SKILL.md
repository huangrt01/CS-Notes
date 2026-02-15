# Find Skills Skill

## 简介

该 Skill 主要作用帮助你发现并安装 Agent Skill。它依托开放的 Agent Skill 生态，让你可以搜索、发现和使用各类模块化技能包；这些技能可扩展 Agent 能力，为其补充特定领域知识、标准化工作流与工具能力。

## 作者

vercel

## 原始地址

https://github.com/vercel-labs/skills/tree/main/skills/find-skills

## 热门技能分类

### 代码质量与审查
- **code-reviewer** - 通用代码审查（你已安装）
- **frontend-code-review** - 前端专用代码审查
- **backend-code-review** - 后端代码审查
- **security-review** - 安全代码审查

### 开发工作流
- **fix** - 代码格式和 linting 修复（你已安装）
- **pr-creator** - PR 创建助手（你已安装）
- **update-docs** - 技术文档更新
- **git-helper** - Git 工作流助手

### 前端开发
- **frontend-design** - 前端设计（生产级别界面）
- **cache-components** - Next.js 缓存组件优化
- **react-best-practices** - React 最佳实践
- **vue-best-practices** - Vue 最佳实践
- **typescript-helper** - TypeScript 助手

### 后端与全栈
- **fullstack-developer** - 全栈开发专家
- **api-designer** - API 设计助手
- **database-schema** - 数据库架构设计

### 测试与调试
- **webapp-testing** - 网页应用测试
- **unit-test-helper** - 单元测试助手
- **debugging-assistant** - 调试助手

### 文档与知识管理
- **update-docs** - 技术文档更新
- **knowledge-base** - 知识库管理
- **note-taking** - 笔记整理助手

## 应用场景

### 探索未知的 Skill
当你希望 Agent 帮忙处理某个特定领域的任务，但不确定 Agent 是否具备相应能力时，可以使用此 Skill 进行探索。例如，当你询问 "你能帮我评审代码吗？" 或 "如何为我的项目生成文档？" 时，该 Skill 会被激活，主动在技能生态中搜索与 "代码评审" 或 "文档生成" 相关的能力，并将找到的可用技能呈现给你。

### 查找特定的 Skill
当你明确知道需要一个 Skill 来解决特定问题，但不知道具体是哪个 Skill 时，可以主动调用此 Skill 进行精确查找。例如，你可以直接说 "帮我找一个用于 React 性能优化的 skill"，该 Skill 会将 "React 性能优化" 作为关键词进行搜索，并返回最匹配的技能选项。

### 提供可执行的 Skill 安装建议
当该 Skill 找到一个或多个匹配的 skill 后，它会自动整理并输出一份标准化的推荐信息。这份信息不仅包含技能的名称和功能简介，还会提供手动安装指南以及指向技能详情页的官方链接。

## 使用示例

### 搜索技能
```
帮我找一个用于代码审查的 skill
```

### 探索相关技能
```
有什么好的前端开发 skill 推荐吗？
```

### 查找特定功能
```
找一个能帮我格式化代码的 skill
```

### 按分类探索
```
推荐一些测试相关的 skills
```

### 热门推荐
```
有哪些必备的开发 skills？
```

## 如何手动安装 Skill

### 方法一：从 GitHub 仓库复制

1. 访问技能的 GitHub 仓库，例如：
   - Anthropic Skills: https://github.com/anthropics/skills
   - Vercel Skills: https://github.com/vercel-labs/skills
   - Google Gemini Skills: https://github.com/google-gemini/gemini-cli

2. 找到感兴趣的 skill 目录

3. 在你的项目中创建对应的目录：
   ```bash
   mkdir -p .trae/skills/{skill-name}
   ```

4. 复制该 skill 的 SKILL.md 文件到你的项目中：
   ```
   .trae/skills/{skill-name}/SKILL.md
   ```

5. 如果 skill 包含其他资源文件（如 references/、examples/ 等），也一并复制

### 方法二：手动创建

1. 在 `.trae/skills/` 目录下创建技能文件夹
2. 在该文件夹中创建 `SKILL.md` 文件
3. 按照标准格式编写技能文档：
   - 名称和简介
   - 应用场景
   - 使用示例
   - 工作流程

### 方法三：使用 Trae IDE 的 Skill 工具

在对话中直接告诉我你需要什么功能，我会帮你推荐和设置合适的技能！

## 最佳实践

### 1. 从基础技能开始
- 先安装代码审查、格式修复等基础技能
- 逐步添加特定领域的技能

### 2. 根据项目类型选择
- 前端项目：优先考虑 frontend-design、cache-components
- 全栈项目：考虑 fullstack-developer
- 库/框架项目：考虑 update-docs、code-reviewer

### 3. 定期更新技能
- 关注技能生态的新发展
- 定期检查是否有更新的技能版本

### 4. 组合使用多个技能
- 代码审查 + 格式修复 = 高质量代码
- PR 创建 + 文档更新 = 完整的提交流程

## 工作流程

1. 理解用户的需求或任务目标
2. 在开放的 Agent Skill 生态中进行关键词搜索
3. 筛选匹配度最高的技能
4. 展示技能的详细信息、功能简介和官方链接
5. 提供手动安装指南
6. 帮助用户理解技能的使用场景和能力边界
7. 根据用户反馈调整推荐