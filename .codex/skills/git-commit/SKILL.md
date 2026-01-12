---
name: auto-git-commit
description: 智能 Git 提交助手。自动读取 git diff 分析变更内容并生成中文 Commit Message；在提交前强制扫描大文件（>20MB），若发现则暂停并询问处理方式，防止仓库膨胀。
---

# Auto Git Commit Skill

## 目标
实现“零手动”代码提交。Agent 需自动阅读代码变更，撰写符合 Researcher 习惯的中文提交记录，并在检测到大文件风险时主动拦截，保障代码库的整洁与安全。

## 执行步骤

1. **大文件预扫描（安全熔断）**
   - **操作**: 使用 `find . -type f -not -path '*/.*' -size +20M` (或等效 Python 逻辑) 扫描当前目录下未被 `.gitignore` 排除的大于 20MB 的文件。
   - **判断逻辑**:
     - **若存在大文件**: 立即停止自动流程，输出文件列表（含路径与大小），并向用户发起询问：
       > "⚠️ 检测到以下大文件（可能为模型权重或数据集），请指示：
       > 1. [推荐] 忽略 (添加到 .gitignore)
       > 2. 使用 Git LFS 追踪
       > 3. 强制提交 (不推荐)
       > 4. 跳过本次提交"
       *等待用户输入指令后，根据指令执行相应操作（如写入 .gitignore），再继续后续步骤。*
     - **若无大文件**: 直接进入下一步。

2. **状态检测与暂存**
   - **操作**: 执行 `git status --porcelain`。
   - **分支**: 若无变更，输出 "✅ 工作区干净，无需提交" 并结束。
   - **操作**: 执行 `git add -A` 将所有变更（新增/修改/删除）加入暂存区。

3. **智能生成提交信息 (Auto-Summary)**
   - **操作**: 执行 `git diff --staged` 获取暂存区差异。
   - **分析**: 调用 LLM 阅读 Diff 内容（若 Diff token 过长，仅截取文件列表及关键代码段）。
   - **生成**: 基于差异内容生成一句精炼的中文 Commit Message。
     - **格式要求**: `<Type>: <Subject>` (遵循 Conventional Commits)
     - **Type 定义**:
       - `feat`: 新增实验/功能/模型
       - `fix`: 修复 Bug/报错
       - `docs`: 修改 Readme/注释
       - `chore`: 调整配置/环境/gitignore
       - `refactor`: 代码重构
     - **示例**: "feat: 增加 LoRA 微调脚本并优化数据加载器"

4. **执行提交与推送**
   - **操作**:
     1. `git commit -m "<生成的中文消息>"`
     2. `git pull --rebase` (防止本地落后导致 Push 失败)
     3. `git push origin <当前分支>`
   - **异常处理**: 若 `git push` 遇到 Conflict，停止并提示用户手动合并。

5. **反馈结果**
   - **输出**: "🚀 已成功提交并推送！\nCommit: `<generated_message>`\nHash: `[commit_hash]`"

## 目录结构
此 Skill 运行于当前 Shell 上下文，不生成持久化文件，主要修改 `.git/` 状态及 `.gitignore`。

## 约束
- **自动化优先**: 除非遇到大文件或 Merge 冲突，否则不需请求用户确认，直接生成并提交。
- **中文强制**: Commit Message 必须使用中文，专业术语（如 Transformer, PyTorch）保留英文。
- **安全第一**: 严禁在未确认情况下提交 `>20MB` 的二进制文件。
- **信息脱敏**: 生成 Message 时忽略 Diff 中的 API Key 或密码等敏感信息。
