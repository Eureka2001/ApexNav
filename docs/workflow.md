# 工作流

## 添加新的开发分支

```bash
# 添加新的 git worktree

git worktree add ../dev

# 切换到新的 worktree
cd ../dev

# 构建工作分支中的相关目录依赖
ln -s /data0/yyw/projects_data/apex-nav/data/ ./data
ln -s /data0/yyw/projects_data/apex-nav/yolov7/ yolov7
ln -s /data0/yyw/projects_data/apex-nav/GroundingDINO/ GroundingDINO
ln -s .vscode ../main/.vscode
```
