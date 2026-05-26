# Git Useful Commands

This guide lists common Git commands for this repository workflow.

Current common branches:

```text
dev
main
```

Use `dev` for active work. Merge or push changes to `main` only when the work is ready.

## Check Current Status

Show changed files:

```bash
git status
```

Short version:

```bash
git status --short
```

Show current branch and tracking status:

```bash
git status -sb
```

## List Branches

List local branches:

```bash
git branch
```

List remote branches:

```bash
git branch -r
```

List all local and remote branches:

```bash
git branch -a
```

Show branches with latest commit and tracking info:

```bash
git branch -vv
```

## Create a New Branch

Create a branch but stay on current branch:

```bash
git branch new-branch-name
```

Create and switch to the new branch:

```bash
git switch -c new-branch-name
```

Example:

```bash
git switch -c feature/update-dashboard
```

## Switch Branches

Switch to `dev`:

```bash
git switch dev
```

Switch to `main`:

```bash
git switch main
```

Older Git versions may use:

```bash
git checkout dev
```

## Pull Latest Changes

Pull latest changes for the current branch:

```bash
git pull
```

Pull latest `dev`:

```bash
git switch dev
git pull origin dev
```

Pull latest `main`:

```bash
git switch main
git pull origin main
```

## Stage Files for Commit

Stage one file:

```bash
git add README.md
```

Stage one folder:

```bash
git add Documentation/
```

Stage all changed files:

```bash
git add .
```

Check what is staged:

```bash
git status
```

## Commit Changes

Commit staged files:

```bash
git commit -m "Add useful Git commands documentation"
```

Use short present-tense messages:

```text
Add deployment guide
Fix login validation
Update README
Document Render setup
```

## Push Changes

Push current branch:

```bash
git push
```

Push `dev` branch:

```bash
git push origin dev
```

Push `main` branch:

```bash
git push origin main
```

Push a new branch and set upstream:

```bash
git push -u origin new-branch-name
```

## Normal Dev Workflow

Use this when making normal changes on `dev`:

```bash
git switch dev
git pull origin dev
git status
git add .
git commit -m "Describe the change"
git push origin dev
```

## Move Changes from Dev to Main

Use this when `dev` is ready and you want to update `main`.

### Option 1: Merge Dev into Main Locally

```bash
git switch main
git pull origin main
git merge dev
git push origin main
```

This keeps the branch history.

### Option 2: Merge Latest Remote Dev into Main

```bash
git switch main
git pull origin main
git fetch origin
git merge origin/dev
git push origin main
```

This is useful when `dev` was updated from another computer or GitHub.

### Option 3: Create a Pull Request

1. Push `dev`:

```bash
git push origin dev
```

2. Open GitHub.
3. Create a pull request:

```text
dev -> main
```

4. Review and merge on GitHub.

This is the cleanest approach when working with collaborators.

## View Commit History

Show recent commits:

```bash
git log --oneline
```

Show recent commits with branches:

```bash
git log --oneline --decorate -10
```

Show graph:

```bash
git log --oneline --graph --decorate --all
```

## See File Differences

Show unstaged changes:

```bash
git diff
```

Show staged changes:

```bash
git diff --staged
```

Show changes in one file:

```bash
git diff README.md
```

## Undo Before Commit

Unstage a file but keep changes:

```bash
git restore --staged README.md
```

Discard changes in one file:

```bash
git restore README.md
```

Warning: `git restore README.md` removes local edits in that file.

## Delete Branches

Delete a local branch:

```bash
git branch -d branch-name
```

Force delete a local branch:

```bash
git branch -D branch-name
```

Delete a remote branch:

```bash
git push origin --delete branch-name
```

## Rename a Branch

Rename current branch:

```bash
git branch -m new-branch-name
```

Rename another local branch:

```bash
git branch -m old-branch-name new-branch-name
```

## Check Remote Repository

Show remotes:

```bash
git remote -v
```

Check remote branches:

```bash
git ls-remote --heads origin
```

## Tag a Version

Create a tag:

```bash
git tag v1.0.0
```

Push the tag:

```bash
git push origin v1.0.0
```

List tags:

```bash
git tag
```

## Useful Safety Checklist Before Pushing

Before pushing, run:

```bash
git status
git diff --staged
```

For this Python project, also run:

```bash
.\.venv\Scripts\python.exe smoke_test.py
```

Then push:

```bash
git push origin dev
```

## Common Command Examples

Commit documentation change:

```bash
git add Documentation/
git commit -m "Update documentation"
git push origin dev
```

Commit README change:

```bash
git add README.md
git commit -m "Update README"
git push origin dev
```

Create a feature branch:

```bash
git switch dev
git pull origin dev
git switch -c feature/new-report
```

Merge feature branch back to dev:

```bash
git switch dev
git pull origin dev
git merge feature/new-report
git push origin dev
```

Update main from dev:

```bash
git switch main
git pull origin main
git merge dev
git push origin main
```
