
# Agent Instructions

## Environment

This repository is developed on a university HPC cluster accessed
over SSH. The cluster has security restrictions that affect how
Codex can modify files.

Known environment properties:
- Linux host with `user.max_net_namespaces=0`.
- Disk space and inode availability are not the issue.
- Network namespace creation is restricted by cluster policy.
- Codex's native filesystem sandbox helper may fail with
  `No space left on device` (ENOSPC).

## IMPORTANT: File Editing

Avoid using Codex's native `apply_patch` tool on this host.

The native patch helper may attempt to create Linux namespaces
and fail before reading or updating files.

Instead:
- Prefer ordinary shell commands running inside the existing
  permitted command sandbox.
- Use GNU `patch`, Python scripts, or other standard file-editing
  utilities to modify files.
- For small edits, Python-based text replacement is acceptable.
- For larger edits, use carefully constructed patches or scripts.
- Preserve original file formatting and avoid unrelated changes.
- After editing, verify changes with `git diff`.

If a file-editing operation fails with ENOSPC:
1. Do not immediately assume the filesystem is full.
2. Remember that network namespace creation is disabled.
3. Do not repeatedly retry the failing native patch helper.
4. Switch to a supported shell-based editing method.
5. Report the blocker if normal shell-based editing also fails.

## Cluster Safety

- Do not change kernel settings or namespace limits.
- Do not attempt to bypass university security restrictions.
- Do not disable sandbox protections to work around editing errors.
- Do not run GPU-intensive workloads on login nodes.
- Use the cluster's supported job scheduler for heavy workloads.
- Do not install system-wide packages or alter shared environments
  without explicit permission.

## Development Workflow

- Inspect relevant files before changing them.
- Keep edits focused on the requested task.
- Avoid unrelated refactoring.
- Use existing project conventions.
- Run appropriate lightweight validation after modifications.
- Report tests that could not be run.
- Never claim a change succeeded without verifying it.
- Do not commit or push unless explicitly requested.
