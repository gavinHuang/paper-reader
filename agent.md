# AI Agent Guidelines

This document contains project-specific guidelines that AI assistants should follow when working on this codebase.

## Python Project Management

- **Use `uv` for all Python package management operations**
  - Install packages: `uv add <package>`
  - Run scripts: `uv run <script>`
  - Sync dependencies: `uv sync`
  - Do NOT use pip, poetry, or other package managers

## Shell Environment

- **You are running under PowerShell on Windows**
  - Use PowerShell-compatible commands
  - Be aware of path separators (backslash on Windows)
  - Quote paths appropriately for PowerShell

## Code Philosophy

- **Don't maintain backward compatibility unless explicitly requested**
  - Delete unused code completely
  - Refactor breaking changes directly
  - No deprecation warnings or compatibility shims

- **Don't create documentation files unless explicitly requested**
  - No README.md generation
  - No CHANGELOG.md or similar files
  - Focus on code, not documentation

- **Don't create test modules unless explicitly requested**
  - No automatic test file generation
  - Only write tests when specifically asked

## Error Handling

- **Don't create fallback solutions in try-catch sections unless explicitly requested**
  - Let errors propagate naturally
  - Don't silently catch and ignore exceptions
  - Only add error handling when specifically needed

- **Don't over-wrap error responses**
  - Print original error messages from external API calls directly
  - Don't transform or prettify error messages unnecessarily
  - Preserve stack traces and debugging information
