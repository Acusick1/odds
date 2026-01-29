# Code Review Checklist

Guidelines for reviewing code in this repository.

## General

- If the PR is based on an issue, read the issue first.
- If a PR is open, unit and integration tests will already have run with our `test.yml` workflow. Check the results using `gh`.

- [ ] **Exercise the code** - Test new functionality through actual use cases (CLI commands, API calls) not just unit tests

## Code Quality

- [ ] **No compatibility hacks** - Delete unused code completely. Avoid `_unused` vars, re-export wrappers, or `# removed` comments
- [ ] **Proper abstraction** - Check for existing patterns that should be reused or consolidated. Avoid duplication
- [ ] **Single responsibility** - Functions and classes should have one clear purpose
- [ ] **Clear naming** - Variables, functions, and classes should reveal intent

## Type Safety & Patterns

- [ ] **Complete type hints** - All function signatures must have parameter and return type hints
- [ ] **Modern syntax** - Use `str | None` not `Optional[str]`, use `list[T]` not `List[T]`
- [ ] **Async correctness** - Proper async/await usage, no mixing sync/async database operations
- [ ] **Error handling** - Appropriate exceptions and logging, no silent failures

## Testing

- [ ] **Behavior-focused tests** - Test actual usage scenarios, not just implementation details
- [ ] **Appropriate coverage** - Critical paths and edge cases are tested
- [ ] **Consistent patterns** - Use pytest fixtures, parametrize where appropriate, follow existing test structure

## Style Consistency

- [ ] **Match codebase patterns** - Follow existing conventions for dataclasses, SQLModel usage, config patterns
- [ ] **No hardcoded values** - Use configuration for environment-specific values
- [ ] **Timezone handling** - Always use UTC with timezone-aware datetimes (`datetime.now(UTC)`)

## Security

- [ ] **No injection vulnerabilities** - SQL, command injection, etc.
- [ ] **Input validation** - Validate and sanitize user inputs
- [ ] **No secrets in code** - API keys, passwords in environment variables only
