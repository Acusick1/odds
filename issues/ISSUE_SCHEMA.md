# GitHub Issue Schema

This document defines the markdown format for issues created by Claude and posted to GitHub.

## Philosophy: Mostly Prescriptive

Issues should be **mostly prescriptive** with strategic outcome-based flexibility:

- **Prescriptive**: Architecture, file locations, patterns to follow, technology choices
- **Outcome-based**: Algorithm choices, performance optimizations, data structures, edge case handling

This approach works because:

- Architectural thinking has already been done (see `CLAUDE.md`)
- External agents focus on execution, not design exploration
- Established patterns should be followed for consistency
- Review agents catch implementation issues

## Adaptive Schema

The schema adapts based on issue complexity. Not all fields are required for every issue.

### Core Fields (Always Required)

1. **Title** - Clear, action-oriented issue title
2. **Goal** - 1-3 sentence problem statement explaining WHY this matters
3. **Scope** - Bulleted list of what needs to change/be implemented
4. **Success Criteria** - Checkbox list of concrete, testable acceptance criteria

### Optional Fields (Include When Relevant)

5. **Location** - Primary files/directories affected (include when >1 file or non-obvious)
6. **Implementation Approach** - Prescribed architecture/pattern to follow (for prescriptive issues)
7. **Constraints** - Non-negotiable requirements (async patterns, no new dependencies, etc.)
8. **Open Questions** - Areas where agent should explore/decide (for outcome-based flexibility)
9. **Complexity** - Easy/Medium/Hard (when prioritizing or triaging)
10. **Prerequisites** - Required knowledge/dependencies (for external agents unfamiliar with codebase)
11. **Related Files** - Additional context files with explanations (for external agents)
12. **Notes** - Gotchas, warnings, documentation links, architectural decisions

---

## Examples

### Simple Issue

```markdown
# Add Type Hints to Test Fixtures

**Goal**: Improve test code quality by adding return type hints to all pytest fixtures.

**Scope**:
- Add return type hints to all `@pytest.fixture` functions in `tests/conftest.py`
- Use modern syntax (`str | None`, not `Optional[str]`)
- Follow existing project style from `core/` modules

**Success Criteria**:
- [ ] All fixtures have proper return type annotations
- [ ] pytest runs successfully with no new errors
- [ ] Type hints use modern syntax
```

---

### Complex Issue

```markdown
# Implement Line Movement Anomaly Detection Strategy

**Goal**: Create a betting strategy that detects and capitalizes on unusual line movement patterns that indicate sharp money activity (reverse line movement and steam moves).

**Complexity**: 🔴 Hard

**Location**:
- `analytics/strategies.py` (new `LineMovementStrategy` class)
- `tests/unit/test_strategies.py`

**Background**:
Line movement anomalies are key indicators of professional betting activity:
- **Reverse line movement**: Line moves toward underdog despite public betting on favorite (sharp money indicator)
- **Steam moves**: Rapid coordinated line movement across multiple books (synchronized sharp action)

**Implementation Approach** (prescriptive):
- Create `LineMovementStrategy` class inheriting from `BettingStrategy`
- Implement `evaluate_opportunity()` method per base class contract
- Support all markets (h2h, spreads, totals)
- Use `storage.readers.get_line_movement()` to fetch historical odds snapshots
- Return `BetOpportunity` with detailed rationale explaining detected anomaly
- Add to `AVAILABLE_STRATEGIES` registry
- Register in CLI backtest command

**Open Questions** (outcome-based flexibility):
- How to distinguish "sharp movement" from normal market making noise?
- What thresholds effectively detect steam without false positives?
- Should we track velocity (points/hour) or just absolute movement magnitude?
- How far back should we analyze snapshots per game?
- Should we use percentage change vs absolute change for different odds ranges?

**Constraints**:
- Must inherit from `BettingStrategy` base class
- Must work with existing backtesting engine
- Should query multiple snapshots to calculate movement (not just current vs opening)
- Cannot assume all bookmakers have data for all snapshots

**Scope**:
- Implement reverse line movement detection algorithm
- Implement steam move detection algorithm
- Add configurable thresholds (explore optimal defaults during implementation)
- Calculate movement metrics (velocity, magnitude, coordination)
- Return opportunities with confidence scores based on signal strength
- Write comprehensive unit tests with fixture data covering various scenarios
- Integration test with backtesting engine

**Success Criteria**:
- [ ] Strategy correctly implements `evaluate_opportunity()` method
- [ ] Detects reverse line movement scenarios
- [ ] Detects steam move scenarios
- [ ] Returns opportunities with confidence scores and detailed rationale
- [ ] Unit tests pass covering: no movement, reverse, steam, partial data
- [ ] Can run via CLI: `odds backtest run --strategy line_movement`
- [ ] Backtesting integration works correctly

**Prerequisites**:
- Understanding of sports betting line movement patterns
- Familiarity with backtesting framework architecture
- Knowledge of sharp vs public betting patterns

**Related Files**:
- `analytics/strategies.py` - See `BasicEVStrategy` and `ArbitrageStrategy` for pattern examples
- `analytics/backtesting/models.py` - `BetOpportunity` model definition
- `storage/readers.py` - `get_line_movement()` method for fetching historical odds
- `cli/commands/backtest.py` - Strategy registration pattern

**Notes**:
- May need to filter out small movements that are normal market making
- Consider time-based windows (e.g., movement in last 2 hours vs last 24 hours)
- Reverse line movement is most reliable when line moves >1 point against public
- Steam moves typically happen within 15-30 minute windows across 3+ books
```

---

## Quick Decision Guide

**Before creating an issue**: Check if the issue is necessary! The codebase may already contain the functionality or something similar to it. After reviewing the codebase, if you are not 100% clear on what the issue should contain, ask succinct clarifying questions. Continue to iterate with the user until you have a rounded overview of the requirements, constraints, and expected outcomes.

Choose fields based on task complexity, whether location is obvious, whether external agent needs context, and whether design decisions are needed.

**Default starting point**: Core 4 fields, add others as needed.

**Include optional fields when:**

- Location: Multiple files affected OR location non-obvious
- Notes: Gotchas, warnings, or important documentation links exist
- Open Questions: Agent should explore different approaches
- Prerequisites/Related Files: External agent unfamiliar with codebase
- Implementation Approach/Constraints: Architectural changes or patterns must be followed

## Tone and Style

- **Imperative mood** for scope items ("Add type hints", not "Adding type hints")
- **Clear and concise** - avoid unnecessary words
- **Specific and actionable** - agent should know exactly what to do
- **Context where needed** - explain WHY when it's not obvious
- **Technical precision** - use correct terminology from codebase

---
