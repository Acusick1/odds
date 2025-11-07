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

## Field Definitions

### Title

- Clear, action-oriented
- Format: `[Verb] [Subject]`
- Examples: "Add Type Hints to Test Fixtures", "Implement Backfill Checkpoint System"

### Goal

- 1-3 sentences explaining the problem or opportunity
- Focus on WHY, not HOW
- Context for decision-making

### Scope

- Bulleted list of implementation requirements
- Can be prescriptive ("Create X class") or outcome-based ("Implement detection for Y")
- Balance specificity with flexibility

### Success Criteria

- Checkbox format: `- [ ] Criterion`
- Concrete and testable
- Includes both functional and quality criteria (tests pass, docs updated, etc.)

### Location (optional)

- Use when multiple files affected or location non-obvious
- Format: File paths or directory paths
- Can be list or single item

### Implementation Approach (optional)

- Prescriptive guidance on architecture
- Specific patterns to follow
- Technology/library choices
- Reference to existing examples in codebase

### Constraints (optional)

- Non-negotiable requirements
- Patterns that must be followed
- Technology limitations
- Compatibility requirements

### Open Questions (optional)

- Areas where agent should explore options
- Design decisions to be made during implementation
- Tradeoffs to consider
- Encourages agent thinking within boundaries

### Complexity (optional)

- 🟢 Easy - Simple mechanical changes, clear path
- 🟡 Medium - Multiple files, some design decisions, moderate testing
- 🔴 Hard - Architectural changes, novel features, extensive testing

### Prerequisites (optional)

- Required knowledge for implementation
- Dependencies or understanding needed
- Useful for external agents unfamiliar with codebase

### Related Files (optional)

- Format: `path/to/file.py` - Context about this file
- Reference implementations or patterns to follow
- Files that provide useful context

### Notes (optional)

- Gotchas or warnings ("Don't use datetime.now() without timezone")
- Links to documentation
- Architectural context or decisions
- Performance considerations
- Security considerations

---

## Examples by Complexity Tier

### Tier 1: Simple Issue (4 core fields)

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

**Why this works**: Mechanical task with clear requirements. No complex decisions needed.

---

### Tier 2: Standard Issue (6-7 fields with prescriptive guidance)

```markdown
# Implement Discord Alert System

**Goal**: Activate the existing alert infrastructure by implementing Discord webhook integration for critical system events (low API quota, data quality issues, fetch failures).

**Complexity**: 🟡 Medium

**Location**:
- `alerts/discord.py` (new file)
- `jobs/fetch_odds.py` (add alert triggers)

**Implementation Approach**:
- Create `DiscordAlertChannel` class inheriting from `BaseAlertChannel`
- Implement `send_alert()` method using Discord webhook API
- Format alerts as Discord embeds with color-coded severity
- Use `aiohttp` for async HTTP requests (already in dependencies)

**Scope**:
- Implement webhook integration following abstract base class pattern
- Add alert triggers for: API quota <20%, critical data quality issues, fetch job failures
- Respect `ALERT_ENABLED` config flag
- Write unit tests with mocked webhook calls
- Handle webhook failures gracefully (log error, don't crash)

**Success Criteria**:
- [ ] Discord alerts send successfully when conditions met
- [ ] Alert embeds include timestamp, severity, description, and context
- [ ] Respects `ALERT_ENABLED=false` (no alerts sent)
- [ ] Gracefully handles webhook failures
- [ ] Unit tests pass with mocked HTTP requests

**Related Files**:
- `alerts/base.py` - `BaseAlertChannel` abstract class to inherit from
- `core/config.py` - `AlertConfig` settings to use

**Notes**:
- Discord webhook API: https://discord.com/developers/docs/resources/webhook
- Include rate limiting (Discord limits to ~5 requests/second)
- Consider batching alerts to avoid spam during outages
```

**Why this works**: Clear architectural guidance (inherit from base, use existing patterns) with implementation freedom (how to batch, rate limiting strategy).

---

### Tier 3: Complex Issue (hybrid prescriptive + outcome-based)

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

**Why this works**: Clear architectural constraints and patterns to follow, but gives agent freedom to explore detection algorithms, threshold tuning, and movement analysis strategies. The open questions guide exploration within boundaries.

---

## Usage Guidelines

### When Creating Issues

**Choose field set based on:**

- Task complexity
- Whether location is obvious
- Whether external agent needs context
- Whether design decisions are needed

**Default starting point**: Core 4 fields, add others as needed

**Ask yourself**:

- Is the location obvious? → Skip Location
- Are there gotchas or warnings? → Add Notes
- Should agent explore approaches? → Add Open Questions
- Is this for an external agent? → Add Prerequisites, Related Files
- Is this architectural? → Add Implementation Approach, Constraints

### Tone and Style

- **Imperative mood** for scope items ("Add type hints", not "Adding type hints")
- **Clear and concise** - avoid unnecessary words
- **Specific and actionable** - agent should know exactly what to do
- **Context where needed** - explain WHY when it's not obvious
- **Technical precision** - use correct terminology from codebase

---
