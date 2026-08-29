# TeleScribe Kanban Rules

## State Mapping
| lane_label | canonical_state |
|------------|-----------------|
| backlog | backlog |
| ready | ready |
| in-progress | in-progress |
| blocked | blocked |
| review | review |
| done | done |

## Prioritization
1. Bugs in production (P0)
2. Feature requests from user (P1)
3. Quality-of-life improvements (P2)
4. Nice-to-haves (P3)

## Policies
- Card IDs use `KB-<number>` and never reuse IDs
- Any done move must include completion evidence in `log.md`
- Do not move blocked cards unless blocker is resolved
- "Done" means the feature is deployed and user has confirmed it works
- User controls when pushes happen — don't auto-push