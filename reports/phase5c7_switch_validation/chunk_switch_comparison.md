# Chunk Switch Comparison

- off chunk_count: 204
- on chunk_count: 209
- chunk_count_delta: 5
- off table_related chunk count: 0 (expected 0)
- on table_related chunk count: 22
- schema same: true
- off field set count: 1
- on field set count: 1
- top-level field diff: {'only_in_off': [], 'only_in_on': []}
- accepted_long_prose: 0
- ordinary long prose misabsorbed: false
- figure/reference/metadata misassociated: false
- no main paths touched: true

## Gate

- schema same = true
- top-level field diff empty = true
- off table_related count = 0: true
- on association_count > 0: true
- accepted_long_prose = 0: true
- no main paths touched: true
