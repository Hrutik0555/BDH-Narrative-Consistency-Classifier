# Data

Place the following files in this folder (they are git-ignored by default):

| File        | Required columns                         |
|-------------|------------------------------------------|
| `train.csv` | `id`, `caption`, `content`, `label`      |
| `test.csv`  | `id`, `caption`, `content`               |

## Column description

- `id` - unique sample identifier (echoed in the output file).
- `caption` - short text (e.g. a backstory / claim) used to initialise the belief state.
- `content` - the long narrative text. It is split into sentences, and each sentence updates the belief state.
- `label` - `consistent` / `neutral` (mapped to 0) or `contradict` / `contradiction` (mapped to 1).

## Example

```csv
id,caption,content,label
1,"The hero was born in a small village.","The hero grew up in a village. He left at sixteen. He never returned home.",consistent
```

## Output

Running the pipeline writes `RES1.csv` (columns: `id`, `label`, `rationale`) to the repository root.

