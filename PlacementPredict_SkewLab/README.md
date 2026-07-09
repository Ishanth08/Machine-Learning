# PlacementPredict — Training–Serving Skew Lab

A tiny, hands-on lab where you **see and measure** how inconsistencies between the
*training* environment and the *deployment (serving)* environment silently corrupt
predictions. This bug class is called **training–serving skew**, and it is the most
common production-ML bug.

The trick of the lab: the model is trained **correctly** using one shared feature
function (`features.py`). Then, at "deployment time", we score new students with a
*slightly wrong* re-implementation of that function (`skews.py`) — exactly the kind
of drift that happens when the serving code is written separately from training.
Nothing crashes. The predictions just quietly get worse, and we measure by how much.

## Files

```
PlacementPredict_SkewLab/
├── make_data.py    # create dummy data (train.csv + live.csv); ~15% missing cgpa
├── features.py     # the CORRECT, shared feature builder (single source of truth)
├── train.py        # trains the model correctly -> model.joblib + train_stats.json
├── skews.py        # the 5 deployment BUGS (each breaks ONE thing)
├── demo_skew.py    # run ONE skew, see the accuracy drop + flipped examples
├── run_all.py      # run ALL skews, one summary table
└── requirements.txt
```

## Setup (once)

```bash
python -m venv .venv
#   Windows:      .venv\Scripts\activate
#   macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

python make_data.py    # make the dummy data
python train.py        # train the clean model
```

## Run it

See every skew at once:

```bash
python run_all.py
```

Example output:

```
                  skew accuracy acc_lost flipped
(none) correct serving    0.903    0.000       0
            no_scaling    0.873   +0.030      27
           zero_impute    0.843   +0.060      22
         category_case    0.897   +0.007       2
           wrong_order    0.907   -0.003       7
            unit_shift    0.873   +0.030      27
```

Examine one skew in detail (shows students whose answer flipped):

```bash
python demo_skew.py zero_impute
python demo_skew.py no_scaling
```

Run with no argument to list the available skews.

## The five skews (what each teaches)

| Skew | The inconsistency | Why it hurts |
|------|-------------------|--------------|
| `no_scaling` | Training divided cgpa/10 and aptitude/100; serving forgot to scale. | The model sees numbers 10–100× larger than it trained on. |
| `zero_impute` | A missing value is filled with `0` at serving, but training filled it with the **training mean**. | A missing-cgpa student looks like a 0.0-cgpa student → predictions flip. |
| `category_case` | `stream` arrives lower-case (`"cse"`), so no one-hot column matches → "unknown category". | The model loses a feature it relied on. |
| `wrong_order` | Two features are placed in the wrong slots of the vector. | The model reads soft-skills as experience and vice-versa. |
| `unit_shift` | An upstream form now sends CGPA as a percentage (0–100), but serving still divides by 10. | Every CGPA is 10× too big. |

## Things to notice (discussion points)

- **The failures are silent.** No exception, no error code — the API returns a
  perfectly valid `placed: true/false`. Only by comparing against the correct
  answers do you see the damage. In production you would not have the correct
  answers, which is what makes skew so dangerous.
- **Accuracy can hide it.** Look at `wrong_order`: several predictions *flipped*
  even though the overall accuracy barely moved. Aggregate metrics can mask a bug
  that is corrupting individual predictions — always watch the *flipped* count too.
- **The cure is structural, not heroic.** You do not fix skew by being careful.
  You fix it by making training and serving call the **same** feature code
  (`features.py`) — which is exactly why real systems use shared feature pipelines
  and feature stores.

## Try this

1. Open `skews.py`. Before running each function, predict what will happen.
2. Add your own skew (e.g. round CGPA to an integer) and add it to the `SKEWS`
   dict — then measure it with `run_all.py`.
3. "Fix" a skew by making it call `build_features` from `features.py` and confirm
   the accuracy loss disappears.

*Teaching project. Data is synthetic; the model is intentionally simple.*
