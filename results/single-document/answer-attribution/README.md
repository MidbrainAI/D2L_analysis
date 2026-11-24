# Answer attribution (Table 7)

The [400 scored rows](probe-and-generation-rows.json) contain the evaluated
responses and correctness scores. The [pinned references](reference-passages-and-answers.json)
contain the 57 SQuAD validation passages and their question-linked gold answers.
Keep a row for a model only when the text ICL teacher scored at least 0.5 token F1
and that model scored below 0.5.

The paper reports that **46% of 65 native D2L errors** matched a different
question’s answer. With whole-percent rounding this implies **30/65**; the
original per-row classifications and exact matching implementation were not
retained in the available artifacts. The numerator is inferred from the
published percentage, not recovered from flags.

The `answer-attribution` [offline command](../../README.md) documents and runs
an independent check. It strips outer ASCII punctuation and whitespace and
case-folds for a literal passage-substring match. For the other-answer check,
it uses the maximum normalized QA token F1 against gold answers of **other**
questions on that passage and flags F1 >= 0.5. It writes one row per error
with both flags; they can overlap.

| Model | Errors | Literal passage match | Other answer, token F1 >= 0.5 |
| --- | ---: | ---: | ---: |
| Native D2L | 65 | 37 | 25 |
| Rank-effect reader | 59 | 37 | 32 |
| Base | 291 | 32 | 18 |

The literal-span counts round to the paper's 57%, 63%, and 11%. The
other-answer check does **not** recover the paper's 46%, 58%, and 10%; use
it as a separate, reproducible diagnostic, not as a reproduction of those
three percentages.
