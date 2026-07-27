# AGENTS.md

Guidance for AI agents and contributors working in this repository.

## What this repository is

A public MkDocs site of reinforcement-learning notes, written by one author and
read by people learning RL. The two jobs of the notes are to build a correct
understanding from the ground up, and to serve as interview preparation
material. Assume the reader knows basic probability, calculus, and neural
networks, but not RL.

## How to write

Clarity first. The reader is learning the material, not admiring the prose.

- Use plain, direct English. Short sentences. Say the thing.
- Define a term the first time it appears, then use it consistently.
- Explain *why* a step is taken, not only what it is. A derivation without its
  motivation is not useful to a beginner.
- Build up in small steps, and connect each section to the previous one.
- No metaphors, no flourish, no filler like "it is worth noting that". Keep it
  professional: this repository is public.
- Do not use emoji.
- Prefer prose and short bullet lists. Use a table only for a small set of
  comparable facts, and keep the explanation in the surrounding text.
- Hard-wrap Markdown around 80 characters.

## Page structure

```
docs/<topic>/index.md      Explanations, derivations
docs/<topic>/q-and-a.md    Interview-style questions
```

- Every new page must be added to `nav` in `mkdocs.yml`, otherwise the strict
  build fails.
- `index.md` starts with an `# H1` title, then `## H2` sections in the order the
  ideas are learned. Use `### H3` for steps inside a section.
- Cross-link related pages with relative Markdown links, e.g.
  `[Off-policy learning](off-policy.md)`.
- A `### Notes to add` list at the end of a page is a placeholder for planned
  content. Remove entries as they are written.
- Pages are written in English. `policy-optimization/policy-improvement.md` also
  carries a `# 中文版本` mirror of the whole page; add one only when asked.

## Q&A pages

These exist to prepare for interviews and to check whether the author's
understanding is deep enough, so a question is only useful if it probes
understanding rather than recall.

- Ask "why" and "what breaks if not", not "what is the definition of".
- Format: the question as an `## H2` heading, the answer as one or two sentences
  directly below. Keep the answer to the point an interviewer wants to hear.
- Wrap the whole list in the styled container:

```markdown
<div class="qa-list" markdown>

## Why does DQN need a target network?

Keeping the bootstrap target temporarily fixed reduces feedback loops during
optimization.

</div>
```

## Math

Math is rendered by MathJax through `pymdownx.arithmatex` in generic mode, so
standard LaTeX works. Keep it consistent with existing pages.

- Inline math with `$...$`, display math with `$$` on its own line above and
  below the expression.
- Use `\mid` for conditioning, `\middle|` inside `\left...\right`,
  `\begin{aligned}` for multi-step derivations, `\mathbb{E}` for expectations.
- Notation already in use: states $s_t$, actions $a_t$, rewards $r$, discount
  $\gamma$, policy $\pi_\theta$ with actor parameters $\theta$, critic
  parameters $\phi$, reward-to-go $G_{i,t}$, value functions $Q^\pi$, $V^\pi$,
  $A^\pi$. Batch-indexed quantities are bold: $\mathbf{s}_{i,t}$ over samples
  $i=1,\dots,N$ and steps $t=1,\dots,T$.
- Explain every new symbol in the sentence that introduces it.
- Use `!!! note` or `!!! warning` admonitions for practical caveats and
  implementation traps, not for core content.

## Before finishing

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv run mkdocs build --strict   # must pass: catches broken links and nav errors
uv run mkdocs serve            # local preview at http://127.0.0.1:8000/
```

The editor's Markdown preview does not render admonitions or the Material
theme, so check anything visual in `mkdocs serve`.
