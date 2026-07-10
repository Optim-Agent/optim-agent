# Prompt Appendix Design

## Goal

Make the paper's prompting description match the current package and disclose
the full runtime sampler and pruner prompt contracts in an appendix.

## Method Updates

- Describe the current sampler efforts: `low` uses a five-trial window without
  requested reasoning or notes; `medium` uses ten trials with reasoning and
  notes; `high` uses twenty trials with reasoning and notes.
- Remove current-method claims about `xhigh`, `max`, full-history prompts, and
  ranked multi-candidate responses.
- Describe the actual summarized history: best, up to five promising, five
  recent, and three weak trials selected from the effort window. Completed and
  pruned trials with objective values are eligible; failed trials are absent.
- Preserve existing `xhigh` empirical results under their original name and
  mark them as legacy results collected before the effort ladder was reduced.
  Do not rename or reinterpret recorded measurements.
- State that the package passes one flat prompt string to the selected CLI; it
  does not construct separate system and user messages.

## Appendix

Add `\appendix` after the bibliography, following the cited paper's convention
of placing literal prompts in a dedicated post-reference appendix.

### AgentSampler Prompt

Present one parametric template rather than three repetitive rendered prompts.
Use bracketed placeholders for runtime data and visibly marked conditional
blocks for:

- study context and context-derived priors;
- the special `early reward` instruction;
- a carried `_note`;
- `_reasoning` and `_note` output fields.

The template must preserve the current instruction order and wording. Follow it
with the invalid-response retry suffix and a compact effort table explaining
which conditional blocks are active.

### AgentPruner Prompt

Present the complete pruner template with placeholders for objective direction,
the five reference curves, current parameters, current curve, and the
level-specific stance. Include the strict boolean JSON response contract.

## LaTeX Presentation

Use existing LaTeX facilities and manual line wrapping for readable literal
text. Add no package solely for prompt listings. Keep appendix material after
the references so the seven-page main body remains unchanged.

## Verification

- Search the paper for obsolete current-method claims and distinguish every
  retained `xhigh` occurrence as a legacy experimental label.
- Compare both appendix templates line by line with `optim_agent/samplers.py`
  and `optim_agent/pruners.py`.
- Run `paper/src/verify_paper.sh` and inspect LaTeX warnings, page count,
  references, and appendix rendering.

