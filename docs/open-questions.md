# Open questions

Everything the docs can't yet answer from the code is marked on its page in one
of two fixed forms, so the gaps can be found with grep:

- `!!! question "Open question"`: something to decide or investigate.
- `!!! note "Not yet documented"`: the answer lives outside this repo.

```bash
grep -rn -A2 -e '"Open question"' -e '"Not yet documented"' docs/
```

## Index

| Page | Marker | Topic |
|---|---|---|
| [How it works](how-it-works.md#building-the-automaton) | Open question | How common decomposed (non-NFC) text is in the shards, and whether to normalise article text before scanning |
| [Python usage](usage.md#terms-file) | Not yet documented | How the RNAcentral import pipeline generates the terms file |
| [RNA relevance filter](classifier.md#model_metadatajson-the-configuration) | Not yet documented | The script that exports a scikit-learn model to `model.safetensors` and `model_metadata.json` |
