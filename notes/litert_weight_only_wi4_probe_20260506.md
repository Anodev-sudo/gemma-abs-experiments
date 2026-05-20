# LiteRT Weight-Only WI4 Probe

## Summary

Tested weight-only int4 as the first alternative to the failing dynamic WI4
LiteRT export for the currently promoted q8 lineage.

The hypothesis was that keeping activations in float while storing weights at
4-bit precision might avoid the severe repetition seen with
`dynamic_wi4_afp32`.

That did not hold for the current export/runtime path.

## Baselines

| Artifact | Size | Smoke result |
| --- | ---: | --- |
| q8 LiteRT | 456,638,464 bytes | coherent |
| dynamic WI4 LiteRT | 238,731,264 bytes | degenerate repetition |

The q8 smoke produced a coherent summary. The dynamic WI4 smoke loaded and
decoded, but repeated the token `on`.

## What Was Tried

### Converter-side weight-only int4

The LiteRT Torch generative quant-config path rejected
`weight_only_int4_blockwise32` before export:

```text
Unsupported LayerQuantRecipe configuration. See get_supported_recipe_matrix()
```

In the installed `litert-torch` version, converter-side INT4 is supported for
dynamic-range blockwise recipes, not true weight-only recipes.

### Post-conversion AI Edge Quantizer recipes

Two preserved recipe files were tested through the post-conversion quantizer:

| Recipe | Output size | Smoke result |
| --- | ---: | --- |
| `weight_only_wi4_all_no_embed_gather.json` | 819,822,592 bytes | degenerate digit loop |
| `weight_only_wi4_all_no_logits.json` | 822,968,320 bytes | degenerate phrase loop |

The first emitted a fragment followed by repeated `3` characters. The second
looped on variants of `it's not as good`.

Both outputs were worse than the q8 control and neither preserves the size goal:
both were substantially larger than the q8 bundle. The likely reason is that
the protected components remain large while the weight-only representation does
not compact the overall `.litertlm` bundle the way the dynamic WI4 path does.

## Cleanup

The large failed export folders were removed after the probe to recover disk
space. Only this note and the smoke harness changes remain.

## Harness Changes

`scripts/smoke_litertlm_generation.py` now:

- detects long repeated non-whitespace characters, catching digit loops
- supports `--hard-exit-after-print` to avoid LiteRT teardown hangs after the
  smoke result is printed

## Conclusion

Weight-only int4 is not a useful first target for this LiteRT path. The
converter-side version is unsupported, and the post-conversion weight-only
recipes tested here both fail behaviorally while producing oversized artifacts.

The next useful knob is a more size-aware selective dynamic WI4 matrix, starting
with smaller blast-radius recipes such as fully-connected-only, MLP-only, or
attention-only, then preserving the most sensitive scopes once we know which
side causes the decode collapse.
