#!/usr/bin/env python3
"""List operation scopes from a TFLite flatbuffer.

Useful for building selective AI Edge Quantizer recipes, where scope regexes
match the op scope reported by the quantizer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_edge_quantizer.utils import tfl_flatbuffer_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--operation", action="append", default=[])
    parser.add_argument("--contains", default="")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    buffer = memoryview(tfl_flatbuffer_utils.get_model_content(args.model))
    model = tfl_flatbuffer_utils.read_model(buffer)
    wanted_ops = {op.upper() for op in args.operation}
    printed = 0
    for subgraph_index, subgraph in enumerate(model.subgraphs):
        for op_index, op in enumerate(subgraph.operators):
            op_code = model.operatorCodes[op.opcodeIndex].builtinCode
            operation_value = tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME.get(
                op_code, f"OP_{op_code}"
            )
            operation = getattr(operation_value, "value", str(operation_value))
            if wanted_ops and operation not in wanted_ops:
                continue
            scope = tfl_flatbuffer_utils.get_op_scope(op, subgraph.tensors)
            if args.contains and args.contains not in scope:
                continue
            print(f"{subgraph_index}\t{op_index}\t{operation}\t{scope}")
            printed += 1
            if args.limit and printed >= args.limit:
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
