"""IOI baseline: token length check and Mary-minus-John logit diff on clean."""

import torch
from transformer_lens import HookedTransformer

CLEAN = "When John and Mary went to the store, John gave a bottle of milk to"
CORRUPT = "When John and Mary went to the store, Mary gave a bottle of milk to"


def first_token_id(model: HookedTransformer, fragment: str) -> int:
    ids = model.tokenizer.encode(fragment, add_special_tokens=False)
    assert len(ids) >= 1, f"No tokens for fragment: {fragment!r}"
    return int(ids[0])


def main() -> None:
    model = HookedTransformer.from_pretrained("gpt2-small")
    prepend_bos = model.cfg.default_prepend_bos

    clean_toks = model.to_tokens(CLEAN, prepend_bos=prepend_bos)
    corrupt_toks = model.to_tokens(CORRUPT, prepend_bos=prepend_bos)

    assert clean_toks.shape == corrupt_toks.shape, (
        "Clean and corrupt must tokenize to the same length for IOI patching. "
        f"clean {clean_toks.shape} vs corrupt {corrupt_toks.shape}"
    )
    print(
        f"Token length check: OK (seq_len={clean_toks.shape[1]}, "
        f"prepend_bos={prepend_bos})"
    )

    id_mary = first_token_id(model, " Mary")
    id_john = first_token_id(model, " John")

    with torch.inference_mode():
        logits = model(clean_toks)
    last_pos = logits.shape[1] - 1
    logit_diff = logits[0, last_pos, id_mary] - logits[0, last_pos, id_john]

    print(f"Clean prompt: {CLEAN!r}")
    print(
        f"Logit diff at final position (last={last_pos}): "
        f"logit(' Mary'[0]) - logit(' John'[0]) = {logit_diff.item():.4f}"
    )
    print(f"  (token ids: Mary_start={id_mary}, John_start={id_john})")


if __name__ == "__main__":
    main()
