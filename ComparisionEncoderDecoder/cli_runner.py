import argparse
from architecture_utils import (
    run_encoder_only, run_decoder_only, run_encoder_decoder, ARCHITECTURE_INFO
)

DEFAULT_MASK_SENTENCE = "The customer said the [MASK] stopped working after two days."
DEFAULT_GPT2_PROMPT = "Dear ShopEase support team, I am writing to report that my order"
DEFAULT_T5_INPUT = (
    "summarize: The customer contacted ShopEase support because their headphones "
    "stopped working two days after delivery. The agent verified the order, "
    "confirmed it was within the 30-day return window, and issued a full refund."
)


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare BERT / GPT-2 / T5 on related tasks.")
    parser.add_argument("--mask-sentence", default=DEFAULT_MASK_SENTENCE)
    parser.add_argument("--gpt2-prompt", default=DEFAULT_GPT2_PROMPT)
    parser.add_argument("--t5-input", default=DEFAULT_T5_INPUT)
    args = parser.parse_args()

    print_section("ENCODER-ONLY  |  bert-base-uncased  |  Masked Language Modeling")
    print(f"Input : {args.mask_sentence}")
    print(f"Info  : {ARCHITECTURE_INFO['encoder']}")
    results = run_encoder_only(args.mask_sentence)
    print("Top predictions for [MASK]:")
    for word, prob in results:
        print(f"  {word:<15} {prob:.4f}")

    print_section("DECODER-ONLY  |  gpt2  |  Autoregressive Generation")
    print(f"Prompt: {args.gpt2_prompt}")
    print(f"Info  : {ARCHITECTURE_INFO['decoder']}")
    continuation = run_decoder_only(args.gpt2_prompt)
    print(f"Generated continuation:\n  ...{continuation}")

    print_section("ENCODER-DECODER  |  t5-small  |  Conditional Generation (summarization)")
    print(f"Input : {args.t5_input}")
    print(f"Info  : {ARCHITECTURE_INFO['encoder_decoder']}")
    summary = run_encoder_decoder(args.t5_input)
    print(f"Output:\n  {summary}")

    print_section("SIDE-BY-SIDE SUMMARY")
    for kind, info in ARCHITECTURE_INFO.items():
        print(f"\n{kind.upper()}")
        for k, v in info.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
