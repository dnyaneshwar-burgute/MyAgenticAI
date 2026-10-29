import argparse
from tokenizer_utils import tokenize_all, truncation_demo, words_vs_tokens_ratio, CONTEXT_WINDOWS

DEFAULT_TEXT = (
    "Hello, I recently ordered a pair of wireless headphones from ShopEase "
    "(order #48213) and they stopped working after just two days of normal use. "
    "I have already tried resetting them and charging them overnight, but the "
    "left earbud will not power on at all. Could you please help me arrange a "
    "replacement or a refund? I've attached my receipt for reference. Thank you!"
)


def main():
    parser = argparse.ArgumentParser(description="Compare tokenizers and context window usage.")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT)
    parser.add_argument("--max-length", type=int, default=20,
                         help="max_length to demo truncation with (kept small on purpose).")
    args = parser.parse_args()

    print("Input text:")
    print(f"  {args.text}\n")
    print(f"Word count (naive split): {len(args.text.split())}\n")

    print("=" * 70)
    print("TOKEN COUNTS BY TOKENIZER")
    print("=" * 70)
    results = tokenize_all(args.text)
    for label, info in results.items():
        ratio = words_vs_tokens_ratio(args.text, label)
        print(f"\n{label}")
        print(f"  Token count       : {info['count']}")
        print(f"  Tokens/word ratio : {ratio}")
        print(f"  Context window    : {info['window']:,} tokens")
        print(f"  % of window used  : {info['pct_of_window']}%")
        print(f"  First 12 tokens   : {info['tokens'][:12]}")

    print("\n" + "=" * 70)
    print(f"TRUNCATION DEMO (max_length={args.max_length})")
    print("=" * 70)
    for label in ("bert-base-uncased (WordPiece)", "gpt2 (BPE)", "t5-small (SentencePiece)"):
        kept, dropped = truncation_demo(args.text, label, args.max_length)
        print(f"\n{label}")
        print(f"  Kept ({len(kept)} tokens): {kept}")
        print(f"  Dropped: {dropped} token(s) beyond max_length={args.max_length}")

    print("\n" + "=" * 70)
    print("CONTEXT WINDOW REFERENCE TABLE")
    print("=" * 70)
    for label, window in CONTEXT_WINDOWS.items():
        print(f"  {label:<35} {window:>10,} tokens")


if __name__ == "__main__":
    main()
