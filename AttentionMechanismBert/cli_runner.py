import argparse
import os
import matplotlib.pyplot as plt

from attention_utils import run_attention, layer_head_matrix, averaged_matrix, top_attended_tokens

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def plot_heatmap(matrix, tokens, title, out_path):
    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.5), max(6, len(tokens) * 0.5)))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Key tokens (attended TO)")
    ax.set_ylabel("Query tokens (attending FROM)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize BERT self-attention for a sentence.")
    parser.add_argument("--text", type=str,
                         default="The customer returned the defective headphones because they stopped working.",
                         help="Input sentence to run through the model.")
    parser.add_argument("--layer", type=int, default=5, help="Layer index (0-11 for bert-base).")
    parser.add_argument("--head", type=int, default=3, help="Head index (0-11 for bert-base).")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model and running forward pass on: {args.text!r}")
    tokens, attentions = run_attention(args.text)
    print(f"Tokenized into {len(tokens)} tokens: {tokens}")
    print(f"Attention tensor shape (layers, heads, seq, seq): {attentions.shape}")

    single = layer_head_matrix(attentions, args.layer, args.head)
    plot_heatmap(single, tokens, f"Layer {args.layer}, Head {args.head}",
                 os.path.join(OUTPUT_DIR, f"attention_layer{args.layer}_head{args.head}.png"))

    avg = averaged_matrix(attentions, args.layer)
    plot_heatmap(avg, tokens, f"Layer {args.layer}, averaged across all heads",
                 os.path.join(OUTPUT_DIR, f"attention_layer{args.layer}_avg.png"))

    last_real_idx = len(tokens) - 2 if len(tokens) > 2 else len(tokens) - 1  # skip trailing [SEP]
    top = top_attended_tokens(attentions, tokens, args.layer, args.head, last_real_idx)
    print(f"\nToken {tokens[last_real_idx]!r} (layer {args.layer}, head {args.head}) attends most to:")
    for tok, weight in top:
        print(f"  {tok:<15} {weight:.4f}")


if __name__ == "__main__":
    main()
