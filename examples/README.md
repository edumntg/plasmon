# Examples

| Folder | Model | Data | Download | Time with one trainer |
|---|---|---|---|---|
| `mnist/` | `mnist_cnn`, 57k parameters | MNIST or Fashion-MNIST, 28×28 grey | 11 or 30 MB | 1 to 2 minutes |
| `cifar10/` | `cifar_cnn`, 0.8M parameters | CIFAR-10, 32×32 colour | 163 MB | 5 to 15 minutes |
| `shakespeare/` | `char_lm`, 0.8M parameters | Shakespeare as bytes | 1.1 MB | 3 to 7 minutes |
| `local-network/` | scripts for a server and trainers on one Wi-Fi | | | |
| `marketplace/` | `mnist_cnn` with funding, join mode, owner approval and sealed shards | MNIST | 11 MB | needs credits on (see docs/MARKETPLACE.md) |

Submit any job file with `plasmon job submit <file>`, or paste it into the dashboard.
`cifar10/` has a pair of files to compare one trainer with two.

## What plasmon can train today

Trainers run no code from the job: the architecture is a name from an allow-list in the
engine (`mlp`, `mnist_cnn`, `cifar_cnn`, `char_lm`) with a small config, and the data is
shards of arrays. That is a deliberate limit, because a trainer executes jobs from people it
does not know. To train another model, add it to `python/plasmon/train/models.py` in a
pull request: a class with a config and a `forward`, plus an entry in `_BUILDERS` and in
`ALLOWED_ARCHS`. Every machine updates once and can train it.

`model.init` starts a job from existing weights, so a job can continue or fine-tune a
model that a previous job produced.

## What it would take for a large language model

The round protocol is the one DiLoCo uses for language models at hundreds of millions
to billions of parameters: local steps, a compressed pseudo-gradient, an outer step. Three
things are missing for, say, a 3B model:

1. **The model and the data.** A transformer of that size as an allow-listed architecture
   that loads a public checkpoint, and token shards from a tokenizer instead of bytes.
2. **Memory.** Full fine-tuning of 3B parameters with AdamW needs about 40 GB of GPU
   memory. Adapter fine-tuning (LoRA) trains a few tens of millions of parameters and
   fits in 12 to 16 GB; it is the realistic path for home GPUs.
3. **Bandwidth.** A top-10 % update of 3B parameters is about 2.4 GB per trainer per
   round. With LoRA only the adapter deltas move: tens of megabytes per round.

Adapter fine-tuning of a public checkpoint is therefore the next step on the roadmap, not
a configuration change.
