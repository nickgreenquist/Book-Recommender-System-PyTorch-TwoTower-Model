"""
Book Recommender System — CLI entry point.

Usage:
    python main.py preprocess [books|interactions]  # Stage 1: raw JSONs/CSVs → data/base_*.parquet
    python main.py features              # Stage 2: base parquets → data/features_*.parquet
    python main.py dataset               # Stage 3: features + raw → data/dataset_softmax_*_v1.pt
    python main.py dataset debug         # Stage 3: same but 10k train users only (fast debug build)
    python main.py train                 # Stage 4: full softmax training
    python main.py canary                # Canary user recommendations (most recent checkpoint)
    python main.py canary <path>         # Canary user recommendations (specific checkpoint)
    python main.py probe                 # Embedding probes (most recent checkpoint)
    python main.py probe <path>          # Embedding probes (specific checkpoint)
    python main.py export                # Stage 5: export serving artifacts to serving/
    python main.py export <path>         # Stage 5: export from specific checkpoint
    python main.py eval                  # Offline eval: Recall@K, NDCG@K, Hit Rate@K, MRR
    python main.py eval <path>           # Same, specific checkpoint
"""
import sys

DATA_DIR = 'data'
VERSION  = 'v1'


def cmd_preprocess(step=None):
    from src.preprocess import run
    run(data_dir=DATA_DIR, step=step)


def cmd_explore():
    from src.explore_thresholds import run
    run(data_dir=DATA_DIR)


def cmd_features():
    from src.features import run
    run(data_dir=DATA_DIR, version=VERSION)


def cmd_dataset(debug=False):
    from src.dataset import load_features, make_softmax_splits, save_softmax_splits

    print("Loading features ...")
    fs = load_features(DATA_DIR, VERSION)

    print("\nBuilding softmax datasets ...")
    kwargs = {'max_users': 10_000} if debug else {}
    train_data, val_data = make_softmax_splits(fs, DATA_DIR, **kwargs)
    save_softmax_splits(train_data, val_data, DATA_DIR, VERSION)


def cmd_train():
    from src.dataset import load_features, load_softmax_splits
    from src.train import get_softmax_config, build_model, train_softmax

    print("Loading features ...")
    fs = load_features(DATA_DIR, VERSION)

    print("\nLoading softmax datasets ...")
    train_data, val_data = load_softmax_splits(DATA_DIR, VERSION)

    config = get_softmax_config()
    model  = build_model(config, fs)
    train_softmax(model, train_data, val_data, config, fs)


def cmd_export(checkpoint_path=None):
    from src.export import run_export
    run_export(data_dir=DATA_DIR, checkpoint_path=checkpoint_path, version=VERSION)


def cmd_canary(checkpoint_path=None):
    from src.evaluate import run_canary
    run_canary(data_dir=DATA_DIR, checkpoint_path=checkpoint_path, version=VERSION)


def cmd_probe(checkpoint_path=None):
    from src.evaluate import run_probes
    run_probes(data_dir=DATA_DIR, checkpoint_path=checkpoint_path, version=VERSION)


def cmd_eval(checkpoint_path=None):
    from src.dataset import load_features
    from src.evaluate import _resolve_checkpoint, _load_model_and_embeddings
    from src.offline_eval import run_offline_eval

    cp = _resolve_checkpoint(checkpoint_path, 'saved_models')
    if cp is None:
        return

    print("Loading features ...")
    fs = load_features(DATA_DIR, VERSION)
    model, *_ = _load_model_and_embeddings(cp, fs)
    run_offline_eval(model, fs, checkpoint_path=cp, data_dir=DATA_DIR)


COMMANDS = {
    'preprocess': cmd_preprocess,
    'explore':    cmd_explore,
    'features':   cmd_features,
    'dataset':    cmd_dataset,
    'train':      cmd_train,
    'canary':     cmd_canary,
    'probe':      cmd_probe,
    'export':     cmd_export,
    'eval':       cmd_eval,
}

if __name__ == '__main__':
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(1)
    elif args[0] == 'preprocess':
        step = args[1] if len(args) > 1 else None
        if step not in (None, 'books', 'interactions'):
            print("Usage: python main.py preprocess [books|interactions]")
            sys.exit(1)
        cmd_preprocess(step=step)
    elif args[0] == 'dataset':
        debug = len(args) > 1 and args[1] == 'debug'
        cmd_dataset(debug=debug)
    elif args[0] in ('canary', 'probe', 'export', 'eval') and len(args) > 1:
        COMMANDS[args[0]](checkpoint_path=args[1])
    elif args[0] in COMMANDS:
        COMMANDS[args[0]]()
    else:
        print(__doc__)
        sys.exit(1)
