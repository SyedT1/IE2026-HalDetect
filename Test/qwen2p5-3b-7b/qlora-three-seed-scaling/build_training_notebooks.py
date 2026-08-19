"""Generate 12 standalone QLoRA training notebooks.

The model, dataset, masking, and manual training-loop cells live here as the canonical
source. This builder specializes them to one training seed and one fixed nested subset
per notebook.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

SEEDS = (13, 73, 101)
SIZES = {
    "2k": (2000, 500),
    "2348": (2348, 587),
    "2600": (2600, 650),
    "3000": (3000, 750),
}
DATA_SUBSET_SEED = 42


ENVIRONMENT_CELL = r"""
import os, warnings

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
os.environ['BNB_CUDA_VERSION'] = '128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore', message='.*use_reentrant.*')

for _major in ('12', '13'):
    _src = f'/usr/local/cuda/lib64/libnvJitLink.so.{_major}'
    _dst = '/usr/local/cuda/lib64/libnvJitLink.so.13'
    if os.path.exists(_src) and not os.path.exists(_dst):
        os.symlink(_src, _dst)
        print(f'Symlinked .{_major} -> .13')
        break

!pip install -q -U 'transformers>=4.49.0' 'peft>=0.10.0' accelerate bitsandbytes qwen-vl-utils safetensors 2>&1 | tail -4
print('Dependencies ready.')
"""


DETERMINISM_CELL = r"""
def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_generator(seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def stable_hash(values):
    payload = json.dumps(values, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def package_versions():
    import platform, transformers, peft, bitsandbytes
    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'peft': peft.__version__,
        'bitsandbytes': bitsandbytes.__version__,
        'cuda': torch.version.cuda,
        'cudnn': str(torch.backends.cudnn.version()),
        'gpu': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


seed_everything(TRAINING_SEED)
print('Deterministic controls enabled.')
"""


DATASET_CELL = r"""
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = (
    'You are a visual fact-checker examining an image from the Arab world.\n'
    'Below are THREE statements about this image. '
    'Exactly ONE statement is grounded in the image (True). '
    'The other two are plausible-sounding hallucinations (False).'
)

USER_TEMPLATE = (
    'Statement 1: {s0}\n'
    'Statement 2: {s1}\n'
    'Statement 3: {s2}\n\n'
    'Instructions:\n'
    '- On the VERY FIRST line write ONLY: "Answer: X" where X is 1, 2, or 3.\n'
    '- For each statement evaluate:\n'
    '    (a) Colour/texture evidence for or against\n'
    '    (b) Shape/form evidence for or against\n'
    '    (c) Contextual evidence for or against\n'
    '- Then state your conclusion.\n'
    'Do not write anything before the Answer line.'
)


class HalDetectTrainDataset(Dataset):
    def __init__(self, records, paths, processor, max_seq_len):
        self.records = records
        self.paths = paths
        self.processor = processor
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        true_idx = record['labels'].index(True)
        target = f'Answer: {true_idx + 1}'
        user_text = USER_TEMPLATE.format(
            s0=record['statements'][0],
            s1=record['statements'][1],
            s2=record['statements'][2],
        )
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': [
                {'type': 'image', 'image': self.paths[record['image']]},
                {'type': 'text', 'text': user_text},
            ]},
            {'role': 'assistant', 'content': target},
        ]
        prompt_text = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(messages)
        enc_full = self.processor(
            text=[full_text], images=image_inputs, truncation=False, return_tensors='pt')
        enc_prompt = self.processor(
            text=[prompt_text], images=image_inputs, truncation=False, return_tensors='pt')

        input_ids = enc_full['input_ids'][0][:self.max_seq_len]
        attention_mask = enc_full['attention_mask'][0][:self.max_seq_len]
        prompt_len = min(enc_prompt['input_ids'].shape[1], self.max_seq_len)
        pad_id = self.processor.tokenizer.pad_token_id or 0
        pad_len = self.max_seq_len - input_ids.shape[0]
        if pad_len:
            input_ids = torch.cat([
                input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            attention_mask = torch.cat([
                attention_mask, torch.zeros(pad_len, dtype=torch.long)])
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': enc_full['pixel_values'],
            'image_grid_thw': enc_full['image_grid_thw'],
            'labels': labels,
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch]),
        'pixel_values': torch.cat([x['pixel_values'] for x in batch], dim=0),
        'image_grid_thw': torch.cat([x['image_grid_thw'] for x in batch], dim=0),
    }


train_processor = AutoProcessor.from_pretrained(VLM_MODEL, max_pixels=TRAIN_MAX_PIXELS)
print('Training processor ready.')
"""


MODEL_CELL = r"""
from collections import Counter
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import (
    LoraConfig, get_peft_model, set_peft_model_state_dict,
    TaskType, prepare_model_for_kbit_training,
)
from safetensors.torch import load_file as load_safetensors


def build_fresh_model(seed):
    seed_everything(seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    max_memory = {i: '13500MiB' for i in range(torch.cuda.device_count())}
    max_memory['cpu'] = '4GiB'
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL,
        torch_dtype=dtype,
        device_map='auto',
        max_memory=max_memory,
        quantization_config=quant_config,
    )
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    for name, parameter in model.named_parameters():
        if 'visual' in name:
            parameter.requires_grad = False

    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            if parameter.requires_grad:
                digest.update(name.encode('utf-8'))
                digest.update(parameter.detach().float().cpu().numpy().tobytes())
    init_hash = digest.hexdigest()
    print('Layer distribution:', dict(Counter(model.hf_device_map.values())))
    model.print_trainable_parameters()
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if 'visual' in name
    ), 'Vision parameters must remain frozen.'
    return model, dtype, init_hash
"""


TRAINING_CELL = r"""
from torch.optim import AdamW
from torch.utils.data import Subset
from transformers import get_cosine_schedule_with_warmup


def checkpoint_with_marker(checkpoint_root):
    for name in ('latest', 'previous'):
        candidate = checkpoint_root / name
        if (candidate / 'COMPLETE').exists():
            return candidate
    return None


def external_checkpoint_with_marker():
    if EXTERNAL_CHECKPOINT_PATH is None:
        return None
    root = Path(EXTERNAL_CHECKPOINT_PATH)
    if root.is_file() and root.suffix.lower() == '.zip':
        extracted = OUTPUT_ROOT / 'external_checkpoint_extracted'
        if extracted.exists():
            shutil.rmtree(extracted)
        extracted.mkdir(parents=True)
        with zipfile.ZipFile(root) as archive:
            base = extracted.resolve()
            for member in archive.infolist():
                target = (extracted / member.filename).resolve()
                if base != target and base not in target.parents:
                    raise ValueError(f'Unsafe checkpoint archive member: {member.filename}')
            archive.extractall(extracted)
        root = extracted
    candidates = (root, root / 'latest', root / 'checkpoints' / 'latest')
    for candidate in candidates:
        if (candidate / 'COMPLETE').exists():
            print('Found attached external checkpoint:', candidate)
            return candidate
    raise FileNotFoundError(
        f'EXTERNAL_CHECKPOINT_PATH has no complete checkpoint: {root}')


def save_rng_state(path):
    torch.save({
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all(),
    }, path)


def restore_rng_state(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])


def save_training_checkpoint(
        seed_dir, model, optimizer, scheduler, log_rows, global_step,
        running_loss, running_batches, elapsed_seconds):
    checkpoint_root = seed_dir / 'checkpoints'
    staging = checkpoint_root / 'staging'
    latest = checkpoint_root / 'latest'
    previous = checkpoint_root / 'previous'
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    model.save_pretrained(staging / 'adapter')
    train_processor.save_pretrained(staging / 'processor')
    torch.save(optimizer.state_dict(), staging / 'optimizer.pt')
    torch.save(scheduler.state_dict(), staging / 'scheduler.pt')
    save_rng_state(staging / 'rng_state.pt')
    pd.DataFrame(log_rows).to_csv(staging / 'training_log.csv', index=False)
    state = {
        'recipe_name': RECIPE_NAME,
        'training_seed': TRAINING_SEED,
        'data_subset_seed': DATA_SUBSET_SEED,
        'train_subsample_n': TRAIN_SUBSAMPLE_N,
        'subset_order_sha256': subset_hash,
        'max_steps': MAX_STEPS,
        'global_step': global_step,
        'completed_microbatches': global_step * GRAD_ACCUM,
        'running_loss': running_loss,
        'running_batches': running_batches,
        'elapsed_seconds': elapsed_seconds,
        'packages': package_versions(),
    }
    (staging / 'training_state.json').write_text(
        json.dumps(state, indent=2), encoding='utf-8')
    (staging / 'COMPLETE').write_text('complete\n', encoding='utf-8')

    # Two-slot rotation: a timeout during replacement leaves either latest or previous.
    if previous.exists():
        shutil.rmtree(previous)
    if latest.exists():
        latest.rename(previous)
    staging.rename(latest)
    if previous.exists():
        shutil.rmtree(previous)

    # A single rolling archive is easy to download or upload as a Kaggle Dataset.
    resume_archive = seed_dir / (
        f'resume_checkpoint_n{TRAIN_SUBSAMPLE_N}_seed{TRAINING_SEED}.zip')
    temporary_archive = resume_archive.with_suffix('.tmp')
    if temporary_archive.exists():
        temporary_archive.unlink()
    with zipfile.ZipFile(temporary_archive, 'w', zipfile.ZIP_STORED) as archive:
        for path in sorted(latest.rglob('*')):
            if path.is_file():
                archive.write(path, Path('latest') / path.relative_to(latest))
    os.replace(temporary_archive, resume_archive)
    print(f'Checkpoint saved at optimizer step {global_step}: {latest}')
    print('Portable resume archive:', resume_archive)


def restore_training_checkpoint(checkpoint, model, optimizer, scheduler):
    state = json.loads((checkpoint / 'training_state.json').read_text(encoding='utf-8'))
    expected = {
        'recipe_name': RECIPE_NAME,
        'training_seed': TRAINING_SEED,
        'data_subset_seed': DATA_SUBSET_SEED,
        'train_subsample_n': TRAIN_SUBSAMPLE_N,
        'subset_order_sha256': subset_hash,
        'max_steps': MAX_STEPS,
        'packages': package_versions(),
    }
    for key, value in expected.items():
        assert state[key] == value, f'Checkpoint mismatch for {key}: {state[key]} != {value}'

    adapter_dir = checkpoint / 'adapter'
    safe_path = adapter_dir / 'adapter_model.safetensors'
    bin_path = adapter_dir / 'adapter_model.bin'
    if safe_path.exists():
        adapter_state = load_safetensors(str(safe_path), device='cpu')
    elif bin_path.exists():
        adapter_state = torch.load(bin_path, map_location='cpu', weights_only=True)
    else:
        raise FileNotFoundError(f'No adapter weights in {adapter_dir}')
    result = set_peft_model_state_dict(model, adapter_state, adapter_name='default')
    if getattr(result, 'unexpected_keys', None):
        raise ValueError(f'Unexpected adapter keys: {result.unexpected_keys[:5]}')

    optimizer.load_state_dict(torch.load(
        checkpoint / 'optimizer.pt', map_location='cpu', weights_only=False))
    scheduler.load_state_dict(torch.load(
        checkpoint / 'scheduler.pt', map_location='cpu', weights_only=False))
    log_rows = pd.read_csv(checkpoint / 'training_log.csv').to_dict(orient='records')
    assert len(log_rows) == state['global_step']
    restore_rng_state(checkpoint / 'rng_state.pt')
    print(f'Restored complete checkpoint from step {state["global_step"]}: {checkpoint}')
    return state, log_rows


def train_one_seed(seed, model, dtype, seed_dir):
    dataset = HalDetectTrainDataset(
        train_records, image_paths, train_processor, MAX_SEQ_LEN)
    loader_seed = seed + 10_000
    fixed_train_order = torch.randperm(
        len(dataset), generator=make_generator(loader_seed)).tolist()
    warmup_steps = int(MAX_STEPS * WARMUP_RATIO)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=MAX_STEPS,
    )

    log_rows = []
    global_step = 0
    completed_microbatches = 0
    running_loss = 0.0
    running_batches = 0
    elapsed_before_resume = 0.0
    checkpoint = checkpoint_with_marker(seed_dir / 'checkpoints')
    if checkpoint is None:
        checkpoint = external_checkpoint_with_marker()
    if checkpoint is not None:
        state, log_rows = restore_training_checkpoint(
            checkpoint, model, optimizer, scheduler)
        global_step = state['global_step']
        completed_microbatches = state['completed_microbatches']
        running_loss = state['running_loss']
        running_batches = state['running_batches']
        elapsed_before_resume = state['elapsed_seconds']

    assert completed_microbatches == global_step * GRAD_ACCUM
    assert completed_microbatches <= len(fixed_train_order)
    remaining_order = fixed_train_order[completed_microbatches:]
    remaining_dataset = Subset(dataset, remaining_order)
    train_loader = DataLoader(
        remaining_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        generator=make_generator(loader_seed + global_step),
        worker_init_fn=seed_worker,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        prefetch_factor=2 if NUM_WORKERS else None,
        persistent_workers=bool(NUM_WORKERS),
        pin_memory=False,
    )
    remaining_steps = len(train_loader) // GRAD_ACCUM
    assert remaining_steps == MAX_STEPS - global_step, (
        remaining_steps, MAX_STEPS - global_step)

    model.train()
    for module in model.modules():
        if 'Visual' in type(module).__name__:
            module.eval()
    optimizer.zero_grad(set_to_none=True)
    started = time.time() - elapsed_before_resume

    if global_step < MAX_STEPS:
        pbar = tqdm(train_loader, desc=f'seed={seed} resume_step={global_step}')
        for batch_index, batch in enumerate(pbar):
            batch['pixel_values'] = batch['pixel_values'].to(dtype)
            batch = {
                key: value.to('cuda:0') if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            outputs = model(**batch)
            unscaled_loss = outputs.loss
            (unscaled_loss / GRAD_ACCUM).backward()
            running_loss += unscaled_loss.item()
            running_batches += 1

            if (batch_index + 1) % GRAD_ACCUM != 0:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            row = {
                'seed': seed,
                'epoch': 1,
                'global_step': global_step,
                'loss': float(unscaled_loss.item()),
                'running_loss': running_loss / running_batches,
                'learning_rate': scheduler.get_last_lr()[0],
                'elapsed_hours': (time.time() - started) / 3600,
            }
            log_rows.append(row)
            pbar.set_postfix(step=global_step, loss=f'{row["running_loss"]:.4f}')

            if global_step % LOGGING_STEPS == 0:
                print(row)
            if global_step % SAVE_STEPS == 0:
                save_training_checkpoint(
                    seed_dir, model, optimizer, scheduler, log_rows, global_step,
                    running_loss, running_batches, time.time() - started)

    assert global_step == MAX_STEPS, f'Expected {MAX_STEPS}, got {global_step}'
    if global_step % SAVE_STEPS:
        save_training_checkpoint(
            seed_dir, model, optimizer, scheduler, log_rows, global_step,
            running_loss, running_batches, time.time() - started)
    pd.DataFrame(log_rows).to_csv(seed_dir / 'training_log.csv', index=False)
    adapter_dir = seed_dir / 'adapter_final'
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    train_processor.save_pretrained(adapter_dir)
    return log_rows
"""


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def config_cell(seed: int, size: int, steps: int) -> str:
    return f"""
    from pathlib import Path
    import gc, hashlib, json, math, random, re, shutil, time, zipfile
    import numpy as np
    import pandas as pd
    import torch
    from transformers import set_seed

    REPO_ID = 'QCRI/AynVQA-ArabicNLP26'
    TASK = 'task1b'
    LANG = 'en'
    VLM_MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'

    TRAINING_SEED = {seed}
    DATA_SUBSET_SEED = {DATA_SUBSET_SEED}
    TRAIN_SUBSAMPLE_N = {size}
    MAX_STEPS = {steps}
    NUM_EPOCHS = 1
    BATCH_SIZE = 1
    GRAD_ACCUM = 4
    LEARNING_RATE = 2e-4
    WARMUP_RATIO = 0.05
    MAX_SEQ_LEN = 1280
    TRAIN_MAX_PIXELS = 256 * 28 * 28

    LORA_RANK = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGETS = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ]

    SAVE_STEPS = 100
    LOGGING_STEPS = 10
    NUM_WORKERS = 2
    RECIPE_NAME = (
        f'qwen2p5-vl-7b_qlora_fixed-n{{TRAIN_SUBSAMPLE_N}}_'
        f'one-epoch_seed{{TRAINING_SEED}}'
    )
    OUTPUT_ROOT = Path(
        f'/kaggle/working/qlora_q7b_n{{TRAIN_SUBSAMPLE_N}}_seed{{TRAINING_SEED}}')
    # For a fresh Kaggle session, attach the prior checkpoint ZIP as a Dataset and set
    # this to its extracted directory, e.g. Path('/kaggle/input/my-checkpoint').
    EXTERNAL_CHECKPOINT_PATH = None

    assert TRAINING_SEED in {{13, 73, 101}}
    assert TRAIN_SUBSAMPLE_N in {{2000, 2348, 2600, 3000}}
    assert MAX_STEPS == math.ceil(TRAIN_SUBSAMPLE_N / (BATCH_SIZE * GRAD_ACCUM))
    assert torch.cuda.is_available(), 'A CUDA GPU is required.'
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print('Recipe:', RECIPE_NAME)
    print('Training seed:', TRAINING_SEED)
    print('Fixed subset seed:', DATA_SUBSET_SEED)
    print('Train examples / optimizer steps:', TRAIN_SUBSAMPLE_N, MAX_STEPS)
    print('Output:', OUTPUT_ROOT)
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        print(f'GPU {{i}}: {{prop.name}} ({{prop.total_memory / 1024**3:.1f}} GiB)')
    """


DOWNLOAD_CELL = r"""
from huggingface_hub import hf_hub_download, login
from tqdm.auto import tqdm

# Public downloads need no credential. If authentication is required, add HF_TOKEN to
# Kaggle Secrets; never paste a token into this notebook.
if os.getenv('HF_TOKEN'):
    login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)


def read_train_split():
    path = hf_hub_download(
        REPO_ID, filename=f'{TASK}/train_{LANG}.jsonl', repo_type='dataset')
    with open(path, encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 3000
    assert all('labels' in row and sum(row['labels']) == 1 for row in rows)
    assert len({row['id'] for row in rows}) == len(rows)
    return rows


all_train_records = read_train_split()
fixed_order = list(range(len(all_train_records)))
random.Random(DATA_SUBSET_SEED).shuffle(fixed_order)
selected_indices = fixed_order[:TRAIN_SUBSAMPLE_N]
train_records = [all_train_records[index] for index in selected_indices]
subset_ids = [record['id'] for record in train_records]
subset_hash = stable_hash(subset_ids)

print('Selected:', len(train_records))
print('Nested subset SHA256:', subset_hash)
print('First five IDs:', subset_ids[:5])

image_paths, failed_images = {}, []
for rel in tqdm(sorted({record['image'] for record in train_records}), desc='images'):
    try:
        image_paths[rel] = hf_hub_download(REPO_ID, filename=rel, repo_type='dataset')
    except Exception as exc:
        failed_images.append({'image': rel, 'error': repr(exc)})

assert not failed_images, f'Image downloads failed: {failed_images[:5]}'
print(f'Downloaded {len(image_paths)} unique images.')
"""


RUN_CELL = r"""
seed_dir = OUTPUT_ROOT
adapter_dir = seed_dir / 'adapter_final'
adapter_weights = [adapter_dir / 'adapter_model.safetensors', adapter_dir / 'adapter_model.bin']

if any(path.exists() for path in adapter_weights):
    raise FileExistsError(
        f'Final adapter already exists in {adapter_dir}; refusing to overwrite it.')

seed_everything(TRAINING_SEED)
model, dtype, lora_init_hash = build_fresh_model(TRAINING_SEED)
manifest = {
    'recipe_name': RECIPE_NAME,
    'training_seed': TRAINING_SEED,
    'data_subset_seed': DATA_SUBSET_SEED,
    'seed_derivations': {
        'global_lora_dropout_framework': TRAINING_SEED,
        'dataloader_order': TRAINING_SEED + 10_000,
    },
    'train_subsample_n': TRAIN_SUBSAMPLE_N,
    'max_steps': MAX_STEPS,
    'num_epochs': NUM_EPOCHS,
    'batch_size': BATCH_SIZE,
    'gradient_accumulation': GRAD_ACCUM,
    'learning_rate': LEARNING_RATE,
    'warmup_ratio': WARMUP_RATIO,
    'max_seq_len': MAX_SEQ_LEN,
    'train_max_pixels': TRAIN_MAX_PIXELS,
    'subset_order_sha256': subset_hash,
    'subset_ids_in_order': subset_ids,
    'lora': {
        'rank': LORA_RANK,
        'alpha': LORA_ALPHA,
        'dropout': LORA_DROPOUT,
        'targets': LORA_TARGETS,
        'initialization_sha256': lora_init_hash,
    },
    'packages': package_versions(),
}
(seed_dir / 'manifest.json').write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

try:
    log_rows = train_one_seed(TRAINING_SEED, model, dtype, seed_dir)
    assert len(log_rows) == MAX_STEPS
    manifest['completed_optimizer_steps'] = len(log_rows)
    manifest['final_loss'] = log_rows[-1]['loss']
    manifest['final_running_loss'] = log_rows[-1]['running_loss']
    (seed_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    assert (adapter_dir / 'adapter_config.json').exists()
    assert any(path.exists() for path in adapter_weights)
    archive_path = seed_dir / f'adapter_q7b_n{TRAIN_SUBSAMPLE_N}_seed{TRAINING_SEED}.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(adapter_dir.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(seed_dir))
        archive.write(seed_dir / 'manifest.json', 'manifest.json')
        archive.write(seed_dir / 'training_log.csv', 'training_log.csv')
    print('Final adapter:', adapter_dir)
    print('Inference-ready archive:', archive_path)
finally:
    del model
    gc.collect()
    torch.cuda.empty_cache()
"""


def build_notebook(seed: int, label: str, size: int, steps: int):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"""
        # Qwen2.5-VL-7B QLoRA — {size:,} examples, seed {seed}

        Standalone Kaggle training notebook for the controlled three-seed scaling study.
        It trains exactly one adapter and performs no dev or test inference. Dataset
        membership is the fixed seed-{DATA_SUBSET_SEED} nested prefix shared by all runs;
        seed {seed} controls only training stochasticity.

        **Run contract:** one clean epoch, {steps} optimizer steps, frozen vision tower,
        answer-only supervised target, and an inference-ready adapter ZIP.
        """),
        md("## 1. Environment"),
        code(ENVIRONMENT_CELL),
        md("## 2. Fixed run configuration"),
        code(config_cell(seed, size, steps)),
        md("## 3. Deterministic RNG and audit helpers"),
        code(DETERMINISM_CELL),
        md("## 4. Fixed nested training subset and images"),
        code(DOWNLOAD_CELL),
        md("## 5. Dataset, answer-first target, and processor"),
        code(DATASET_CELL),
        md("## 6. Fresh quantized model and LoRA adapter"),
        code(MODEL_CELL),
        md("## 7. One-epoch deterministic training loop"),
        code(TRAINING_CELL),
        md("## 8. Train, validate, and package this adapter"),
        code(RUN_CELL),
        md("""
        ## Using the result

        Download the generated `adapter_q7b_n<size>_seed<seed>.zip` from
        `/kaggle/working/`. Attach it to the later inference notebook as a Kaggle dataset.
        Keep `manifest.json` with the adapter: it records the exact subset hash, package
        versions, initialization hash, and training configuration needed by the paper.
        """),
    ]
    nb.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10'},
        'accelerator': 'GPU',
        'kaggle': {'accelerator': 'gpu'},
        'experiment': {
            'training_seed': seed,
            'data_subset_seed': DATA_SUBSET_SEED,
            'train_subsample_n': size,
            'max_steps': steps,
        },
    }
    return nb


def main():
    written = []
    for label, (size, steps) in SIZES.items():
        folder = HERE / label
        folder.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            path = folder / f'train_q7b_qlora_n{size}_seed{seed}.ipynb'
            nbf.write(build_notebook(seed, label, size, steps), path)
            written.append(path)
            print('Wrote', path.relative_to(REPO_ROOT))
    assert len(written) == 12


if __name__ == '__main__':
    main()
