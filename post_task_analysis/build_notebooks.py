"""Build the four post-shared-task analysis notebooks.

The generated notebooks are committed artifacts; rerun this builder after editing their
shared scaffolding so path handling and validation remain consistent.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


OUT = Path(__file__).resolve().parent


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


HEADER = """
> **Post-shared-task analysis.** Gold test labels were public when this analysis was
> designed. Nothing in this notebook changes the official CI=0.035 submission or the
> third-place ranking. Results are retrospective and must not be described as untouched
> test-set estimates.
"""


COMMON = r'''
from pathlib import Path
from collections import Counter
import csv, io, json, os, urllib.request, zipfile

import numpy as np
import pandas as pd


def find_repo_root():
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    for candidate in candidates:
        if (candidate / 'README.md').exists() and (candidate / 'Test').exists():
            return candidate.resolve()
    raise FileNotFoundError('Run this notebook from the IE2026-HalDetect repository.')


ROOT = find_repo_root()
HERE = ROOT / 'post_task_analysis'
CACHE = HERE / 'cache'
OUTPUT = HERE / 'outputs'
CACHE.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

GOLD_URL = (
    'https://huggingface.co/datasets/QCRI/ImageEval-ArabicNLP26/'
    'resolve/main/task1b/test_en.jsonl'
)
gold_override = os.getenv('IMAGEEVAL_TEST_GOLD')
GOLD_PATH = Path(gold_override) if gold_override else CACHE / 'test_en.jsonl'
if not GOLD_PATH.exists():
    print('Downloading released gold test JSONL...')
    urllib.request.urlretrieve(GOLD_URL, GOLD_PATH)


def read_gold(path=GOLD_PATH):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 1000, f'Expected 1,000 gold items, found {len(rows)}'
    frame = pd.DataFrame(rows)
    assert frame['id'].is_unique
    assert frame['labels'].map(lambda x: len(x) == 3 and sum(x) == 1).all()
    frame['gold_idx'] = frame['labels'].map(lambda x: x.index(True))
    return frame


def load_prediction_zip(path, gold):
    path = Path(path)
    assert path.exists(), path
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith('.csv')]
        assert len(csv_names) == 1, (path, csv_names)
        with io.TextIOWrapper(archive.open(csv_names[0]), encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
    raw = pd.DataFrame(rows)
    required = {'id', 'statement_index', 'prediction'}
    assert required.issubset(raw.columns), (path, raw.columns)
    raw['statement_index'] = raw['statement_index'].astype(int)
    raw['pred_bool'] = raw['prediction'].str.strip().str.lower().map(
        {'true': True, 'false': False})
    assert raw['pred_bool'].notna().all(), f'Unparseable prediction in {path}'
    assert not raw.duplicated(['id', 'statement_index']).any()
    assert set(raw['statement_index']) == {0, 1, 2}
    grouped = raw.sort_values(['id', 'statement_index']).groupby('id', sort=False)
    vectors = grouped['pred_bool'].apply(list)
    assert vectors.map(len).eq(3).all()
    assert set(vectors.index) == set(gold['id']), f'ID mismatch in {path}'

    result = gold[['id', 'gold_idx']].copy()
    by_id = vectors.to_dict()
    result['pred_vector'] = result['id'].map(by_id)
    result['format_valid'] = result['pred_vector'].map(lambda x: sum(x) == 1)
    result['pred_idx'] = result['pred_vector'].map(
        lambda x: x.index(True) if sum(x) == 1 else np.nan)
    result['correct'] = result['format_valid'] & result['pred_idx'].eq(result['gold_idx'])
    result['error'] = ~result['correct']
    return result


SUBMISSIONS = {
    'Elimination': ROOT / 'Test/qwen2p5-3b-7b/All COT variations/cot-elimination/prediction_en.zip',
    'Socratic': ROOT / 'Test/qwen2p5-3b-7b/All COT variations/cot-socratic/prediction_en.zip',
    'Devils advocate': ROOT / 'Test/qwen2p5-3b-7b/All COT variations/cot-devils-advocate/prediction_en.zip',
    'Evidence first': ROOT / 'Test/qwen2p5-3b-7b/All COT variations/cot-evidence-first/prediction_en.zip',
    'Attribute checklist': ROOT / 'Test/qwen2p5-3b-7b/All COT variations/cot-attribute-checklist/prediction_en.zip',
    'Confidence ranked': ROOT / 'Test/qwen2p5-3b-7b/All COT variations/cot-confidence-ranked/prediction_en.zip',
    'QLoRA 2,000': ROOT / 'Test/qwen2p5-3b-7b/qlora-q7b-2k-image/prediction_en.zip',
    'QLoRA 2,348': ROOT / 'Test/qwen2p5-3b-7b/qlora-q7b-2p3k-image/prediction_en.zip',
    'QLoRA 2,600': ROOT / 'Test/qwen2p5-3b-7b/qlora-q7b-2p6k-image/prediction_en.zip',
    'QLoRA 3,000 legacy': ROOT / 'Test/qwen2p5-3b-7b/qlora-q7b-3k-image/prediction_en.zip',
}

gold = read_gold()
predictions = {name: load_prediction_zip(path, gold) for name, path in SUBMISSIONS.items()}
summary = pd.DataFrame([
    {
        'system': name,
        'n': len(frame),
        'errors': int(frame['error'].sum()),
        'CI': frame['error'].mean(),
        'accuracy': frame['correct'].mean(),
        'format_failures': int((~frame['format_valid']).sum()),
    }
    for name, frame in predictions.items()
]).sort_values(['CI', 'system']).reset_index(drop=True)
display(summary)
'''


def notebook(title: str, cells):
    nb = nbf.v4.new_notebook()
    nb['cells'] = [md(f'# {title}\n\n{HEADER}'), *cells]
    nb['metadata'] = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3'},
    }
    return nb


nb1 = notebook(
    '01 — Paired McNemar Tests, Error Overlap, and Error Migration',
    [
        md('''
        This notebook compares all ten eligible Task 1b English test submissions. It
        computes exact paired McNemar tests with Holm correction, pairwise error-set
        overlap/Jaccard, and row-level error migration for predeclared comparisons.
        '''),
        code(COMMON),
        md('## Exact paired McNemar tests'),
        code(r'''
        from itertools import combinations
        from scipy.stats import binomtest


        def holm_adjust(p_values):
            p_values = np.asarray(p_values, dtype=float)
            order = np.argsort(p_values)
            adjusted = np.empty_like(p_values)
            running = 0.0
            m = len(p_values)
            for rank, index in enumerate(order):
                running = max(running, (m - rank) * p_values[index])
                adjusted[index] = min(1.0, running)
            return adjusted


        rows = []
        for system_a, system_b in combinations(SUBMISSIONS, 2):
            a = predictions[system_a].set_index('id').loc[gold['id']]
            b = predictions[system_b].set_index('id').loc[gold['id']]
            a_correct = a['correct'].to_numpy()
            b_correct = b['correct'].to_numpy()
            a_only = int((a_correct & ~b_correct).sum())
            b_only = int((~a_correct & b_correct).sum())
            discordant = a_only + b_only
            p_exact = binomtest(a_only, discordant, 0.5).pvalue if discordant else 1.0
            rows.append({
                'system_a': system_a,
                'system_b': system_b,
                'CI_a': (~a_correct).mean(),
                'CI_b': (~b_correct).mean(),
                'delta_CI_a_minus_b': (~a_correct).mean() - (~b_correct).mean(),
                'a_only_correct': a_only,
                'b_only_correct': b_only,
                'discordant': discordant,
                'mcnemar_exact_p': p_exact,
            })
        mcnemar = pd.DataFrame(rows)
        mcnemar['holm_p'] = holm_adjust(mcnemar['mcnemar_exact_p'])
        mcnemar['holm_significant_0.05'] = mcnemar['holm_p'] < 0.05
        mcnemar.to_csv(OUTPUT / 'pairwise_mcnemar_exact.csv', index=False)
        display(mcnemar.sort_values(['holm_p', 'mcnemar_exact_p']).head(20))
        '''),
        md('## Error overlap and Jaccard'),
        code(r'''
        error_sets = {
            name: set(frame.loc[frame['error'], 'id']) for name, frame in predictions.items()
        }
        overlap_rows = []
        for system_a, system_b in combinations(SUBMISSIONS, 2):
            a, b = error_sets[system_a], error_sets[system_b]
            overlap_rows.append({
                'system_a': system_a,
                'system_b': system_b,
                'errors_a': len(a),
                'errors_b': len(b),
                'intersection': len(a & b),
                'union': len(a | b),
                'jaccard': len(a & b) / len(a | b) if a | b else 1.0,
                'a_only': len(a - b),
                'b_only': len(b - a),
            })
        overlap = pd.DataFrame(overlap_rows)
        overlap.to_csv(OUTPUT / 'pairwise_error_overlap.csv', index=False)
        display(overlap.sort_values('jaccard').head(20))

        import matplotlib.pyplot as plt
        import seaborn as sns
        names = list(SUBMISSIONS)
        matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
        for row in overlap.itertuples():
            matrix.loc[row.system_a, row.system_b] = row.jaccard
            matrix.loc[row.system_b, row.system_a] = row.jaccard
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=1)
        plt.title('Jaccard similarity of error sets')
        plt.tight_layout()
        plt.savefig(OUTPUT / 'error_jaccard_heatmap.png', dpi=180)
        plt.show()
        '''),
        md('## Error migration'),
        code(r'''
        MIGRATION_PAIRS = [
            ('Attribute checklist', 'QLoRA 2,600'),
            ('QLoRA 2,348', 'QLoRA 2,600'),
            ('Elimination', 'QLoRA 2,600'),
        ]
        migration_frames = []
        for system_a, system_b in MIGRATION_PAIRS:
            a = predictions[system_a][['id', 'pred_idx', 'correct']].rename(
                columns={'pred_idx': 'pred_a', 'correct': 'correct_a'})
            b = predictions[system_b][['id', 'pred_idx', 'correct']].rename(
                columns={'pred_idx': 'pred_b', 'correct': 'correct_b'})
            merged = gold.merge(a, on='id').merge(b, on='id')
            conditions = [
                merged['correct_a'] & merged['correct_b'],
                merged['correct_a'] & ~merged['correct_b'],
                ~merged['correct_a'] & merged['correct_b'],
            ]
            labels = ['both_correct', 'a_only_correct', 'b_only_correct']
            merged['migration'] = np.select(conditions, labels, default='both_wrong')
            merged['system_a'] = system_a
            merged['system_b'] = system_b
            migration_frames.append(merged)
        migration = pd.concat(migration_frames, ignore_index=True)
        migration['statements_json'] = migration['statements'].map(json.dumps)
        keep = [
            'system_a', 'system_b', 'id', 'country', 'category', 'subcategory',
            'gold_idx', 'pred_a', 'pred_b', 'correct_a', 'correct_b', 'migration',
            'image', 'statements_json',
        ]
        migration[keep].to_csv(OUTPUT / 'error_migration_items.csv', index=False)
        migration_summary = migration.groupby(
            ['system_a', 'system_b', 'migration']).size().rename('n').reset_index()
        migration_summary.to_csv(OUTPUT / 'error_migration_summary.csv', index=False)
        display(migration_summary)
        '''),
    ],
)


nb2 = notebook(
    '02 — QLoRA Majority Ensembles and Oracle Upper Bound',
    [
        md('''
        This notebook evaluates fixed majority-vote ensembles of the four submitted QLoRA
        adapters. Three-model ensembles use ordinary majority vote. The four-model
        ensemble uses the devtest-selected 2,348 adapter for deterministic 2--2 ties.
        The oracle is not deployable; it measures complementarity only.
        '''),
        code(COMMON),
        code(r'''
        from itertools import combinations

        QLORA = ['QLoRA 2,000', 'QLoRA 2,348', 'QLoRA 2,600', 'QLoRA 3,000 legacy']
        indexed = {
            name: predictions[name].set_index('id').loc[gold['id']] for name in QLORA
        }
        pred_matrix = np.stack([indexed[name]['pred_idx'].to_numpy(dtype=int) for name in QLORA])
        gold_idx = gold['gold_idx'].to_numpy(dtype=int)


        def majority_prediction(member_names, tie_break='QLoRA 2,348'):
            member_rows = [QLORA.index(name) for name in member_names]
            votes = pred_matrix[member_rows]
            tie_row = QLORA.index(tie_break)
            out = []
            for column in votes.T:
                counts = np.bincount(column, minlength=3)
                winners = np.flatnonzero(counts == counts.max())
                out.append(int(winners[0]) if len(winners) == 1 else int(pred_matrix[tie_row, len(out)]))
            return np.asarray(out)


        ensemble_rows, ensemble_predictions = [], {}
        fixed_memberships = [list(x) for x in combinations(QLORA, 3)] + [QLORA]
        for members in fixed_memberships:
            name = ' + '.join(members)
            pred = majority_prediction(members)
            ensemble_predictions[name] = pred
            ensemble_rows.append({
                'ensemble': name,
                'n_members': len(members),
                'tie_breaker': 'QLoRA 2,348' if len(members) % 2 == 0 else 'not needed',
                'errors': int((pred != gold_idx).sum()),
                'CI': (pred != gold_idx).mean(),
                'accuracy': (pred == gold_idx).mean(),
            })

        individual_correct = pred_matrix == gold_idx
        oracle_correct = individual_correct.any(axis=0)
        ensemble_rows.append({
            'ensemble': 'ORACLE: any QLoRA adapter correct',
            'n_members': len(QLORA),
            'tie_breaker': 'not deployable',
            'errors': int((~oracle_correct).sum()),
            'CI': (~oracle_correct).mean(),
            'accuracy': oracle_correct.mean(),
        })
        ensemble_summary = pd.DataFrame(ensemble_rows).sort_values('CI')
        ensemble_summary.to_csv(OUTPUT / 'qlora_ensemble_summary.csv', index=False)
        display(ensemble_summary)

        output = gold[['id', 'gold_idx']].copy()
        for name, pred in ensemble_predictions.items():
            output[name] = pred
        output.to_csv(OUTPUT / 'qlora_ensemble_item_predictions.csv', index=False)
        '''),
        md('## Adapter agreement and complementarity'),
        code(r'''
        agreement = pd.DataFrame(index=QLORA, columns=QLORA, dtype=float)
        for a in QLORA:
            for b in QLORA:
                agreement.loc[a, b] = (
                    indexed[a]['pred_idx'].to_numpy() == indexed[b]['pred_idx'].to_numpy()
                ).mean()
        agreement.to_csv(OUTPUT / 'qlora_prediction_agreement.csv')
        display(agreement.round(3))

        error_pattern = pd.DataFrame({'id': gold['id']})
        for name in QLORA:
            error_pattern[name] = indexed[name]['error'].to_numpy()
        error_pattern['n_adapters_wrong'] = error_pattern[QLORA].sum(axis=1)
        error_pattern.to_csv(OUTPUT / 'qlora_error_patterns.csv', index=False)
        display(error_pattern['n_adapters_wrong'].value_counts().sort_index().rename('items'))
        '''),
    ],
)


nb3 = notebook(
    '03 — Country, Category, and Position Analysis',
    [
        md('''
        This notebook analyzes the best official submission (QLoRA 2,600) by released
        metadata and true-statement position. Wilson intervals are descriptive; overlapping
        groups and small denominators should not be interpreted as country effects.
        '''),
        code(COMMON),
        code(r'''
        import math

        SYSTEM = 'QLoRA 2,600'
        evaluated = gold.merge(
            predictions[SYSTEM][['id', 'pred_idx', 'correct', 'error']], on='id')


        def wilson_interval(errors, total, z=1.959963984540054):
            if total == 0:
                return np.nan, np.nan
            p = errors / total
            denominator = 1 + z*z/total
            center = (p + z*z/(2*total)) / denominator
            margin = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
            return center - margin, center + margin


        def grouped_errors(field):
            rows = []
            for value, frame in evaluated.groupby(field, dropna=False):
                n = len(frame)
                errors = int(frame['error'].sum())
                low, high = wilson_interval(errors, n)
                rows.append({
                    'field': field, 'group': value, 'n': n, 'errors': errors,
                    'error_rate': errors/n, 'wilson_95_low': low, 'wilson_95_high': high,
                })
            return pd.DataFrame(rows).sort_values(['error_rate', 'errors'], ascending=False)


        subgroup = pd.concat([
            grouped_errors('country'), grouped_errors('category'), grouped_errors('subcategory')
        ], ignore_index=True)
        subgroup.to_csv(OUTPUT / 'subgroup_error_intervals.csv', index=False)
        display(subgroup.query("field == 'country'").head(15))
        display(subgroup.query("field == 'category'"))
        display(subgroup.query("field == 'subcategory'").head(15))
        '''),
        md('## Position-specific error and confusion'),
        code(r'''
        position_rows = []
        for position, frame in evaluated.groupby('gold_idx'):
            n = len(frame)
            errors = int(frame['error'].sum())
            low, high = wilson_interval(errors, n)
            position_rows.append({
                'gold_position_1based': int(position + 1),
                'n': n, 'errors': errors, 'error_rate': errors/n,
                'wilson_95_low': low, 'wilson_95_high': high,
            })
        position_rates = pd.DataFrame(position_rows)
        position_rates.to_csv(OUTPUT / 'position_error_intervals.csv', index=False)
        display(position_rates)

        confusion = pd.crosstab(
            evaluated['gold_idx'] + 1,
            evaluated['pred_idx'] + 1,
            rownames=['gold position'], colnames=['predicted position'], dropna=False)
        confusion.to_csv(OUTPUT / 'position_confusion_matrix.csv')
        display(confusion)

        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(4.8, 4))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{SYSTEM}: true-position confusion')
        plt.tight_layout()
        plt.savefig(OUTPUT / 'position_confusion_heatmap.png', dpi=180)
        plt.show()
        '''),
        md('## Export the 35 errors for qualitative inspection'),
        code(r'''
        errors = evaluated.loc[evaluated['error']].copy()
        errors['statements_json'] = errors['statements'].map(json.dumps)
        errors.drop(columns=['labels', 'statements']).to_csv(
            OUTPUT / 'qlora_2600_test_errors.csv', index=False)
        print(f'Exported {len(errors)} errors; expected 35.')
        assert len(errors) == 35
        '''),
    ],
)


nb4 = notebook(
    '04 — Blinded Error-Taxonomy Annotation and Cohen’s Kappa',
    [
        md('''
        This notebook prepares two independent, blinded annotation sheets for the 35
        errors of QLoRA 2,600 and computes raw agreement and Cohen's kappa after both
        raters finish. Do not expose either completed sheet to the other rater before
        annotation. A full 35-item second pass is preferred; set `IAA_N=15` only if
        resources require a predeclared subsample.

        **Labels**

        - `function_intent_event`: same main entities, different purpose/function/activity.
        - `visual_material`: directly observable colour/texture/material/shape/script cue.
        - `recognition`: different object, place, food, instrument, or scene identity.
        '''),
        code(COMMON),
        md('## Create blinded sheets (safe: existing sheets are never overwritten)'),
        code(r'''
        ANNOTATION_DIR = HERE / 'annotations'
        ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
        IAA_N = 35
        IAA_SEED = 2026
        SYSTEM = 'QLoRA 2,600'

        evaluated = gold.merge(
            predictions[SYSTEM][['id', 'pred_idx', 'correct', 'error']], on='id')
        errors = evaluated.loc[evaluated['error']].copy()
        assert len(errors) == 35
        if IAA_N < len(errors):
            selected_ids = errors.sample(IAA_N, random_state=IAA_SEED)['id']
            errors = errors[errors['id'].isin(selected_ids)].copy()

        errors['statement_1'] = errors['statements'].str[0]
        errors['statement_2'] = errors['statements'].str[1]
        errors['statement_3'] = errors['statements'].str[2]
        errors['image_url'] = errors['image'].map(
            lambda path: 'https://huggingface.co/datasets/QCRI/ImageEval-ArabicNLP26/resolve/main/' + path)
        columns = [
            'id', 'image_url', 'country', 'category', 'subcategory',
            'statement_1', 'statement_2', 'statement_3',
        ]
        for rater, seed in [('rater1', IAA_SEED + 1), ('rater2', IAA_SEED + 2)]:
            path = ANNOTATION_DIR / f'{rater}.csv'
            if path.exists():
                print('Preserving existing sheet:', path)
                continue
            sheet = errors[columns].sample(frac=1, random_state=seed).copy()
            sheet['label'] = ''
            sheet['notes'] = ''
            sheet.to_csv(path, index=False)
            print('Created:', path)
        '''),
        md('## Optional item viewer'),
        code(r'''
        from IPython.display import Image, Markdown, display


        def show_annotation_item(rater='rater2', row_number=0):
            sheet = pd.read_csv(ANNOTATION_DIR / f'{rater}.csv', keep_default_na=False)
            row = sheet.iloc[row_number]
            display(Markdown(
                f"**{row_number + 1}/{len(sheet)} — {row['country']} | "
                f"{row['category']} | {row['subcategory']}**"
            ))
            display(Image(url=row['image_url'], width=600))
            for index in range(1, 4):
                display(Markdown(f"**Statement {index}:** {row[f'statement_{index}']}"))


        show_annotation_item('rater2', 0)
        '''),
        md('## Agreement and Cohen’s kappa'),
        code(r'''
        LABELS = {'function_intent_event', 'visual_material', 'recognition'}


        def load_completed_rater(name):
            path = ANNOTATION_DIR / f'{name}.csv'
            frame = pd.read_csv(path, keep_default_na=False)
            frame['label'] = frame['label'].str.strip()
            blanks = frame['label'].eq('').sum()
            unknown = sorted(set(frame['label']) - LABELS - {''})
            if blanks or unknown:
                raise ValueError(f'{name}: blanks={blanks}, unknown={unknown}')
            assert frame['id'].is_unique
            return frame[['id', 'label']].rename(columns={'label': name})


        try:
            rater1 = load_completed_rater('rater1')
            rater2 = load_completed_rater('rater2')
        except ValueError as exc:
            print('Complete both sheets, then rerun this cell:', exc)
        else:
            paired = rater1.merge(rater2, on='id', validate='one_to_one')
            assert len(paired) == IAA_N
            observed = (paired['rater1'] == paired['rater2']).mean()
            p1 = paired['rater1'].value_counts(normalize=True)
            p2 = paired['rater2'].value_counts(normalize=True)
            expected = sum(p1.get(label, 0) * p2.get(label, 0) for label in LABELS)
            kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
            summary = {
                'n_double_annotated': len(paired),
                'raw_agreement': observed,
                'cohen_kappa': kappa,
                'n_disagreements': int((paired['rater1'] != paired['rater2']).sum()),
            }
            print(json.dumps(summary, indent=2))
            confusion = pd.crosstab(paired['rater1'], paired['rater2'])
            display(confusion)
            disagreements = paired.loc[paired['rater1'] != paired['rater2']]
            disagreements.to_csv(ANNOTATION_DIR / 'disagreements_for_adjudication.csv', index=False)
            (ANNOTATION_DIR / 'iaa_summary.json').write_text(
                json.dumps(summary, indent=2), encoding='utf-8')
        '''),
        md('''
        ## Reporting template

        “A second annotator independently labelled **N** errors using a pre-specified
        three-class rubric. Raw agreement was **X** and Cohen's $\\kappa$ was **Y**.
        Disagreements were [adjudicated by discussion / retained without adjudication].”
        '''),
    ],
)


for filename, nb in {
    '01_paired_comparisons.ipynb': nb1,
    '02_qlora_ensembles.ipynb': nb2,
    '03_subgroup_position_analysis.ipynb': nb3,
    '04_error_taxonomy_iaa.ipynb': nb4,
}.items():
    nbf.write(nb, OUT / filename)
    print('Wrote', OUT / filename)
