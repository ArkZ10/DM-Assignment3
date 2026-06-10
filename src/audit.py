#!/usr/bin/env python3
"""Factual audit of the HAR Kaggle dataset. No modeling beyond a naive LR floor."""
import os, glob, io, sys
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

BASE = "/root/dm-assignment3/nycu-data-mining-assignment-3"
TRAIN_DIR = os.path.join(BASE, "train", "train")
TEST_DIR = os.path.join(BASE, "test", "test")
SUB = os.path.join(BASE, "sample_submission.csv")
FEATS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]

buf = io.StringIO()
def out(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    buf.write(s + "\n")

def list_files(d):
    return sorted(glob.glob(os.path.join(d, "User_*", "*.csv")))

def user_of(path):
    return os.path.basename(os.path.dirname(path))

out("# HAR Dataset Audit Report")
out("")
out("Data root:", BASE)
out("Train dir:", TRAIN_DIR)
out("Test dir: ", TEST_DIR)
out("Submission:", SUB)
out("")

train_files = list_files(TRAIN_DIR)
test_files = list_files(TEST_DIR)

# ---- Load everything once ----
def load_all(files, has_label):
    frames, meta = [], []
    for fp in files:
        df = pd.read_csv(fp)
        fid = os.path.splitext(os.path.basename(fp))[0]
        u = user_of(fp)
        nrows = len(df)
        idx = df["index"].to_numpy() if "index" in df.columns else np.array([])
        contiguous = (nrows > 0 and np.array_equal(idx, np.arange(nrows)))
        lab = None
        if has_label and "label" in df.columns and nrows > 0:
            lab = df["label"].iloc[0]
            multi = df["label"].nunique() > 1
        else:
            multi = False
        meta.append(dict(path=fp, file_id_name=fid,
                         file_id_col=(df["file_id"].iloc[0] if "file_id" in df.columns and nrows>0 else None),
                         user=u, nrows=nrows, contiguous=contiguous,
                         label=lab, multi_label=multi))
        df["_user"] = u
        df["_fname_id"] = fid
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    return big, pd.DataFrame(meta)

train_big, train_meta = load_all(train_files, True)
test_big, test_meta = load_all(test_files, False)

# ================= Q1 COUNTS =================
out("## 1. COUNTS")
out(f"- Train files: {len(train_files)}")
out(f"- Test files:  {len(test_files)}")
out(f"- Train unique users: {train_meta['user'].nunique()}")
out(f"- Test unique users:  {test_meta['user'].nunique()}")
sub = pd.read_csv(SUB)
out(f"- sample_submission.csv rows (excl header): {len(sub)}")
out(f"- Test files == submission rows == 6849 ? "
    f"{len(test_files) == len(sub) == 6849}  "
    f"(test={len(test_files)}, sub={len(sub)})")
# Do the submission Ids match the test file_ids?
sub_ids = set(sub['Id'].astype(str))
test_fid_names = set(test_meta['file_id_name'].astype(str))
out(f"- Submission Ids set == test file_id set ? {sub_ids == test_fid_names}"
    f"  (only in sub: {len(sub_ids - test_fid_names)}, only in files: {len(test_fid_names - sub_ids)})")
out("")

# ================= Q2 SEQUENCE LENGTH =================
def len_report(meta, name):
    L = meta["nrows"].to_numpy()
    out(f"### {name}")
    out(f"- rows/file: min={L.min()}, max={L.max()}, mean={L.mean():.2f}, median={np.median(L):.1f}")
    vc = pd.Series(L).value_counts().sort_index()
    out(f"- length histogram (length: #files):")
    for length, cnt in vc.items():
        out(f"    {length}: {cnt}")
    all300 = bool((L == 300).all())
    out(f"- all exactly 300 ? {all300}")
    noncontig = meta.loc[~meta["contiguous"], "file_id_name"].tolist()
    out(f"- files where index is NOT contiguous 0..N-1: {len(noncontig)}"
        + ("" if not noncontig else f" e.g. {noncontig[:5]}"))
    out("")

out("## 2. SEQUENCE LENGTH")
len_report(train_meta, "TRAIN")
len_report(test_meta, "TEST")

# ================= Q3 CLASS BALANCE (files per label) =================
out("## 3. CLASS BALANCE (train, FILES per label)")
lab_counts = train_meta["label"].value_counts().sort_index()
total = lab_counts.sum()
for lab, c in lab_counts.items():
    out(f"- label {int(lab)}: {c} files ({100*c/total:.2f}%)")
out(f"- total labeled files: {total}")
out(f"- imbalance ratio (max/min): {lab_counts.max()/lab_counts.min():.2f}")
out("")

# ================= Q4 USER vs LABEL =================
out("## 4. USER vs LABEL (train)")
ux = pd.crosstab(train_meta["user"], train_meta["label"])
# ensure all labels 0-5 present as cols
for l in range(6):
    if l not in ux.columns:
        ux[l] = 0
ux = ux[sorted(ux.columns)]
labels_per_user = (ux > 0).sum(axis=1)
out(f"- users total: {len(ux)}")
out(f"- distinct labels per user: min={labels_per_user.min()}, "
    f"max={labels_per_user.max()}, mean={labels_per_user.mean():.2f}, "
    f"median={labels_per_user.median():.1f}")
out(f"- users performing all 6 activities: {(labels_per_user==6).sum()} / {len(ux)}")
out(f"- distribution of labels-per-user (#labels: #users):")
for k, v in labels_per_user.value_counts().sort_index().items():
    out(f"    {k} labels: {v} users")
# per-label: how many users have at least one file of that label
out(f"- #users having >=1 file of each label:")
for l in sorted(ux.columns):
    out(f"    label {l}: {(ux[l]>0).sum()} users")
out("- Full user x label table (files):")
out(ux.to_string())
out("")

# ================= Q5 FEATURE STATS =================
out("## 5. FEATURE STATS (train rows)")
out("Overall (across all train rows):")
desc = train_big[FEATS].agg(["min", "max", "mean", "std"]).T
out(desc.to_string(float_format=lambda x: f"{x:.5f}"))
out("")
out("Per-label MEAN of each feature (rows grouped by label):")
per_lab = train_big.groupby("label")[FEATS].mean()
out(per_lab.to_string(float_format=lambda x: f"{x:.5f}"))
out("")
out("Per-label STD (spread) of each feature, for context:")
per_lab_std = train_big.groupby("label")[FEATS].std()
out(per_lab_std.to_string(float_format=lambda x: f"{x:.5f}"))
out("")

# ================= Q6 DATA QUALITY =================
out("## 6. DATA QUALITY")
# NaN / inf
def bad_counts(df, cols):
    nan = int(df[cols].isna().sum().sum())
    inf = int(np.isinf(df[cols].to_numpy(dtype=float)).sum())
    return nan, inf
tr_nan, tr_inf = bad_counts(train_big, FEATS)
te_nan, te_inf = bad_counts(test_big, FEATS)
out(f"- Train feature NaN: {tr_nan}, inf: {tr_inf}")
out(f"- Test  feature NaN: {te_nan}, inf: {te_inf}")
out(f"- Train label NaN: {int(train_big['label'].isna().sum())}")
# zero-row files
z_tr = train_meta.loc[train_meta['nrows']==0,'file_id_name'].tolist()
z_te = test_meta.loc[test_meta['nrows']==0,'file_id_name'].tolist()
out(f"- Train 0-row files: {len(z_tr)} {z_tr[:5]}")
out(f"- Test  0-row files: {len(z_te)} {z_te[:5]}")
# labels outside 0-5
labset = set(train_big['label'].dropna().unique().tolist())
out(f"- distinct label values: {sorted(labset)}; outside 0-5: {sorted(labset - set(range(6)))}")
# multi-label files
ml = train_meta.loc[train_meta['multi_label'], 'file_id_name'].tolist()
out(f"- train files with >1 distinct label inside: {len(ml)} {ml[:5]}")
# duplicate file_ids across users/folders
tr_ids = train_meta['file_id_name'].astype(str)
te_ids = test_meta['file_id_name'].astype(str)
dup_tr = tr_ids[tr_ids.duplicated(keep=False)].tolist()
dup_te = te_ids[te_ids.duplicated(keep=False)].tolist()
out(f"- duplicate file_ids within train: {len(set(dup_tr))} unique ids dup'd {sorted(set(dup_tr))[:10]}")
out(f"- duplicate file_ids within test:  {len(set(dup_te))} unique ids dup'd {sorted(set(dup_te))[:10]}")
overlap = set(tr_ids) & set(te_ids)
out(f"- file_ids appearing in BOTH train and test: {len(overlap)} {sorted(overlap)[:10]}")
# file_id column vs filename agreement (compare numerically; train filenames are zero-padded)
def as_int(s):
    return pd.to_numeric(s, errors="coerce").astype("Int64")
mismatch_tr = train_meta[as_int(train_meta['file_id_col']) != as_int(train_meta['file_id_name'])]
mismatch_te = test_meta[as_int(test_meta['file_id_col']) != as_int(test_meta['file_id_name'])]
pad_tr = (train_meta['file_id_name'].astype(str) != train_meta['file_id_col'].astype(str)).sum()
out(f"- train filename vs file_id-column numeric mismatches: {len(mismatch_tr)} "
    f"(note: {pad_tr} differ only by zero-padding, e.g. file '00001' has file_id=1)")
out(f"- test  filename vs file_id-column numeric mismatches: {len(mismatch_te)}")
out("")

# ================= Q7 TRAIN vs TEST DRIFT =================
out("## 7. TRAIN vs TEST DRIFT")
out("Per-feature distribution comparison (train vs test, over all rows):")
hdr = f"{'feature':10} {'train_mean':>12} {'test_mean':>12} {'Δmean':>10} {'train_std':>12} {'test_std':>12} {'Δstd':>10}"
out(hdr)
for c in FEATS:
    tm, ts = train_big[c].mean(), train_big[c].std()
    em, es = test_big[c].mean(), test_big[c].std()
    out(f"{c:10} {tm:12.5f} {em:12.5f} {em-tm:10.5f} {ts:12.5f} {es:12.5f} {es-ts:10.5f}")
out("")
# standardized mean shift flag
out("Standardized mean shift |test_mean - train_mean| / train_std (flag if > 0.10):")
for c in FEATS:
    tm, ts = train_big[c].mean(), train_big[c].std()
    em = test_big[c].mean()
    z = abs(em - tm)/ts if ts else float('nan')
    flag = "  <-- FLAG" if z > 0.10 else ""
    out(f"  {c:10} {z:.4f}{flag}")
out("")
tr_users = set(train_meta['user'])
te_users = set(test_meta['user'])
out(f"- train users: {len(tr_users)}, test users: {len(te_users)}")
out(f"- users in BOTH train and test: {len(tr_users & te_users)} {sorted(tr_users & te_users)[:10]}")
out(f"- test users disjoint from train ? {len(tr_users & te_users)==0}")
out("")

# ================= Q8 NAIVE SEPARABILITY =================
out("## 8. NAIVE SEPARABILITY (LogisticRegression, GroupKFold-5 by user)")
# collapse each train file to 12 features
def collapse(meta, big):
    rows = []
    g = big.groupby("_fname_id")
    for fid_name in meta["file_id_name"]:
        sub_df = g.get_group(str(fid_name)) if str(fid_name) in g.groups else big[big["_fname_id"]==fid_name]
        m = sub_df[FEATS].mean()
        s = sub_df[FEATS].std()
        vec = {f"{c}_m": m[c] for c in FEATS}
        vec.update({f"{c}_s": s[c] for c in FEATS})
        rows.append(vec)
    return pd.DataFrame(rows)

X = collapse(train_meta, train_big)
X = X.fillna(0.0)  # std of single-row file -> NaN; guard
y = train_meta["label"].astype(int).to_numpy()
groups = train_meta["user"].to_numpy()
out(f"- design matrix: X={X.shape}, y={y.shape}, groups={len(set(groups))} users, 12 features")

gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(y), dtype=int)
fold_f1 = []
for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000))
    clf.fit(X.iloc[tr], y[tr])
    pred = clf.predict(X.iloc[va])
    oof[va] = pred
    f = f1_score(y[va], pred, average="macro")
    fold_f1.append(f)
    out(f"  fold {fold}: macro-F1 = {f:.4f}  (val users={len(set(groups[va]))}, "
        f"val labels present={sorted(int(v) for v in set(y[va]))})")

overall_macro = f1_score(y, oof, average="macro")
per_class = f1_score(y, oof, average=None, labels=list(range(6)))
out(f"- mean fold macro-F1: {np.mean(fold_f1):.4f} (+/- {np.std(fold_f1):.4f})")
out(f"- OOF macro-F1 (pooled): {overall_macro:.4f}")
out("- per-class F1 (pooled OOF):")
for l in range(6):
    out(f"    label {l}: F1 = {per_class[l]:.4f}")
out("")

# ================= ANOMALIES =================
out("## ANOMALIES")
anoms = []
if not (len(test_files) == len(sub) == 6849):
    anoms.append(f"Test file count / submission rows != 6849 (test={len(test_files)}, sub={len(sub)}).")
if (train_meta['nrows'] != 300).any() or (test_meta['nrows'] != 300).any():
    nt = (train_meta['nrows']!=300).sum(); ne=(test_meta['nrows']!=300).sum()
    anoms.append(f"Variable sequence length: {nt} train + {ne} test files are NOT 300 rows.")
nc = (~train_meta['contiguous']).sum() + (~test_meta['contiguous']).sum()
if nc:
    anoms.append(f"{nc} files have a non-contiguous index column.")
if tr_nan or te_nan or tr_inf or te_inf:
    anoms.append(f"NaN/inf present (train NaN={tr_nan} inf={tr_inf}, test NaN={te_nan} inf={te_inf}).")
if z_tr or z_te:
    anoms.append(f"Zero-row files exist (train={len(z_tr)}, test={len(z_te)}).")
if sorted(labset - set(range(6))):
    anoms.append(f"Labels outside 0-5 found: {sorted(labset-set(range(6)))}.")
if ml:
    anoms.append(f"{len(ml)} train files contain >1 distinct label internally.")
if overlap:
    anoms.append(f"{len(overlap)} file_ids appear in BOTH train and test.")
if len(set(dup_tr)) or len(set(dup_te)):
    anoms.append(f"Duplicate file_ids within a split (train={len(set(dup_tr))}, test={len(set(dup_te))}).")
if len(tr_users & te_users):
    anoms.append(f"{len(tr_users & te_users)} users overlap between train and test.")
if (labels_per_user < 6).any():
    anoms.append(f"{(labels_per_user<6).sum()}/{len(ux)} users do NOT cover all 6 labels "
                 f"(min {labels_per_user.min()}) -> GroupKFold folds may miss classes.")
flagged = [c for c in FEATS if (abs(test_big[c].mean()-train_big[c].mean())/train_big[c].std() if train_big[c].std() else 0) > 0.10]
if flagged:
    anoms.append(f"Train/test drift flagged on features: {flagged}.")
if not anoms:
    out("- None detected; data matches the described schema.")
else:
    for a in anoms:
        out(f"- {a}")

# write report
with open("/root/dm-assignment3/audit_report.md", "w") as fh:
    fh.write(buf.getvalue())
print("\n[written to /root/dm-assignment3/audit_report.md]")
