# HAR Dataset Audit Report

Data root: /root/dm-assignment3/nycu-data-mining-assignment-3
Train dir: /root/dm-assignment3/nycu-data-mining-assignment-3/train/train
Test dir:  /root/dm-assignment3/nycu-data-mining-assignment-3/test/test
Submission: /root/dm-assignment3/nycu-data-mining-assignment-3/sample_submission.csv

## 1. COUNTS
- Train files: 11020
- Test files:  6849
- Train unique users: 60
- Test unique users:  40
- sample_submission.csv rows (excl header): 6849
- Test files == submission rows == 6849 ? True  (test=6849, sub=6849)
- Submission Ids set == test file_id set ? True  (only in sub: 0, only in files: 0)

## 2. SEQUENCE LENGTH
### TRAIN
- rows/file: min=300, max=300, mean=300.00, median=300.0
- length histogram (length: #files):
    300: 11020
- all exactly 300 ? True
- files where index is NOT contiguous 0..N-1: 0

### TEST
- rows/file: min=300, max=300, mean=300.00, median=300.0
- length histogram (length: #files):
    300: 6849
- all exactly 300 ? True
- files where index is NOT contiguous 0..N-1: 0

## 3. CLASS BALANCE (train, FILES per label)
- label 0: 4643 files (42.13%)
- label 1: 4695 files (42.60%)
- label 2: 358 files (3.25%)
- label 3: 656 files (5.95%)
- label 4: 142 files (1.29%)
- label 5: 526 files (4.77%)
- total labeled files: 11020
- imbalance ratio (max/min): 33.06

## 4. USER vs LABEL (train)
- users total: 60
- distinct labels per user: min=2, max=6, mean=4.75, median=5.0
- users performing all 6 activities: 6 / 60
- distribution of labels-per-user (#labels: #users):
    2 labels: 1 users
    3 labels: 1 users
    4 labels: 16 users
    5 labels: 36 users
    6 labels: 6 users
- #users having >=1 file of each label:
    label 0: 60 users
    label 1: 60 users
    label 2: 52 users
    label 3: 59 users
    label 4: 19 users
    label 5: 35 users
- Full user x label table (files):
label       0    1   2   3   4   5
user                              
User_001   90   67   3  17   0  19
User_002   66   42   4  10   0  30
User_003   24  122   5   6   0  14
User_004   73   74  13  26   3  18
User_005   70  126   2   3   0   0
User_006   98   65   2   4   0  11
User_007   81   61   7   6   0   5
User_008   55   90  15  12   0   0
User_009   78   81   3  12   0  10
User_010   44   72   4   1   0   6
User_011  103   61  16   7   5   8
User_012   59   56  11  15   0  50
User_013  137   93   7  15   0   4
User_014   81  106  21  15   3   0
User_015   66  101   1  13   0   0
User_016   89  126   5  11   0   0
User_017   83   71   1   7   3   0
User_018   98   89   1  22  21   0
User_019   87   84  12  14   0  15
User_020   79   57   0  14   0  30
User_021  116   43   8   8   0   8
User_022   93   61   0   8   0   9
User_023   35   58  10  14   0   0
User_024   57  107   6  13   0   2
User_025   85  109   0   0   0   0
User_026   71   46  19  13   4   7
User_027   36  133   8   7   5   0
User_028   51   64   2   9   0  13
User_029   54  118  18  16   0   0
User_030   54    4   3  11   0  87
User_031   63  131   7  13   0  32
User_032  127   90   6   9   3   2
User_033   83   70   0   8   4   4
User_034   53   85   9  28   0  26
User_035   83   67  17  17   0   7
User_036   78   66   2   8   0   0
User_037  105   57   3  13   3   0
User_038   85   93   8   9   1   0
User_039   71  103  10   7   3   3
User_040   72   49   1  10   0  13
User_041  111   25   3   7   0   0
User_042  115  101   0  18   0  13
User_043  142   13   0  12   0   1
User_044   90   48   0   8   0   0
User_045   80   51   5  13  21   0
User_046   51   86   4   5   0   0
User_047   65   68  21   4  10   0
User_048   89   82   6  19   0   0
User_049   56   26   5  10   0  11
User_050   57   89   1  12  16   0
User_051   49  102   0   6  24   0
User_052  113   89  10   5   0   0
User_053   56   69  13   5   0   5
User_054   92   95   1  13   0   7
User_055   69   82   5  17   0  22
User_056   69   44   1   9   0  13
User_057   61   98   2  15   4   5
User_058  105   99   3  10   3   0
User_059   55  126   6   4   6   0
User_060   85  104   2  13   0  16

## 5. FEATURE STATS (train rows)
Overall (across all train rows):
            min     max     mean     std
mean_x -2.04313 1.41968 -0.14450 0.61921
mean_y -2.76188 4.13459  0.01143 0.46502
mean_z -1.71906 1.22565  0.19591 0.57527
std_x   0.00000 4.10899  0.05144 0.10677
std_y   0.00000 3.71988  0.04409 0.10092
std_z   0.00000 3.75380  0.04705 0.09616

Per-label MEAN of each feature (rows grouped by label):
        mean_x   mean_y  mean_z   std_x   std_y   std_z
label                                                  
0      0.00061  0.01225 0.18005 0.00911 0.00792 0.00906
1     -0.25195  0.03784 0.24042 0.05732 0.04396 0.05350
2     -0.24248 -0.03208 0.04091 0.10181 0.08888 0.09539
3     -0.24988 -0.16667 0.00696 0.18019 0.17232 0.15520
4     -0.02117 -0.20776 0.46819 0.29979 0.36283 0.26958
5     -0.30159  0.07947 0.20625 0.11065 0.08805 0.09698

Per-label STD (spread) of each feature, for context:
       mean_x  mean_y  mean_z   std_x   std_y   std_z
label                                                
0     0.58781 0.44810 0.64328 0.03936 0.03349 0.03342
1     0.62820 0.43046 0.53016 0.09836 0.08277 0.09385
2     0.61852 0.58151 0.45349 0.13976 0.12904 0.12990
3     0.57757 0.63904 0.43483 0.17166 0.16308 0.14077
4     0.54694 0.49508 0.42667 0.22748 0.29432 0.20682
5     0.60449 0.48347 0.49600 0.12689 0.10804 0.11105

## 6. DATA QUALITY
- Train feature NaN: 0, inf: 0
- Test  feature NaN: 0, inf: 0
- Train label NaN: 0
- Train 0-row files: 0 []
- Test  0-row files: 0 []
- distinct label values: [0, 1, 2, 3, 4, 5]; outside 0-5: []
- train files with >1 distinct label inside: 0 []
- duplicate file_ids within train: 0 unique ids dup'd []
- duplicate file_ids within test:  0 unique ids dup'd []
- file_ids appearing in BOTH train and test: 0 []
- train filename vs file_id-column numeric mismatches: 0 (note: 9999 differ only by zero-padding, e.g. file '00001' has file_id=1)
- test  filename vs file_id-column numeric mismatches: 0

## 7. TRAIN vs TEST DRIFT
Per-feature distribution comparison (train vs test, over all rows):
feature      train_mean    test_mean      Δmean    train_std     test_std       Δstd
mean_x         -0.14450     -0.14458   -0.00008      0.61921      0.62433    0.00512
mean_y          0.01143      0.00654   -0.00490      0.46502      0.48435    0.01933
mean_z          0.19591      0.18030   -0.01560      0.57527      0.55868   -0.01659
std_x           0.05144      0.04803   -0.00341      0.10677      0.09754   -0.00923
std_y           0.04409      0.04230   -0.00179      0.10092      0.09452   -0.00640
std_z           0.04705      0.04501   -0.00204      0.09616      0.09087   -0.00528

Standardized mean shift |test_mean - train_mean| / train_std (flag if > 0.10):
  mean_x     0.0001
  mean_y     0.0105
  mean_z     0.0271
  std_x      0.0319
  std_y      0.0177
  std_z      0.0212

- train users: 60, test users: 40
- users in BOTH train and test: 0 []
- test users disjoint from train ? True

## 8. NAIVE SEPARABILITY (LogisticRegression, GroupKFold-5 by user)
- design matrix: X=(11020, 12), y=(11020,), groups=60 users, 12 features
  fold 0: macro-F1 = 0.5160  (val users=12, val labels present=[0, 1, 2, 3, 4, 5])
  fold 1: macro-F1 = 0.5610  (val users=12, val labels present=[0, 1, 2, 3, 4, 5])
  fold 2: macro-F1 = 0.5577  (val users=12, val labels present=[0, 1, 2, 3, 4, 5])
  fold 3: macro-F1 = 0.5776  (val users=12, val labels present=[0, 1, 2, 3, 4, 5])
  fold 4: macro-F1 = 0.5000  (val users=12, val labels present=[0, 1, 2, 3, 4, 5])
- mean fold macro-F1: 0.5425 (+/- 0.0294)
- OOF macro-F1 (pooled): 0.5476
- per-class F1 (pooled OOF):
    label 0: F1 = 0.8929
    label 1: F1 = 0.7968
    label 2: F1 = 0.0162
    label 3: F1 = 0.6582
    label 4: F1 = 0.7229
    label 5: F1 = 0.1985

## ANOMALIES
- 54/60 users do NOT cover all 6 labels (min 2) -> GroupKFold folds may miss classes.
