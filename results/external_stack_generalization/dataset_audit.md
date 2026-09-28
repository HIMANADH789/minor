# External Dataset Integrity Audit (PHASE 2)

Source: canonical UCR/aeon zips from timeseriesclassification.com/aeon-toolkit (same canonical host as the repository's ECG5000). Local zips: `data/external/ucr_zips/<DS>.zip` (unzip-verified), extracted under `data/external/extracted/<DS>/`.

Download provenance note: the Kaggle mirror referenced in the spec (m4ur1c10/dataset) was audited via the Kaggle CLI file listing and found to be a ~7.5 MB FORECASTING archive (Birmingham Parking / LD2011_2014 electricity CSVs) with no UCR classification datasets. Per Phase 1C the canonical UCR/aeon source was used instead. The UCI CSV (yasserhessein/harunshimanto epileptic-seizure-recognition) was NOT used.

Validation split policy (PHASE 3): EpilepticSeizures uses its PROVIDED canonical val.ts (20 samples, 10/10). Haptics and Phoneme get a deterministic per-class stratified 15% of TRAIN (seed 42). Phoneme has 8 singleton classes which make sklearn's stratified split impossible; the documented per-class rule (round(0.15*n_c) members to val for classes with n_c >= 2, singletons always kept in train, no class emptied) is used. TEST is untouched in all cases.

## EpilepticSeizures (PRIMARY)

- source: timeseriesclassification.com/aeon-toolkit/EpilepticSeizures.zip
- exact paths: train=`data\external\extracted\EpilepticSeizures\EpilepticSeizures\EpilepticSeizures_TRAIN.ts`, test=`data\external\extracted\EpilepticSeizures\EpilepticSeizures\EpilepticSeizures_TEST.ts`, val=`data\external\extracted\EpilepticSeizures\EpilepticSeizures\val.ts`
- format: .ts (aeon/UCR), univariate=True, sequence length=178, classes=2
- train: N=80 dist={'0': 40, '1': 40} nan=0 inf=0 dupes=0
- val: N=20 dist={'0': 10, '1': 10} nan=0 inf=0 dupes=0
- test: N=11420 dist={'0': 2260, '1': 9160} nan=0 inf=0 dupes=0
- validation availability: provided_canonical_val_ts
- canonical split preserved: YES (train/test from source files)
- class entropy (train): 1.0000 bits

## Haptics (secondary)

- source: timeseriesclassification.com/aeon-toolkit/Haptics.zip
- exact paths: train=`data\external\extracted\Haptics\Haptics_TRAIN.ts`, test=`data\external\extracted\Haptics\Haptics_TEST.ts`
- format: .ts (aeon/UCR), univariate=True, sequence length=1092, classes=5
- train: N=132 dist={'0': 15, '1': 29, '2': 29, '3': 31, '4': 28} nan=0 inf=0 dupes=0
- val: N=23 dist={'0': 3, '1': 5, '2': 5, '3': 5, '4': 5} nan=0 inf=0 dupes=0
- test: N=308 dist={'0': 60, '1': 58, '2': 59, '3': 64, '4': 67} nan=0 inf=0 dupes=0
- validation availability: per_class_stratified_15pct_of_train_seed42_singletons_kept_in_train
- canonical split preserved: YES (train/test from source files)
- class entropy (train): 2.2826 bits

## Phoneme (secondary)

- source: timeseriesclassification.com/aeon-toolkit/Phoneme.zip
- exact paths: train=`data\external\extracted\Phoneme\Phoneme_TRAIN.ts`, test=`data\external\extracted\Phoneme\Phoneme_TEST.ts`
- format: .ts (aeon/UCR), univariate=True, sequence length=1024, classes=39
- train: N=185 dist={'0': 3, '1': 1, '2': 6, '3': 6, '4': 3, '5': 3, '6': 3, '7': 2, '8': 9, '9': 7, '10': 2, '11': 4, '12': 8, '13': 11, '14': 6, '15': 13, '16': 2, '17': 3, '18': 1, '19': 5, '20': 9, '21': 10, '22': 20, '23': 3, '24': 13, '25': 1, '26': 1, '27': 2, '28': 3, '29': 2, '30': 1, '31': 2, '32': 1, '33': 3, '34': 1, '35': 3, '36': 4, '37': 1, '38': 7} nan=0 inf=0 dupes=0
- val: N=29 dist={'0': 1, '2': 1, '3': 1, '4': 1, '5': 1, '8': 2, '9': 1, '11': 1, '12': 2, '13': 2, '14': 1, '15': 2, '19': 1, '20': 2, '21': 2, '22': 4, '24': 2, '36': 1, '38': 1} nan=0 inf=0 dupes=0
- test: N=1896 dist={'0': 38, '1': 2, '2': 59, '3': 61, '4': 37, '5': 34, '6': 23, '7': 15, '8': 98, '9': 75, '10': 16, '11': 49, '12': 90, '13': 112, '14': 64, '15': 134, '16': 20, '17': 29, '18': 3, '19': 58, '20': 102, '21': 111, '22': 214, '23': 25, '24': 130, '25': 8, '26': 5, '27': 21, '28': 25, '29': 18, '30': 13, '31': 22, '32': 1, '33': 23, '34': 8, '35': 30, '36': 42, '37': 11, '38': 70} nan=0 inf=0 dupes=0
- validation availability: per_class_stratified_15pct_of_train_seed42_singletons_kept_in_train
- canonical split preserved: YES (train/test from source files)
- class entropy (train): 4.8113 bits
