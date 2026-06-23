### Audio Data

The audio recordings used in this demo are not distributed with this repository.

Please apply for and download the ‘forensic_eval_01’ dataset from the official website page:

```text
https://forensic-voice-comparison.net/databases/#forensic_eval_01
```

After obtaining the dataset, please place the audio files into folders as follows:

```text
forensic_eval_01/
└── 0_audio/
    ├── forensic_eval_01_train_2016-05-12/
    ├── forensic_eval_01_test_2016-05-12/
    ├── forensic_eval_01_train_2016-05-12_VADed/
    └── forensic_eval_01_test_2016-05-12_VADed/
```

Folders without the `_VADed` suffix correspond to the original dataset recordings.

Folders with the `_VADed` suffix are voice-activity-detected versions for experiments involving pre-segmented or VAD-processed audio.