# Official HotpotQA files

Download the labeled [training set](https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json)
and [distractor development set](https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json)
from the [HotpotQA project](https://hotpotqa.github.io/). Place them here with
these exact filenames:

- hotpot_train_v1.1.json
- hotpot_dev_distractor_v1.json

The prepare command verifies the expected full record counts, records SHA-256
hashes of both files, and locks a train/validation split. The labeled official
development set is reserved for the experiment's final test. These files are
not included in this repository.
