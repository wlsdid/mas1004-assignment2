# Notes for coding agents

README.md says what the student has to do and why. `PLAN.md` gives the steps
with a check for each. This file says how to set things up and run them, what
the given tools do, how to make the usual changes, how to train on Colab, how
to release, and what is the student's to decide.

The student works in VS Code with Cline and may never have used a terminal.
Run the commands yourself, and tell the student only what they have to do with
their own hands, such as logging in to a website or pressing Run all on Colab.

## First setup

The template is https://github.com/jdasam/mas1004-assignment2. Each step
checks first, so the setup can be started again after VS Code is reopened.

1. Check the computer. PyTorch has no packages for Intel Macs or for Windows
   on an ARM processor, and needs macOS 14 or later on Apple Silicon. On those
   computers, tell the student that training runs on Colab, skip step 6, and
   see "Computers that cannot run PyTorch" below.
2. git, GitHub CLI and login: the same as Assignment 1. `git --version`,
   `gh --version`, `gh auth status`; install or log in as needed
   (`gh auth login --hostname github.com --git-protocol https --web`, then
   `gh auth setup-git`).
3. Ask the student for a folder (suggest their Documents folder) and run
   `gh repo create mas1004-assignment2 --template jdasam/mas1004-assignment2 --public --clone`
   there. It has to be public: the grader and Colab read it without logging
   in. Work in the new folder from here on.
4. Install uv if `uv --version` fails (see "Running Python").
5. Ask the student for the link to the starter photos, which is on Cyber
   Campus, and run `uv run python src/get_data.py --link "<the link>"`. It
   downloads the photos (about 6 GB; it needs 13 GB free while it unpacks
   them) into `data/starter/` and checks them. It keeps the link in
   `data/starter_link.txt`, so later runs need no `--link`. Do not write the
   link into any file that is pushed: the photos are for this class only.
6. `uv run python src/official.py --quick`. It checks the recipe, trains 20
   steps and measures on 200 validation photos. It must end with
   "validation score" and a line saying how long a full run would take on this
   computer. Tell the student that time.
7. `uv run python src/colab.py link`. It writes the address of the student's
   repository into `assignment2_colab.ipynb` and prints the link that opens it
   in Colab. Commit and push the notebook.
8. Tell the student where the folder is, to open it in VS Code (File, Open
   Folder) and use Cline there from now on, to keep the Colab link, and to
   fill in the registration form on Cyber Campus.

## Updating the given files

```
git remote add template https://github.com/jdasam/mas1004-assignment2.git
git fetch template
git checkout template/main -- README.md PLAN.md AGENTS.md CLAUDE.md .gitignore .python-version pyproject.toml uv.lock requirements.txt assignment2_colab.ipynb src/rules.py src/official.py src/export.py src/check_submission.py src/curate.py src/gps_sort.py src/get_data.py src/colab.py src/cities.geojson short_report/mas1004.sty long_report/mas1004.sty
uv run python src/colab.py link
git add assignment2_colab.ipynb
git commit -m "Update the given files"
git push
```

Skip the first line if the remote exists. This does not touch
`track_a/`, `src/train_b.py`, `submission/`, `data/`, `experiments.csv`,
`colab_runs.txt` or `LOG.md`.

## Running Python in this repository

Run every Python command through uv, from the repository folder, as in
Assignment 1: `uv run python src/official.py ...`. uv installs Python 3.12 and
the exact versions in `uv.lock`, including torch 2.6.0 and torchvision 0.21.0,
the versions the grader uses. Do not use pip, conda or another Python, and do
not change versions: a recipe that uses something only a newer torchvision has
works here and fails on the grader.

If `uv` is missing: macOS or Linux
`curl -LsSf https://astral.sh/uv/install.sh | sh`; Windows PowerShell
`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.
Then quit and reopen VS Code.

### Computers that cannot run PyTorch

On an Intel Mac or Windows on ARM, `uv run python` stops with an error about
the platform. There, run `src/get_data.py` and `src/colab.py` with
`uv run --no-project python ...` (they need only Python), `src/curate.py drop`,
`undrop` and `count` with `uv run --no-project --with numpy --with pillow python ...`,
and everything that trains or runs a model on Colab.

## Where to train

`official.py --quick` prints how long a full run would take on this computer.
If it says more than 30 minutes, train on Colab (see "Training on Colab");
otherwise train here, unless the student prefers Colab. The same goes for
Track B, whose runs take longer than Track A runs.

## What the student decides

Do not decide these for the student. Do the work around them once the student
has decided.

- What tells the cities apart, in their own words.
- Decisions from looking at photos: what they see, the rule, and which
  photos to drop or collect. Run `curate.py` and show the student the sheets. Do not suggest
  which photos to drop or what the photos have in common; drop exactly the
  photos the student names, with the student's reason.
- Which change to try next is the student's choice. Answer when they ask for
  ideas, and explain what a change does, but let them choose. Never make a
  change the student did not ask for, even one that their words seem to
  point to; ask first.
- The reason for each change, and what the student thinks the result means,
  in `LOG.md`. Ask with an open question, not with wordings to choose from,
  and write down what the student says. Do not make one up.
- The short report: do not write, edit, translate or tidy it, and do not touch
  `short_report/`.

## Track A

```
uv run python src/official.py --tag balanced
uv run python src/official.py --tag baseline --seed 1
uv run python src/official.py --quick
```

- `official.py` trains ResNet18 on `track_a/selection.csv` (or all training
  photos) with `track_a/recipe.py`, for exactly 250,000 photos, then writes
  `results/<tag>_seed<seed>/model.onnx` and adds a row to `experiments.csv`
  with the validation score. Never change `official.py`, `rules.py`,
  `export.py` or `check_submission.py`.
- Train each recipe once, with seed 0 (the default). Only the starter recipe
  (tag baseline) is trained with seeds 0 and 1, to show the student how much
  the score moves by chance. Train another seed only when the student asks.
- `recipe.py` must keep all its functions and their arguments. It may import
  only torch (not torch.hub, torch.utils, torch.jit, torch.onnx and a few
  others), torchvision.transforms, numpy, math, random, PIL, functools,
  itertools, collections, copy, typing and dataclasses, and may not use
  open, eval, exec, getattr, load, save, hub, models, datasets or io.
  `official.py` lists what breaks the rules before it trains. Everything the
  recipe needs has to be in the one file.
- Batch size 16 to 256. A bigger batch means fewer steps, not more photos.

### Making a change

1. Change only what the student asked for. Show the student
   `git diff track_a/` and say in one sentence what the change does.
2. `uv run python src/official.py --quick`. If it fails, fix it before
   training.
3. Choose a new tag for every new recipe or selection: short, lowercase, and
   saying what changed (`balanced`, `no_flip`, `mixup`). Never use a tag again
   for a different recipe.
4. Commit the change, with the tag in the message, before training, so that
   the commit in `experiments.csv` is the recipe that was trained.
5. Train it, here or on Colab.
6. Add an entry to `LOG.md` in the form at the top of that file: the score,
   and the student's reason for the change.

The usual changes in `recipe.py`:

- Every city drawn equally often: in `sample_weights`, give each photo one
  over the number of photos of its city, with `import collections` at the top
  of the file:
  ```
  counts = collections.Counter(cities)
  return [1.0 / counts[c] for c in cities]
  ```
- No left-right flips: delete `T.RandomHorizontalFlip()` from
  `train_transform`.
- Other augmentation: `train_transform()` takes no argument and returns a
  function (for example a `T.Compose`) that gets one PIL image and returns a
  224 x 224 PIL image or a float tensor (3, 224, 224) with values 0 to 1. Do
  not normalise; official.py does. Transforms that need a tensor, such as
  `T.RandomErasing`, go after `T.ToTensor()`.
- Mixing photos (mixup, cutmix): in `mix`, return the mixed images and
  targets as probabilities of shape (BATCH_SIZE, 5), made with
  `F.one_hot(labels, 5).float()`. `F.cross_entropy` in `loss_fn` takes
  probabilities as well as labels, so `loss_fn` can stay. Draw random numbers
  with torch, random or numpy, which official.py seeds.
- Learning rate and schedule: `make_optimizer` and `make_scheduler`. The
  scheduler is stepped once after every step; `total_steps` is 250,000 divided
  by `BATCH_SIZE`.
- After each step, or at the end (for example averaging the weights):
  `on_step` and `final_model`. `final_model` must return the same ResNet18.

### Making selection.csv

`selection.csv` has one column, `path`, relative to `data/starter/train`, for
example `busan/commons_104708282.jpg`. A path may appear more than once. Make
it with a script in `scripts/` (for example `scripts/make_selection.py`),
commit the script with the file, and print the counts with
`uv run python src/curate.py count`.

- File names say where a photo came from: `commons_<id>.jpg` (Wikimedia
  Commons), `osv5m_<id>.jpg` (cameras on cars and bikes) and
  `youtube_<video id>_<seconds>.jpg` (a frame of a walking or driving video).
- `data/starter/ATTRIBUTION.csv` has a row per photo: `file`
  (`train/<city>/<name>`), `source`, `group` (the photographer or the YouTube
  channel), `title`, `author`, `license` and `source_page`.
- `curate.py drop` and `undrop` change `selection.csv` and keep the reasons in
  `track_a/dropped.csv` (columns `path`, `reason`, `when`). A script that
  writes `selection.csv` must leave out every path in `dropped.csv`, unless
  the student asks otherwise.

## Curating photos

```
uv run python src/curate.py count
uv run python src/curate.py look busan --every 20
uv run python src/curate.py mistakes results/balanced_seed0/model.onnx
uv run python src/curate.py suspects results/balanced_seed0/model.onnx --every 2
uv run python src/curate.py drop look_busan_03 5 17 40 --reason "inside a shop"
uv run python src/curate.py undrop busan/commons_104708282.jpg
```

Sheets go to `results/curate/<name>_NN.png`, each with a `.csv` of the numbers
and paths on it. Sheets of `mistakes` and `suspects` carry the run's name
(`mistakes_baseline_seed0_busan_00`), so a later run does not replace them.
Open the PNG for the student to look at. `look` without `--every` draws every
selected photo of the city (Tokyo is about 280 sheets); `--every 20` gives a
few sheets for a first look. `mistakes` and `suspects` run the model on the
CPU and take a few minutes; `--every 4` uses every fourth photo. After
training on Colab, run them on Colab too (the models stay there), with
`--every 4` for `suspects`: Colab has two CPU cores, and `suspects` on every
training photo takes almost half an hour there. Open the sheets that come
back.

## Track B

```
uv run python src/train_b.py --tag b1 --model mobilenet_v3_large --size 320 --epochs 6 --balance
uv run python src/train_b.py --tag b2 --folders data/starter/train data/mine --submit
uv run python src/check_submission.py
uv run python src/gps_sort.py <folder of photos>
```

- `train_b.py` is a starting point. Change it as the student wants. What
  counts is `submission/model.onnx` and `submission/submission.json`.
- `check_submission.py` checks the format and the limits (12 million
  parameters, 2.0 billion multiply-adds at the chosen size) exactly as the
  grader does, then scores the model on `data/starter/val`. It must print
  "format and limits: OK".
- Some sizes, measured: ResNet18 at 224 has 11.2 million parameters and 1.81
  billion multiply-adds (288 is over the limit); MobileNetV3-Large 4.2 million,
  0.44 billion at 320 and 0.85 at 448; EfficientNet-B0 4.0 million, 0.78
  billion at 320; RegNetY-1.6GF 10.3 million, 1.61 billion at 224. ResNet34
  does not fit. `train_b.py` prints both numbers after training.
- Starting weights: any, for example torchvision's `IMAGENET1K_V1` and
  `IMAGENET1K_V2`. Write the weights in `submission.json`.
- Extra photos go in `data/mine/<city>/` (or other folders of `<city>/`).
  `gps_sort.py` does this for phone photos. Never commit photos; `data/` is
  in `.gitignore`.
- Do not train on `data/starter/val` if the student wants the validation
  score to mean anything.
- Export a model of your own with `export.export_onnx(net, path, size)`; it puts
  the normalisation inside the file.

## Training on Colab

The student only opens a link and presses Run all. You prepare the runs here
and bring the results back.

1. Write the commands into `colab_runs.txt`, one per line, each starting with
   `python src/`, as you would run them here without `uv run`. Replace what
   was in the file before. For example:
   ```
   python src/official.py --tag balanced
   python src/curate.py mistakes results/balanced_seed0/model.onnx --every 2
   ```
   For Track B, add `--submit` to the training line and then
   `python src/check_submission.py --no-val`: only `submission/` comes back
   from Colab, not the models in `results/`. `--no-val` skips scoring the
   validation photos again on Colab's two CPU cores (about 7 minutes);
   `train_b.py` has printed that score already.
2. Commit and push `colab_runs.txt` with the code it needs.
3. Tell the student: open the Colab link from the setup, choose Runtime, Run
   all, and tell you when `from_colab.zip` has downloaded. One Track A run
   takes about 25 minutes there, plus 5 to 10 minutes to start; a 6-epoch
   MobileNetV3-Large run of `train_b.py` at 320 takes about 45 minutes. In
   every new session Colab warns that the notebook was not written by Google
   (the student chooses "Run anyway") and asks to connect Google Drive (the
   student allows everything it asks for). The first time, the notebook also
   asks for the link to the starter photos. If the Colab link is lost,
   `uv run python src/colab.py link` prints it again.
4. When the student says it has downloaded:
   `uv run python src/colab.py take`. It finds the newest `from_colab*.zip`
   in the Downloads folder (or give it the path), adds the new rows to
   `experiments.csv`, and copies the sheets, `submission/` and the output of
   every run (`results/colab_logs/`). Then continue with step 6 of "Making a
   change".

A line that already ran on Colab with the same recipe, selection and script is
skipped, so a session that broke off carries on where it stopped. The free GPU
time runs out after a few hours a day. Colab then says the runtime
disconnected "because of inactivity or maximum length" and, on reconnecting,
that it cannot connect to a GPU because of usage limits. Tell the student to
close that window, not to connect without a GPU, and to press Run all the next
day. If a run failed, the reason is at the end of its file in
`results/colab_logs/`; fix it, push, and ask for Run all again.

For Track B photos of the student's own, the student puts the `data/mine`
folder into Google Drive as `My Drive/mas1004/assignment2/data/mine` (on
drive.google.com, New, Folder upload). Tell them which folder to choose.

## Releasing

When the student asks to submit:

1. Track A: run `uv run python src/official.py --quick`. If it fails, fix it
   with the student before releasing.
   Track B: run `uv run python src/check_submission.py`. It must say
   "format and limits: OK".
2. Commit and push the code, `LOG.md` and `experiments.csv`.
3. Ask the student for a short title that says what changed, then:

```
gh release create track-a-<n> track_a/recipe.py track_a/selection.csv --title "<title>" --notes ""
gh release create track-b-<n> submission/model.onnx submission/submission.json --title "<title>" --notes ""
```

   Leave out `track_a/selection.csv` if there is none. `<n>` is one more than
   the last release of that track (`gh release list`).
4. Tell the student that the grader takes the newest release of each track
   made before midnight, and that the leaderboard is updated when it has
   finished, usually the next morning.

## The long report

Like Assignment 1: `long_report/long_report.tex`, filled in with the student,
with the run table from `experiments.csv` and the student's `LOG.md`. Every
number must be one that code printed in this repository, here or on Colab
(`results/colab_logs/`).
