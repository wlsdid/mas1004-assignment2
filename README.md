# Assignment 2: Which city was this photo taken in?

MAS1004 Data & AI, 2026 Fall

- Out: Thursday 22 October (introduced in a recorded lecture)
- Mid-point check: Friday 30 October, 22:00
- Due: Tuesday 10 November, 22:00
- Final results: in class, Thursday 12 November

## What you do

Everyone in the class solves the same problem: given a photo, say whether it
was taken in Seoul, Busan, Tokyo, Osaka or Taipei. A leaderboard scores
everyone's model every night on frames from YouTube walking and driving
videos, from channels that are not in the starter data. The final ranking
uses a different set: street photos taken with a phone, which nobody sees
before the deadline.

You work on it in two ways, called tracks. Both are required.

- Track A: the same model, the same photos, the same amount of training for
  everyone. The model is ResNet18. The photos are the starter photos we give
  you. You choose which of those photos to use and how the model is trained,
  and the instructor's computer trains it with your choices. The score shows
  only how good your choices were.
- Track B: anything goes, as long as the model stays small. Take photos
  yourself, ask friends for their travel photos, find more photos online,
  try another small model or a larger image size, ask anyone for ideas. You train it and hand in the trained model.

The point of the assignment is to find out, by measuring, what makes a model
better: cleaning the photos it learns from, augmenting them, and collecting
more of them. Try several ways, measure what each one did, and understand
why. The leaderboard is there to make that fun. Three things matter most:

- One change at a time, measured. "I left out the indoor photos, and the
  validation score went from 0.498 to 0.515" tells us something. "I changed six things and it got better" does not.
- Your own judgement in the loop. Look at the photos your model gets wrong,
  decide what to do about them, and measure whether your decision helped.
  Your agent can make the changes; you decide which ones.
- Honest explanations. A method that did not work, with a good guess about
  why, is worth as much as one that did.

## What you hand in

On GitHub, as releases of your repository (your agent makes them, see "How to
submit"):

1. Track A: your `track_a/recipe.py` and `track_a/selection.csv`
2. Track B: your `submission/model.onnx` and `submission/submission.json`

On Cyber Campus (LMS), by the deadline:

3. The short report as a PDF: one page, written by you.
4. The long report as a PDF: written with your agent.

By the mid-point check (Friday 30 October, 22:00): your nickname and
repository are registered, and you have made at least one Track A release,
for example the starter recipe as it is. This checks that your releases
reach the leaderboard; the next morning you should see your nickname there.

## Getting started

You need VS Code with Cline, as in Assignment 1, and your GitHub account. In
VS Code, open Cline and give it this request:

```
Set up MAS1004 Assignment 2 for me. Follow the section "First setup" in
https://raw.githubusercontent.com/jdasam/mas1004-assignment2/main/AGENTS.md
```

It makes your copy of this repository, downloads the starter photos (about
6 GB), installs what the assignment needs and trains the starter recipe once
for 20 steps to check that everything works. It asks you for the link to the
starter photos, which is on Cyber Campus. The photos are for this class only:
do not share the link or put it in your repository. Then fill in the registration
form on Cyber Campus with your student number, the address of your
repository, and a nickname for the leaderboard. Only the nickname is shown.

### Google Colab

Colab gives you a GPU for free. One Track A training takes about 25 minutes
there, against two to four hours on a laptop without a GPU. At the end of
the setup, your agent tells you how long one takes on your computer and gives
you a link to the notebook in your repository, which opens it in Colab.

You do not write anything in Colab. Your agent writes the runs into your
repository and pushes them; you open the link and choose Runtime, Run all;
when your browser has downloaded `from_colab.zip`, you tell your agent, and it
brings the results into your repository. In every new Colab session you also:

- choose "Run anyway" when Colab warns that the notebook was not written by
  Google
- choose "Connect to Google Drive", choose your Google account in the window
  that opens, and allow everything it asks for

The first time, the notebook also asks for the link to the starter photos.
It takes 5 to 10 minutes before the first training starts.

The free GPU time is limited, a few hours a day. When it runs out, Colab says
the runtime disconnected "because of inactivity or because it reached its
maximum length", and when you reconnect it says it cannot connect to a GPU
because of usage limits. Close that window; do not choose "Connect without
GPU". The runs that finished are kept in your Drive, and Run all on the next
day carries on with the rest.

## How you work

`PLAN.md` is the plan for the whole assignment, step by step, with a check for
each step. Read it, then give it to your agent one step at a time. `AGENTS.md`
tells your agent how to run things in this repository, how to make the usual
changes, and what is yours to decide.

You are responsible for everything you hand in, including what your agent
wrote. Every number in your reports has to come from output that code printed,
and when the agent says it changed something, look at what it changed.

## The rules

### Track A: same model, same photos, same budget

`src/official.py` trains the model. You cannot change it; the instructor's
computer runs the same file. It fixes:

- the model: ResNet18 with its ImageNet weights, and a new last layer for the
  five cities
- the photos: only starter training photos, each with the city of the folder
  it is in. You cannot change a photo's city. The starter photos come from
  Wikimedia Commons, OSV-5M (cameras on cars and bikes, through Mapillary)
  and frames of YouTube walking and driving videos. The training photos are
  as they were collected, not cleaned: some are museum objects, maps, food or
  press conferences, and the frames of one video look much alike. The
  validation photos were cleaned by a program, which took out most such
  photos but not all.
- the budget: 250,000 photos go through the model during training, so it
  sees each training photo between five and six times. A batch of 64 means
  3,906 steps; a batch of 128 means 1,953.
- the image size the model sees: 224 by 224
- the seeds: the instructor trains your recipe three times, with seeds 0, 1
  and 2, and your score is the mean

You change two files.

- `track_a/selection.csv` lists the training photos to use. Leave a photo out
  to drop it; list a photo twice to show it twice as often. Without this file,
  every training photo is used.
- `track_a/recipe.py` decides how the model is trained: how photos are
  changed before the model sees them (augmentation), how often each photo is
  drawn, the optimizer and learning rate, the loss, mixing photos in a batch,
  anything you want to do after each step, and the batch size. The file
  explains each part.

`recipe.py` works only with what official.py gives it. It cannot read files,
use the network, run other programs or load another model, and official.py
refuses a recipe that tries. You can make selection.csv any way you like, on
your own computer.

### Track B: anything that fits

- At most 12 million parameters, and at most 2.0 billion multiply-adds for
  one photo. ResNet18 at 224 by 224 has 11.2 million parameters and uses 1.8
  billion. `src/check_submission.py` measures both from your file, exactly as
  the grader does.
- You choose the image size, from 96 to 512. A smaller network can look at a
  bigger image: MobileNetV3-Large at 448 by 448 uses 0.9 billion.
- Any starting weights, for example torchvision's `IMAGENET1K_V1` and
  `IMAGENET1K_V2`, or none at all.
- If you take frames from YouTube videos: the frames of one video look
  almost the same, so keep all frames of a channel on one side, all in
  training or all in the photos you validate on.

## The leaderboard

Every night at midnight (Korean time), the grader takes the newest release of
each track from each repository, trains the Track A recipes and scores
everything on the leaderboard photos. When it has finished, usually in the
morning, it updates the leaderboard at
https://jdasam.github.io/mas1004-leaderboard/

- The leaderboard photos are street scenes from YouTube walking and driving
  videos, at most 20 frames from one video, the same number for every city.
  None of their channels is in the starter data. If you take YouTube frames
  yourself and one of your channels is a leaderboard channel, your
  leaderboard score comes out higher than it should.
- The score is the share of photos placed in the right city, averaged over
  the five cities.
- Only your newest release before midnight counts, so you get one score a
  day per track.
- The title of your release appears next to your score. Use it to say what
  you changed.
- If your release could not be scored, the reason is shown in red. Fix it and
  release again.
- For Track A, it also shows the validation score of the three models the
  grader trained, so you can compare it with your own runs.
- After the deadline, your last releases are scored once on the final
  photos. That decides the final ranking. The final photos are a different
  set from the leaderboard photos: street photos taken with a phone. A change
  that helps on the leaderboard does not always help on the final photos,
  and the final leaderboard shows both scores side by side.

A difference of two or three points between two people on the leaderboard is
mostly luck: the same recipe trained again can move that much.
Compare your own changes on the validation photos in `data/starter/val`,
which `official.py` and `check_submission.py` score for you.

## How to submit

Ask your agent. It follows "Releasing" in AGENTS.md, which checks your files
first and then runs, for example:

```
gh release create track-a-4 track_a/recipe.py track_a/selection.csv --title "cities balanced, no flips"
gh release create track-b-2 submission/model.onnx submission/submission.json --title "MobileNetV3 at 384, my Seoul photos"
```

---

## Ways to go about it

Nothing in this section is required. It is one way to work through the
assignment, with ideas to start from: take what is useful and go your own
way. `PLAN.md` turns it into steps for your agent.

### Look at the photos yourself

Before any training, look at photos of each city yourself (ask your agent:
"Show me some training photos of Busan.") and notice what tells the cities
apart: the writing on signs, which side of the road cars drive on, road
markings, scooters, power lines, buildings, plants. Which of these can a model
still see in a 224 by 224 version of the photo?

### Start from the starter recipe

Train the starter recipe with `official.py` twice, with seeds 0 and 1, and
release it. The seed sets the random choices in training, such as the order
of the photos, and it is the only thing you change between the two runs.
Their scores still differ, and a change that moves the score by less than
this difference may be luck. Look at the mistakes with
`src/curate.py mistakes`.

### Track A: better choices with the same photos and budget

Change one thing at a time, measure it against the run before, and keep what
helps. Some directions, not a list to work through:

- Busan has fewer than half as many training photos as Tokyo. What does the
  model do about that, and what can you do?
- Some starter photos are not street photos at all: food, museum objects,
  the inside of a shop. Does leaving them out help or hurt, when the budget
  stays 250,000?
- Augmentation changes a photo a little every time it is drawn: a crop, a
  zoom, other colours, a blur. The model sees each photo five or six times,
  enough to learn the photos themselves by heart; augmentation makes that
  harder. The starter photos are 512 pixels on the shorter side, so a crop
  that zooms in still has detail. Which changes help, and which hurt?
- A left-right flip turns writing into mirror writing and puts the traffic
  on the other side of the road. Tokyo drives on the left.
- The budget is fixed. Is it better spent on all photos once or on fewer
  photos several times? On a bigger batch or more steps?

### Your own judgement in the loop

Look at your model's most confident mistakes (`curate.py mistakes`) or at the
training photos it doubts most (`curate.py suspects`), and see what they have
in common. Decide what to do and write the rule down before you act, for
example "drop training photos taken indoors" or "collect 100 more photos of
residential streets in Busan". Then do it (`curate.py drop` takes photos out
of your selection and records your reason), and measure the score before and
after. You make these decisions, not your agent; your agent makes the changes
you ask for.

### Track B: anything that fits

Start from `src/train_b.py` or from anything else. Things people do:

- more photos, especially photos like the final ones. `src/gps_sort.py`
  sorts a folder of phone photos into the five cities by the GPS in them,
  and saves copies without the location data.
- a different small model, or a bigger image size for a smaller model
- everything that worked in Track A

---

## The short report

One page of text, written by you, with no language model at any stage, as in
Assignment 1. If Korean is your first language, write it in Korean. Write
freely about what you tried, what surprised you, and what you learned. The
template is in `short_report/`.

## The long report

Written with your agent, from `long_report/long_report.tex`. Its sections
are a suggestion; arrange it as you like. A reader should be able to see:

- what you tried in each track, why, and what you think each result means,
  with the run table from `experiments.csv`
- the decisions you made yourself from looking at photos, what they changed,
  and the requests you gave your agent for them
- how you collected any extra photos
- your validation scores next to your leaderboard scores, and why they might
  disagree
- which change helped most, and what you would do next with one more week

Your `LOG.md` goes at the end. Every number in the report has to come from
output of code you ran.

## How this is graded

No particular path is required. These are looked at:

- the mid-point check
- whether each change you made was measured fairly, and what you concluded
- whether some of your decisions came from photos you looked at yourself,
  and whether you measured their effect
- the long report: whether a reader can follow what you did and why
- the leaderboard: valid releases in both tracks at the deadline, and whether
  they beat the starter recipe (Track A) and the starter model (Track B). The
  ranking itself counts only a little.
- the short report

## When you are stuck

- Ask your agent to run `python src/official.py --quick`. It checks your recipe
  against the rules and trains for 20 steps.
- Ask your agent to run `python src/check_submission.py` for Track B. It checks
  your model the way the grader does.
- Look at what others wrote in the titles of their releases on the
  leaderboard, and ask them. Ideas are free to share.
- Ask in class.
