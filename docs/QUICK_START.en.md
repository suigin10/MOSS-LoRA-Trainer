# MOSS Studio with Kohaku v0.2.1

Version 0.2.0 adds automatic Triton installation and a separate Kohaku backend. Open **Start MOSS Studio.bat**, select English, then use **Easy Setup**. Choose sd-scripts (cu128/torch 2.8 or cu124/torch 2.6) or Kohaku (cu130/torch 2.13). Kohaku requires a compatible NVIDIA driver and Visual Studio C++ Build Tools. Select the installed engine in **04 VRAM / Speed** before starting training.

Triton is paired with the selected PyTorch version and checked by compiling and executing a small GPU kernel. The GUI and existing global Python environments are not used for backend installation.

The user confirmed completed 1500-step training runs on their RTX 4070 12GB with both Kohaku and sd-scripts, including both the old and new Kohaku settings (reported 2026-10-21). A 100-step run and a 50-step continuation were also confirmed. A short benchmark with the old rank16/alpha1, AdamW8bit, constant, INT8 / ckpt-resident settings measured about 4.10 s/step and 10.3 GiB device VRAM. This is not a speed or VRAM measurement of the current defaults. A full 4000-step run with the current defaults, clean dependency installation and generated-image quality comparisons remain unverified. Existing sd-scripts projects keep their original backend.

Kohaku uses KohakuMoss License 1.0, based on Apache-2.0 with an additional naming condition. The primary product branding is **MOSS Studio with Kohaku**. See the full license in `licenses/KohakuMoss-LICENSE.txt` and the Japanese integration guide [USER_GUIDE.ja.md#kohakuバックエンド](USER_GUIDE.ja.md#kohakuバックエンド).

This integration supports step-based LoRA training, disk caches, AdamW/AdamW8bit, checkpoint resume and step-based sampling. It rejects epoch limits, initial LoRA weights, per-module settings, Text embedder training, random augmentation and inline sample options. Kohaku uses its own MOSS objective and resolution planner. Its resume checkpoints cannot be shared with sd-scripts. Checkpoints are under `runs/<run>/kohaku/checkpoints`; LoRA exports are also copied to `runs/<run>/models` before old checkpoints are pruned.

New Kohaku training is based on the official recipe: rank/alpha 32/32, AdamW8bit, LR 1e-4, cosine with 100 warmup steps and a 10% LR floor, 4000 total steps, save every 200 steps and caption dropout 0.1. The total-step limit matches the upstream common recipe at 4000 steps; this GUI uses AdamW8bit and saves every 200 steps, while upstream uses AdamW and saves every 500 steps. Existing projects retain saved values, including the 0% floor of older cosine settings. The 12GB memory configuration stays INT8 / ckpt-resident / batch 1 / 6144 joint tokens. See [the settings guide](STABLE_SETTINGS.md) for both engines. The earlier 4.10 s/step measurement used rank16/alpha1 and AdamW8bit; it is not a benchmark of the new recipe.

The following guide describes the existing sd-scripts workflow:

# MOSS Studio — English quick start

This is the **v0.2.1 release**, adding elapsed time, measured average s/it, device-wide VRAM monitoring, editable CPU processing threads (default 2), a 12GB swap-16 preset and console-free GUI startup. The provider has confirmed operation through version 0.1.1 and previously verified training on an RTX 4070 (12 GB VRAM). Automated tests and monitoring checks are described in the [validation notes](../CONTRIBUTING.md#検証状況); see the development notes for automated tests, GPU benchmarks and remaining limitations.

## Install and launch

1. Extract the ZIP. To update an existing installation, close the app and copy the contents of `MOSS-LoRA-Trainer` over the existing application folder. Keep `data/` and `backend/`.
2. Install 64-bit Python 3.12 if needed. Run `Start MOSS Studio.bat` and choose **English** in the header. Your language choice is saved for future launches.
3. For a new backend, open **Easy setup** and start setup. It creates a dedicated venv on first use and checks or repairs that same environment on later runs. An existing working installation does not need a new venv for this GUI update.
4. In **Model**, browse for the MOSS-Image DiT, Qwen3.5-9B text encoder and FLUX.2 VAE. Use bf16 or supported ConvRot INT8 weights; W4A8 GPTQ Qwen is unsupported. Select the first shard for sharded models.
5. Add image folders in **Dataset**. Place `.txt` captions beside images with matching base names. Choose a VRAM preset, check settings and start training.

## Projects and settings

Mouse wheel input over a closed dropdown scrolls the settings page without changing its value. Open the dropdown to select another option.

**Add / Copy** can create a project from recommended defaults or copy another project's settings. New projects retain the current model settings while resetting training settings and image subsets. Copy also keeps data and training settings. **Import** loads this app's `project_snapshot.json` or `project.json` into a new project. Language changes preserve unsaved settings, paths, prompts and log contents; project names and user content are never translated. Fix invalid numeric input before switching languages. Language selection is disabled during training, setup and environment checks.

Use **Count images / captions** to see recognized captions as N / M. A missing count of 0 means no captions are missing. Put `image.png` and `image.txt` in the same folder and save text as UTF-8. Select the folder containing the images directly; parent folders are not scanned recursively. The report shows expected filenames, empty text and encoding errors. It does not rename or edit your files.

For caption wording, start with natural-language descriptions or a hybrid of tags and natural language. Briefly describe the visible subject, clothing, composition and background. Examples:

- Natural language: `A girl with blue hair is standing on a beach, wearing a white dress.`
- Tags + natural language: `character_token, blue hair, white dress. A girl is standing on a beach with the ocean in the background.`

Keep any character identifier token consistent across captions. There is no need to add unseen traits or excessive quality tags.

Use **Delete** to remove the selected project from the list. After confirmation, its entire folder moves to `data/projects/_trash/`. External models and images stay in place; files stored inside the project folder move with it. Unsaved edits are discarded. An empty project list stays empty across restarts; use **Add / Copy** to create a project again.

For sd-scripts, the recommended starting plan is 1500 steps, learning rate 1e-4, rank 16, alpha 1, and AdamW8bit with a constant scheduler, saving models and resume states every 100 steps. Apply these values manually; new projects use the Kohaku defaults described above. You can change values or select epoch mode for new training.

LoRA saving is flexible: change the step interval, the epoch interval, or use both. Set an interval to 0 to disable it. **Keep only the latest state** is ON by default: after a new state saves successfully, older completed states from this run are removed. The final state replaces the previous periodic state after a successful final save. LoRA files, resume source states and states from other runs are kept. Turn this option OFF to retain all states. The policy reduces retained disk usage; it still writes each scheduled state and temporarily requires space for the old and new states.

## Resume and extend training

Open **Resume / extend from state**, choose an interrupted or final state, and restore the original model/data/training settings. Keeping the same total limit restores the original optimizer and scheduler without LR recalculation.

To extend a completed run, increase the total step limit. A state at step 1500 with a new total of 3000 runs 1500 additional steps, starting with update 1501. Epoch-based projects can extend the total epoch limit. Optimizer moments and counters are retained.

Changing the total limit recalculates LR at the saved step using the new total schedule. Constant LR stays constant; linear and cosine LR change at resume. A final state whose old schedule reached zero LR can therefore train further on the extended curve. Scheduler and optimizer are not stepped artificially; inconsistent saved counters stop the run with an error. Changing optimizer type, scheduler type, base LR or data requires starting new training from LoRA weights instead.

The compatibility launcher leaves the installed sd-scripts source intact. Resume writes to a new run folder and records the source state and policy. Force stop loses unsaved progress. Use a completed, consistent state checkpoint.

## Logs and test scope

App labels, help text, dialogs, validation messages and ETA are translated. Backend/installer process logs retain their original language. Native Windows dialog controls follow the OS language.

Automated checks include GUI behavior and a CPU AdamW / Accelerate save-resume comparison with constant, linear and cosine schedulers. This test build's MOSS GPU training and AdamW8bit resume remain to be checked on real training data. If a problem persists, keep `training.log`, `train.toml`, `resume_source.json` and the original state's `train_state.json`.

The new monitor uses nvidia-smi every ~3 seconds; device-wide memory includes other applications and excludes shared system memory. Observed peak is sampled, not an exact maximum. Elapsed time includes loading and saving; average s/it uses elapsed differences between training STEP logs from this invocation only. Existing project settings are preserved. Increase swap from 16 to 20 or more if memory is tight.

CPU processing threads are editable in 04 VRAM / Speed (default 2; positive integer). More threads may not be faster. This is separate from DataLoader workers and requires no extra installation. Project copies and exported settings retain the value; older projects with no saved value default to 2. Existing saved values are preserved.

The batch launchers detach the GUI without retaining a console window. An existing terminal used to invoke a launcher stays open. Startup failures are shown in a dialog and appended to data/startup.log (or MOSS_TRAINER_DATA). Use python launch.py for console diagnostics.

Preflight check validates paths and settings. Use Check images / captions in the Dataset tab for dataset details. Preview launch settings and Export settings are together in Run / Log.
