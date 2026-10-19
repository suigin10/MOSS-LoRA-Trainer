# MOSS Studio — English quick start

This is the **0.1.1rc5 preview release**, including a mouse wheel fix, resume fixes, an English GUI, model settings retained for new projects, clearer caption counts and state retention. The MOSS GPU resume fix is awaiting user testing.

## Install and launch

1. Extract the ZIP. To update an existing installation, close the app and copy the contents of `MOSS-LoRA-Trainer` over the existing application folder. Keep `data/` and `backend/`.
2. Install 64-bit Python 3.12 if needed. Run `start.bat` and choose **English** in the header. Your language choice is saved for future launches.
3. For a new backend, open **Easy setup** and start setup. It creates a dedicated venv on first use and checks or repairs that same environment on later runs. An existing working installation does not need a new venv for this GUI update.
4. In **Model**, browse for the MOSS-Image DiT, Qwen3.5-9B text encoder and FLUX.2 VAE. Use bf16 or supported ConvRot INT8 weights; W4A8 GPTQ Qwen is unsupported. Select the first shard for sharded models.
5. Add image folders in **Dataset**. Place `.txt` captions beside images with matching base names. Choose a VRAM preset, check settings and start training.

## Projects and settings

Mouse wheel input over a closed dropdown scrolls the settings page without changing its value. Open the dropdown to select another option.

**New / Copy** can create a project from recommended defaults or copy another project's settings. New projects retain the current model settings while resetting training settings and image subsets. Copy also keeps data and training settings. **Import** loads this app's `project_snapshot.json` or `project.json` into a new project. Language changes preserve unsaved settings, paths, prompts and log contents; project names and user content are never translated. Fix invalid numeric input before switching languages. Language selection is disabled during training, setup and environment checks.

Use **Count images / captions** to see recognized captions as N / M. A missing count of 0 means no captions are missing. Put `image.png` and `image.txt` in the same folder and save text as UTF-8. Select the folder containing the images directly; parent folders are not scanned recursively. The report shows expected filenames, empty text and encoding errors. It does not rename or edit your files.

The default plan is 1500 steps, learning rate 1e-4, rank 16, alpha 1, and AdamW8bit with a constant scheduler. Model and resume state saves default to every 100 steps. You can change values or select epoch mode for new training.

LoRA saving is flexible: change the step interval, the epoch interval, or use both. Set an interval to 0 to disable it. **Keep only the latest state** is ON by default: after a new state saves successfully, older completed states from this run are removed. The final state replaces the previous periodic state after a successful final save. LoRA files, resume source states and states from other runs are kept. Turn this option OFF to retain all states. The policy reduces retained disk usage; it still writes each scheduled state and temporarily requires space for the old and new states.

## Resume testing

Open **Resume from state**, choose a completed state save and enable **Restore original model, data and training settings**. Keep the original optimizer, learning rate, scheduler, data and step/epoch limit. The limit is the total training plan, not additional steps. For example, a state saved at step 600 resumes at 600, with the next update at 601 and the next 100-step save at 700.

The compatibility launcher fixes cumulative step accounting and batch positions without editing the installed sd-scripts checkout. Accelerate restores the actual optimizer and scheduler state. To change the training plan or continue past its original limit, clear resume and start new training from LoRA weights; that creates a new optimizer and scheduler.

Force stop loses unsaved progress. Use the latest completed state. A state already re-saved with incorrect counters cannot have its historical progress reconstructed by this fix; use a state from before the drift if possible.

## Logs and test scope

App labels, help text, dialogs, validation messages and ETA are translated. Backend/installer process logs retain their original language. Native Windows dialog controls follow the OS language.

Automated checks include GUI behavior and a CPU AdamW / Accelerate save-resume comparison with constant, linear and cosine schedulers. This test build's MOSS GPU training and AdamW8bit resume remain to be checked on real training data. If a problem persists, keep `training.log`, `train.toml`, `resume_source.json` and the original state's `train_state.json`.
