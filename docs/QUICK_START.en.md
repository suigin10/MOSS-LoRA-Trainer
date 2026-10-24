# MOSS Studio with Kohaku v0.2.2

Windows GUI for MOSS-Image LoRA training with Kohaku and sd-scripts. New projects use Kohaku. Saved projects keep their engine and settings.

## Install and update

1. Extract the release ZIP and run **Start MOSS Studio.bat** with 64-bit Python 3.12. Choose **English** in the header.
2. Use **Easy setup** to create or repair a dedicated backend: sd-scripts uses cu128 / PyTorch 2.8 or cu124 / 2.6; Kohaku uses cu130 / 2.13. Kohaku also needs a compatible NVIDIA driver and Visual Studio C++ Build Tools.
3. Select the engine in **04 VRAM / Speed**. Setup preserves unrelated Python environments. Triton is paired with PyTorch and verified with a small GPU kernel.
4. In **01 Model**, select the MOSS-Image DiT, Qwen3.5-9B encoder and FLUX.2 VAE. For sharded models, select the first shard. W4A8 GPTQ Qwen is unsupported. See the [settings guide](STABLE_SETTINGS.md) for download links.
5. Add image folders in **02 Dataset**, review settings and run the preflight check before training.

To update, close the app and copy the new application files into its folder. Keep `data/`, `backend/` and your model files. Existing environments can be repaired through Easy setup; CAME requires `pytorch-optimizer` 4.0.0, which setup now installs. Old `start.bat` / `install.bat` files can be removed; use **Start MOSS Studio.bat**.

## New-project defaults

Kohaku starts with FFN_down only, rank/alpha 32/32, AdamW, LR 1e-4, cosine, 100 warmup steps, a 10% LR floor, **2000 steps and saves every 100 steps**. Caption dropout starts at 0.1; **0 is recommended for natural-language captions**. Timestep sampling is shift with flow shift 3.0 and sigmoid scale 1.0. The 12GB starting memory plan uses INT8 model files, ckpt-resident, batch 1 and 6144 joint image/text tokens. Compilation and training samples start OFF.

The new-project step count, save interval, optimizer and scheduler also apply when switching to sd-scripts. Apply the sd-scripts rank, LR and other starting values from the [settings guide](STABLE_SETTINGS.md) manually. Switching engines and updating the app do not reset saved values.

- **Training target:** Kohaku offers `ffn-down`, `attn-mlp` (Attention + FFN, the previous behavior) and `attn`. Projects saved before this selector existed retain `attn-mlp`. For sd-scripts, the guide describes the include/exclude patterns for FFN_down only.
- **Optimizer:** AdamW is the default for both engines. AdamW8bit and CAME are optional. Adafactor is sd-scripts only. Choosing CAME does not reset LR or scheduler.
- **Scheduler:** cosine remains the default. Optional `cosine_with_restarts` starts at 2 cycles, meaning one restart. The cycle field is editable only for that scheduler. Warmup runs once at the start. Kohaku applies its LR floor to each cycle; sd-scripts decays toward zero.
- **Timestep:** Kohaku maps shift to its logit-normal sampler with mu=0 and the selected sigmoid scale and shift. The previous MOSS sampler remains available through sigma. Keep Kohaku's weighting scheme at logit_normal. Training flow shift is separate from sample-generation flow shift.

Start a new run when changing target, optimizer, scheduler, cycle count or timestep settings. Saved projects retain their original values, including older cosine LR floors of 0.

## Projects, captions and settings

**Add / Copy** creates a fresh project while retaining current model paths, or copies an existing project's settings and subsets. Run history and resume paths are not copied. **Import settings** accepts this app's `project.json` or `project_snapshot.json` and creates a new project; review paths when importing from another PC. It does not import arbitrary TOML or other GUIs' formats.

Put UTF-8 captions beside each image with matching names, such as `image.png` and `image.txt`. Add the folder containing images directly; subfolders are not scanned recursively. Use **Check images / captions** for missing captions, empty files and encoding errors. Missing captions use the subset's class tokens.

Natural-language descriptions or a hybrid of tags and natural language are useful starting points:

- Natural language: `A girl with blue hair is standing on a beach, wearing a white dress.`
- Tags + natural language: `character_token, blue hair, white dress. A girl is standing on a beach with the ocean in the background.`

Keep identifier tokens consistent and describe visible features. For natural-language captions, set Caption dropout to 0 if desired; the app does not change it automatically.

Use STEP or Epoch mode to choose the training limit. STEP=0 activates the Epoch limit; the STEP field stays editable so you can switch back. Kohaku supports STEP limits only. In STEP mode, an unused Epoch value of 0 is accepted.

Preflight checks paths and settings. **Preview launch settings** and **Export settings** are in **Run / Log**. Language changes preserve valid unsaved settings, prompts and logs. Closed dropdowns ignore wheel selection changes. Deleting a project moves its directory into `data/projects/_trash/`; external images and models remain in place.

## Save, resume and extend

Run folders are under `data/projects/<ID>/runs/<run>/`. Final and periodic LoRAs are copied into `models/`; Kohaku exports both ComfyUI and LyCORIS formats. Its checkpoints are under `kohaku/checkpoints/step_XXXXXXXX/`. Keep only latest state is ON by default and removes older completed states from that run after a successful save. LoRA exports and other runs are kept.

Open **Resume / extend from state**, select a completed checkpoint and restore its original settings. Engines cannot share states. Interrupted runs should normally keep the same total limit and training settings. Force stop loses unsaved progress.

sd-scripts can extend a completed run by increasing the total limit. For example, a checkpoint at step 1500 with a new limit of 3000 runs 1500 additional steps. The scheduler LR is recalculated at the saved step on the new curve while optimizer moments and counters are retained. Kohaku permits total-limit changes only for constant; other schedulers must keep the original total.

Changing training conditions requires a new run. sd-scripts can use initial LoRA weights for further training; Kohaku's GUI integration currently supports checkpoint resume rather than initial LoRA weights. It also rejects epoch limits, custom per-module rank/LR, text embedder training, random augmentation and inline sample options. The target selector described above is supported.

## Logs and verification

The footer shows elapsed time, measured average step time, ETA and dedicated GPU memory sampled with nvidia-smi. GPU memory includes other applications. Setup, caching, saving and sampling affect total time. Completion/failure sounds are optional in Run / Log; manual stops are silent. Process logs keep their original language.

The user reported completion with the defaults at the initial v0.2.2 publication on an RTX 4070 12GB on 2026-10-24. Earlier 4000-step Kohaku and 1500-step runs used different settings. CAME updates and restart-scheduler resume were checked on CPU. See [development and verification notes](../CONTRIBUTING.md#検証状況) for scope and historical measurements.

KohakuMoss License 1.0 includes an additional naming condition; the product name is **MOSS Studio with Kohaku**. See [third-party notices](../THIRD_PARTY_NOTICES.md) and the bundled license in `licenses/`.
