# Design principles

1. **A general vision problem.** The task is to map any array of intensities to the orientations of the objects in
   it, and of groups of those objects. Nothing in the model, the targets or the losses is specific to one modality or
   one kind of material. Biological images are the first test, not the scope.

2. **Synthetic data only, with truth by construction.**
   - Every training volume is generated, and every orientation target is computed from the generator's geometry.
   - Real images are used only after training: for validation against an independent measurement, and as visual
     sense checks.
   - Real images never enter training, calibration, hyperparameter choice or loss design.

3. **Learn shape, not appearance.** Contrast, resolution, noise, artefacts and deformation are randomised much more
   widely than any real acquisition (domain randomisation). An image's appearance carries no reliable information; its
   geometry does.

4. **Targets describe the material, not what a reader can see.** A target is the orientation distribution of the
   material inside the query window, from the generator's geometry. It contains no visibility label, no estimate from
   a classical method and no other quantity that depends on how the image was rendered. Where the image is ambiguous,
   the correct output is a broad, honest distribution, and the loss must reward it.

5. **Proper scoring rules.** ODFs are normalised densities over directions, trained with cross-entropy against the
   true distribution. A proper rule is minimised by the true conditional distribution, so the model is never rewarded
   for collapsing to a flat answer, or a confident wrong one, under uncertainty.

6. **No hand-crafted orientation estimators inside the model.** The structure tensor and similar estimators are
   benchmarks to beat. The training configuration (`scripts/train.sh`) uses none as an input, feature, initialisation
   or target.

7. **Established methods.** Domain randomisation (SynthSeg, anatomix), set prediction (DETR), proper scoring rules
   and standard network trunks. Every component must be easy to describe and defend.

8. **One latent, many readouts.** The network runs once per volume. Window, grouping level and ODF order are chosen at
   readout, and the model is trained across all of them.

9. **Separable by object.** A voxel can hold several objects: fibres, vessels, sheets, cells, artefacts. The output
   keeps them apart, so one kind can be extracted on its own, for example only the fibrous part of a voxel's
   orientations.

10. **Simple interfaces.** An array goes in and an array comes out: spherical-harmonic channels per voxel, with a
    population axis for per-population readouts. Chunked and out-of-core processing is a separate layer on top.

11. **Small steps, shown.** Every change is tested with short runs (minutes, not hours) and inspected visually on
    synthetic crops and known-truth phantoms before any longer training.
