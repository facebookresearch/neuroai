🏆 EEG/EMG Foundation Challenge 2026
=====================================

Starter kit for the `EEG/EMG Foundation Challenge 2026
<https://neural-interfaces26.github.io/>`_, whose four tracks ask one question
each: does your model still work on a new stimulus, a new session, or a new
person? The challenge runs from 21 September to 21 November 2026, and its
winners are announced at the Brain & Body Workshop at NeurIPS 2026.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Track 1 -- EEG-to-Image
      :link: plot_track1_eeg_to_image
      :link-type: doc
      :img-top: https://neural-interfaces26.github.io/exports/eeg-to-image.gif
      :class-card: sd-shadow-sm

      **Cross-stimulus.** Retrieve the viewed image from one EEG epoch,
      against a gallery of images the model never trained on.

   .. grid-item-card:: Track 2 -- BCI decoding
      :link: plot_track2_eeg_to_bci
      :link-type: doc
      :img-top: https://neural-interfaces26.github.io/exports/bci-decoding.gif
      :class-card: sd-shadow-sm

      **Cross-session.** Decode one of three cued mental commands on a day
      the user was not calibrated on.

   .. grid-item-card:: Track 3 -- Sleep onset
      :link: plot_track3_sleep_onset
      :link-type: doc
      :img-top: https://neural-interfaces26.github.io/exports/sleep-onset.gif
      :class-card: sd-shadow-sm

      **Cross-user.** Predict seconds to first stable N2 from wearable EEG,
      on sleepers never seen in training.

   .. grid-item-card:: Track 4 -- EMG-to-Pose
      :link: plot_track4_emg_to_pose
      :link-type: doc
      :img-top: https://neural-interfaces26.github.io/exports/emg-to-pose.gif
      :class-card: sd-shadow-sm

      **Cross-user.** Regress 20 hand-joint angles from wrist sEMG, on new
      users and new movement stages.

Read the :doc:`overview <plot_overview>` first if you are new here: it explains
what NeuralBench is, how it relates to the challenge, and what the baseline
numbers are. Then jump to the track you plan to submit to. The `competition
website <https://neural-interfaces26.github.io/>`_ owns registration, the
submission window, the rules, and the prizes.

.. note::
   These pages assume ``neuralbench`` is already installed and
   configured. See :doc:`/neuralbench/install` and the
   :doc:`quickstart </neuralbench/auto_examples/quickstart/01_run_first_task>`
   if you have not run a task yet.

Found a bug in these pages or in ``neuralbench``, or have feedback on either?
Please `open an issue or a pull request
<https://github.com/facebookresearch/neuroai/issues>`_ on the ``neuroai``
repository.
