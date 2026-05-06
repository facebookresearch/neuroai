REVE
====

| **Config name**: ``REVE``
| **Pretrained checkpoint**: ``brain-bzh/reve-base`` (HuggingFace Hub)
| **Reference**: El Ouahidi et al., *REVE: A Foundation Model for EEG -- Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects*, NeurIPS 2025

REVE is a vision-transformer-style EEG foundation model pretrained via masked
autoencoding on over 60,000 hours of EEG data from 92 datasets spanning
25,000 subjects at 200 Hz.  Its defining feature is a **4D positional encoding**
that jointly encodes the 3-D spatial coordinates of each electrode and the
temporal position of each patch, enabling transfer across arbitrary electrode
configurations without retraining.

Pretraining data overlap
~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   REVE was pretrained on 92 public EEG datasets.  Several of these overlap
   with NeuralBench evaluation datasets, which means results on the affected
   tasks may be inflated due to the model having seen the same *recordings*
   (though not the same *labels*) during self-supervised pretraining.

The known overlaps with NeuralBench evaluation tasks are:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Pretraining dataset
     - NeuralBench task
     - NeuralBench study
     - Notes
   * - TUH (26,847 h)
     - ``pathology``
     - Lopez2017
     - TUH Abnormal EEG subset
   * - TUH
     - ``clinical_event``
     - Harati2015
     - TUH EEG Events subset
   * - TUH
     - ``artifact``
     - Hamid2020
     - TUH EEG Artifact subset
   * - Lee2019_ERP
     - ``p3``
     - Lee2019Erp
     - Via MOABB
   * - Lee2019_SSVEP
     - ``ssvep``
     - Lee2019Ssvep
     - Via MOABB
   * - PhysionetMI
     - ``motor_imagery``
     - Schalk2009
     - PhysioNet eegmmidb
   * - Schirrmeister2017
     - ``motor_execution``
     - Schirrmeister2017
     - Via MOABB
   * - Liu2024
     - ``video``
     - Liu2024
     - Via MOABB
   * - THINGS2 / ON_ds003825
     - ``image``
     - Gifford2022Large
     - THINGS-EEG2 (OpenNeuro ds003825)

Known limitations
~~~~~~~~~~~~~~~~~

* **Pretrained weights require registration** --- Access to the ``brain-bzh``
  checkpoints on HuggingFace requires agreeing to the authors' data usage
  terms.
