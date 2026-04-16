.. note::
   **Concepts at a glance:**
   ``Study`` → ``Events DataFrame`` → ``Transforms`` → ``Segmenter`` → ``Dataset`` → ``DataLoader``

   Everything stays lightweight (metadata only) until you call the DataLoader.
   Every step is cacheable via ``exca``.
