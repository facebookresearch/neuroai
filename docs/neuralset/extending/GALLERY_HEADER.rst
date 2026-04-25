# Extending NeuralSet

NeuralSet is modular by design: every component — Study, Event, Transform, Extractor — is an independent Pydantic model that you can subclass locally and still benefit from typed configuration, caching, and pipeline composition. See :doc:`/neuralset/philosophy` for the design rationale.

Each page below shows a working example of building a custom component. Start with the component type that matches your use case.