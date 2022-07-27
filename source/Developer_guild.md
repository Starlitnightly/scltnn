# Developer guild

> **Note**
>
> To better understand the following guide, you may check out our [publication]() first to learn about the general idea.

Below we describe main components of the framework, and how to extend the existing implementations.

## Main components

A LTNN model is primarily composed of 3 components (all by Keras).

- ANN-A framework: to produce a pre-trainning model by existing scVelo latent time dataset and help researchers to determined the start and end node.
- ANN-B framework: to regress the pseudo time by the norm distribution of start and end node
- PAGA graph: to produce the pseudotime by PAGA trajectory

Current implementation for these components are all located in scltnn.models.sc. New extensions can be added to this module as well.