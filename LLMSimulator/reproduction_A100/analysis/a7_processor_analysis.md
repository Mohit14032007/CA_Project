# Processor Analysis

A3 -> A4 (Duplex): Attention generation operations shifted from GPU to Logic-PIM.
A4 -> A5 (PE): Operations largely stayed on the same processors as A4, but scheduler rules allowed overlap.
A5 -> A6 (ET): Processor assignments for existing operations remained identical, but ET introduced new communication on the GPU.
