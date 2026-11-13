# Availability features

The archived implementation uses branch probabilities pv, pa and pt; OCR/STT presence o and s; and log1p character lengths ol and sl. The base feature block is [pv, pa, pt], the flags block is [o, s], and the extended coverage block is [ol, sl, pt*o, pt*s, pv*o, pa*s]. The retained reference implementation is model_and_fusion_reference.py. These execution definitions do not establish when the dictionaries or feature design were originally selected.
