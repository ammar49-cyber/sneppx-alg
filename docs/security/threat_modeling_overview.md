# Threat Modeling Overview

This directory contains threat modeling artifacts for the SneppX-ALG platform.

## Methodology

SneppX-ALG uses the following threat modeling methodologies:

1. **STRIDE** (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege)
2. **PASTA** (Process for Attack Simulation and Threat Analysis)
3. **Attack Trees** - Hierarchical representation of attacker goals and sub-goals
4. **DREAD** (Damage, Reproducibility, Exploitability, Affected Users, Discoverability)
5. **LINDUN** (Linkability, Identifiability, Non-repudiation, Detectability, Disclosure of information, Unawareness, Non-compliance)

## Process

1. Define system scope and boundaries
2. Identify assets and trust boundaries
3. Enumerate threats (STRIDE per component)
4. Rank threats (DREAD)
5. Document mitigations
6. Validate through testing

## Directory Structure

- `data_flow_diagrams/` - DFDs for each major subsystem
- `attack_trees/` - Attack tree diagrams
- `risk_assessments/` - Risk scoring matrices
- `mitigation_strategies/` - Documented countermeasures
- `review_minutes/` - Threat modeling session records

## Standard Templates

Each threat model includes:
- Component name and version
- Trust boundaries identified
- List of threats with STRIDE category
- DREAD scores (0-10 each)
- Overall risk rating
- Mitigation status (Mitigated/Accepted/Transferred/Open)
- Verification evidence
